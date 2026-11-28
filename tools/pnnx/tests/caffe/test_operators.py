# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess

import torch
import torch.nn.functional as F

from test_utils import blob, compare, inference, input_layer, layer, run


def caffe_pool(x, average, kernel=3, stride=2, pad=1):
    # independent Caffe window definition. Include declared padding in the
    # average divisor, but exclude the extra cells from the final ceil window
    height, width = x.shape[-2:]
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    oh = math.ceil((height + 2 * ph - kh) / sh) + 1
    ow = math.ceil((width + 2 * pw - kw) / sw) + 1
    if (ph or pw) and (oh - 1) * sh >= height + ph:
        oh -= 1
    if (ph or pw) and (ow - 1) * sw >= width + pw:
        ow -= 1
    result = torch.empty(*x.shape[:2], oh, ow)
    for h in range(oh):
        for w in range(ow):
            hs, ws = h * sh - ph, w * sw - pw
            he, we = min(hs + kh, height + ph), min(ws + kw, width + pw)
            divisor = (he - hs) * (we - ws)
            window = x[..., max(hs, 0):min(he, height), max(ws, 0):min(we, width)]
            if not average and window.numel() == 0:
                result[..., h, w] = -torch.finfo(x.dtype).max
            else:
                result[..., h, w] = window.sum((-2, -1)) / divisor if average else window.amax((-2, -1))
    return result


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    schema = Path(args.schema).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cases = []
    x = torch.rand(1, 4, 7, 8) * 2 - 1
    for kind, groups in [('Convolution', 1), ('Convolution', 2), ('ConvolutionDepthwise', 4), ('DepthwiseConvolution', 4)]:
        weight = torch.rand(4, 4 // groups, 3, 3) - .5
        bias = torch.rand(4) - .5
        params = 'convolution_param { num_output: 4 kernel_size: 3 pad: 1 group: %d }' % groups
        proto = input_layer(x.shape) + layer('conv', kind, ['data'], ['conv'], params)
        proto += layer('relu', 'ReLU', ['conv'], ['conv'])
        proto += layer('cat', 'Concat', ['conv', 'data'], ['out'], 'concat_param { axis: 1 }')
        weights = layer('conv', kind, [], [], blob(weight) + blob(bias))
        reference = torch.cat([F.conv2d(x, weight, bias, padding=1, groups=groups).relu(), x], 1)
        cases.append((kind + str(groups), proto, weights, x, [reference], []))
    for groups in (1, 2):
        weight = torch.rand(4, 6 // groups, 3, 3) - .5
        bias = torch.rand(6) - .5
        proto = input_layer(x.shape) + layer('deconv', 'Deconvolution', ['data'], ['out'],
            'convolution_param { num_output: 6 kernel_size: 3 stride: 2 pad: 1 group: %d }' % groups)
        weights = layer('deconv', 'Deconvolution', [], [], blob(weight) + blob(bias))
        cases.append(('deconv%d' % groups, proto, weights, x,
                      [F.conv_transpose2d(x, weight, bias, stride=2, padding=1, groups=groups)], []))
    for groups in (1, 2):
        for dilation in (1, 2):
            weight = torch.rand(4, 6 // groups, 2, 3) - .5
            proto = input_layer(x.shape) + layer('deconv', 'Deconvolution', ['data'], ['out'],
                'convolution_param { num_output: 6 kernel_h: 2 kernel_w: 3 stride_h: 2 stride_w: 1 '
                'pad_h: 1 pad_w: 1 bias_term: false group: %d dilation: %d }' % (groups, dilation))
            weights = layer('deconv', 'Deconvolution', [], [], blob(weight))
            reference = F.conv_transpose2d(x, weight, stride=(2, 1), padding=1, groups=groups, dilation=dilation)
            cases.append(('deconv_rect_%d_%d' % (groups, dilation), proto, weights, x, [reference], []))
    weight = torch.rand(4, 6, 3, 3) - .5
    bias = torch.rand(6) - .5
    proto = input_layer(x.shape) + layer('deconv', 'Deconvolution', ['data'], ['out'],
        'convolution_param { num_output: 6 kernel_size: 3 }')
    weights = layer('deconv', 'Deconvolution', [], [], blob(weight) + blob(bias))
    cases.append(('deconv_defaults', proto, weights, x, [F.conv_transpose2d(x, weight, bias)], []))
    weight = torch.rand(4, 6, 2, 3) - .5
    proto = input_layer(x.shape) + layer('deconv', 'Deconvolution', ['data'], ['out'],
        'convolution_param { num_output: 6 kernel_h: 2 kernel_w: 3 pad_h: 0 pad_w: 1 bias_term: false }')
    weights = layer('deconv', 'Deconvolution', [], [], blob(weight))
    cases.append(('deconv_unequal_padding', proto, weights, x,
                  [F.conv_transpose2d(x, weight, padding=(0, 1))], []))
    weight = torch.rand(4, 1, 3, 3) - .5
    bias = torch.rand(4) - .5
    proto = input_layer(x.shape) + layer('deconv', 'Deconvolution', ['data'], ['out'],
        'convolution_param { num_output: 4 kernel_size: 3 group: 4 }')
    weights = layer('deconv', 'Deconvolution', [], [], blob(weight) + blob(bias))
    cases.append(('deconv_depthwise', proto, weights, x,
                  [F.conv_transpose2d(x, weight, bias, groups=4)], []))
    for shared in (False, True):
        slope = torch.rand(1 if shared else 4)
        proto = input_layer(x.shape) + layer('prelu', 'PReLU', ['data'], ['out'],
            'prelu_param { channel_shared: %s }' % str(shared).lower())
        cases.append(('prelu%d' % shared, proto, layer('prelu', 'PReLU', [], [], blob(slope)),
                      x, [F.prelu(x, slope)], []))
    scale = torch.rand(4)
    proto = input_layer(x.shape) + layer('normalize', 'Normalize', ['data'], ['out'],
        'norm_param { across_spatial: false channel_shared: false eps: 0.001 }')
    reference = x / torch.sqrt(x.square().sum(1, keepdim=True) + .001) * scale.reshape(1, 4, 1, 1)
    cases.append(('normalize', proto, layer('normalize', 'Normalize', [], [], blob(scale)), x, [reference], []))
    for across in (False, True):
        for shared in (False, True):
            scale = torch.rand(1 if shared else x.shape[1])
            dims = (1, 2, 3) if across else (1,)
            params = 'norm_param { across_spatial: %s channel_shared: %s eps: 0.001 }' % (
                str(across).lower(), str(shared).lower())
            proto = input_layer(x.shape) + layer('normalize', 'Normalize', ['data'], ['out'], params)
            reference = x / torch.sqrt(x.square().sum(dims, keepdim=True) + .001)
            reference *= scale.reshape(1, -1, 1, 1)
            cases.append(('normalize_%d_%d' % (across, shared), proto,
                layer('normalize', 'Normalize', [], [], blob(scale)), x, [reference], []))
        for variance in (False, True):
            dims = (1, 2, 3) if across else (2, 3)
            params = 'mvn_param { across_channels: %s normalize_variance: %s eps: 0.001 }' % (
                str(across).lower(), str(variance).lower())
            proto = input_layer(x.shape) + layer('mvn', 'MVN', ['data'], ['out'], params)
            reference = x - x.mean(dims, keepdim=True)
            if variance:
                reference /= x.var(dims, unbiased=False, keepdim=True).sqrt() + .001
            cases.append(('mvn_%d_%d' % (across, variance), proto, '', x, [reference], []))
    for shape in ((1, 4), (1, 4, 7)):
        value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 7 - 1
        dims = tuple(range(1, len(shape)))
        for scale_value in (-.75, 1.25):
            params = 'norm_param { across_spatial: true channel_shared: true eps: 0.001 }'
            proto = input_layer(shape) + layer('normalize', 'Normalize', ['data'], ['out'], params)
            scale = torch.tensor([scale_value])
            reference = value / torch.sqrt(value.square().sum(dims, keepdim=True) + .001) * scale_value
            weights = layer('normalize', 'Normalize', [], [], blob(scale))
            cases.append(('normalize_rank%d_scale%s' % (len(shape), scale_value), proto, weights, value, [reference], []))
        for variance in (False, True):
            params = 'mvn_param { across_channels: true normalize_variance: %s eps: 0.001 }' % str(variance).lower()
            proto = input_layer(shape) + layer('mvn', 'MVN', ['data'], ['out'], params)
            reference = value - value.mean(dims, keepdim=True)
            if variance:
                reference /= value.var(dims, unbiased=False, keepdim=True).sqrt() + .001
            cases.append(('mvn_rank%d_variance%d' % (len(shape), variance), proto, '', value, [reference], []))
    weight = torch.rand(6, 4, 2, 3) - .5
    proto = input_layer(x.shape) + layer('conv', 'Convolution', ['data'], ['out'],
        'convolution_param { num_output: 6 kernel_h: 2 kernel_w: 3 stride_h: 2 stride_w: 1 pad_h: 1 pad_w: 2 dilation: 2 bias_term: false }')
    cases.append(('conv_rect_dilation', proto, layer('conv', 'Convolution', [], [], blob(weight)), x,
                  [F.conv2d(x, weight, stride=(2, 1), padding=(1, 2), dilation=2)], []))
    for axis, scale in [(0, torch.tensor(.7)), (2, torch.rand(7, 8))]:
        num_axes = 0 if axis == 0 else 2
        proto = input_layer(x.shape) + layer('scale', 'Scale', ['data'], ['out'],
            'scale_param { axis: %d num_axes: %d bias_term: false }' % (axis, num_axes))
        weights = layer('scale', 'Scale', [], [], blob(scale))
        cases.append(('scale_axis%d' % axis, proto, weights, x, [x * scale], []))
    for name, shape, bias_term in [('scale_channel_defaults', (1, 4, 7, 8), False),
                                    ('scale_channel_bias', (1, 4, 7, 8), True),
                                    ('scale_rank2', (1, 4), True),
                                    ('scale_rank3', (1, 4, 7), False)]:
        value = torch.rand(shape) * 2 - 1
        scale = torch.rand(4)
        bias = torch.rand(4) - .5
        params = 'scale_param { bias_term: true }' if bias_term else ''
        proto = input_layer(shape) + layer('scale', 'Scale', ['data'], ['out'], params)
        weights = layer('scale', 'Scale', [], [], blob(scale) + (blob(bias) if bias_term else ''))
        broadcast_shape = (1, 4) + (1,) * (len(shape) - 2)
        reference = value * scale.reshape(broadcast_shape)
        if bias_term:
            reference += bias.reshape(broadcast_shape)
        cases.append((name, proto, weights, value, [reference], []))
    scale = torch.rand(4)
    proto = input_layer(x.shape) + layer('scale_input', 'Input', [], ['scale'], 'input_param { shape { dim: 4 } }')
    proto += layer('scale', 'Scale', ['data', 'scale'], ['out'], 'scale_param { axis: 1 bias_term: false }')
    # Scale's parameter tensor has no batch dimension. The generic runner below
    # accepts per-case input batch-axis declarations for such auxiliary inputs
    cases.append(('scale_second_input', proto, '', [x, scale], [x * scale.reshape(1, 4, 1, 1)], []))
    for dynamic in (False, True):
        for bias_term in (False, True):
            scale = torch.rand(7, 8)
            bias = torch.rand(7, 8) - .5
            proto = input_layer(x.shape)
            bottom = ['data']
            if dynamic:
                proto += layer('scale_input', 'Input', [], ['scale'],
                    'input_param { shape { dim: 7 dim: 8 } }')
                bottom.append('scale')
            proto += layer('scale', 'Scale', bottom, ['out'],
                'scale_param { axis: -2 num_axes: -1 bias_term: %s }' % str(bias_term).lower())
            blobs = '' if dynamic else blob(scale)
            if bias_term:
                blobs += blob(bias)
            weights = layer('scale', 'Scale', [], [], blobs) if blobs else ''
            name = 'scale_%s_spatial_bias%d' % ('second_input' if dynamic else 'static', bias_term)
            reference = x * scale + bias if bias_term else x * scale
            cases.append((name, proto, weights, [x, scale] if dynamic else x, [reference], []))
    for average in (False, True):
        for shape in ((1, 4, 7, 8), (1, 4, 8, 7)):
            value = torch.rand(shape) * 2 - 1
            proto = input_layer(shape) + layer('pool', 'Pooling', ['data'], ['out'],
                'pooling_param { pool: %s kernel_size: 3 stride: 2 pad: 1 }' % ('AVE' if average else 'MAX'))
            cases.append(('pool%d_%d' % (average, shape[-1]), proto, '', value, [caffe_pool(value, average)], []))
    value = torch.rand(1, 4, 5, 6) * 2 - 1
    proto = input_layer(value.shape) + layer('pool', 'Pooling', ['data'], ['out'],
        'pooling_param { pool: MAX kernel_size: 3 stride: 2 pad: 2 }')
    cases.append(('pool_large_pad', proto, '', value, [caffe_pool(value, False, pad=2)], []))
    # With stride larger than the kernel, Caffe retains empty trailing windows
    # when pad=0. Exercise both the values and a shape-sensitive consumer.
    for shape in ((1, 1, 4, 4), (1, 2, 4, 5), (1, 2, 5, 4)):
        value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        reference = caffe_pool(value, False, kernel=1, stride=2, pad=0)
        for flatten in (False, True):
            proto = input_layer(shape) + layer('pool', 'Pooling', ['data'], ['pooled' if flatten else 'out'],
                'pooling_param { pool: MAX kernel_size: 1 stride: 2 }')
            if flatten:
                proto += layer('flatten', 'Flatten', ['pooled'], ['out'])
            name = 'pool_empty_tail_%dx%d_flatten%d' % (shape[-2], shape[-1], flatten)
            cases.append((name, proto, '', value, [reference.flatten(1) if flatten else reference], []))
    for average in (False, True):
        proto = input_layer(x.shape) + layer('pool', 'Pooling', ['data'], ['out'],
            'pooling_param { pool: %s kernel_h: 2 kernel_w: 3 stride_h: 2 stride_w: 3 pad_h: 0 pad_w: 1 }' % ('AVE' if average else 'MAX'))
        cases.append(('pool_rect%d' % average, proto, '', x,
            [caffe_pool(x, average, kernel=(2, 3), stride=(2, 3), pad=(0, 1))], []))
    for average in (False, True):
        value = torch.rand(1, 4, 8, 8) * 2 - 1
        proto = input_layer(value.shape) + layer('pool', 'Pooling', ['data'], ['out'],
            'pooling_param { pool: %s kernel_h: 2 kernel_w: 3 stride_h: 5 stride_w: 3 pad_h: 0 pad_w: 1 }' % ('AVE' if average else 'MAX'))
        cases.append(('pool_asymmetric_clip%d' % average, proto, '', value,
            [caffe_pool(value, average, kernel=(2, 3), stride=(5, 3), pad=(0, 1))], []))
    for transpose in (False, True):
        weight = torch.rand(5, x[0].numel()) - .5
        bias = torch.rand(5) - .5
        proto = input_layer(x.shape) + layer('linear', 'InnerProduct', ['data'], ['out'],
            'inner_product_param { num_output: 5 axis: 1 transpose: %s }' % str(transpose).lower())
        weights = layer('linear', 'InnerProduct', [], [], blob(weight.t().contiguous() if transpose else weight) + blob(bias))
        cases.append(('linear%d' % transpose, proto, weights, x, [F.linear(x.flatten(1), weight, bias)], []))
    for axis in (2, -1):
        start = axis if axis >= 0 else x.ndim + axis
        value = x.flatten(start)
        for transpose in (False, True):
            weight = torch.rand(5, value.shape[-1]) - .5
            proto = input_layer(x.shape) + layer('linear', 'InnerProduct', ['data'], ['out'],
                'inner_product_param { num_output: 5 axis: %d transpose: %s bias_term: false }' % (axis, str(transpose).lower()))
            weights = layer('linear', 'InnerProduct', [], [], blob(weight.t().contiguous() if transpose else weight))
            cases.append(('linear_axis%d_transpose%d' % (axis, transpose), proto, weights, x,
                [F.linear(value, weight)], []))
    mean, variance = torch.rand(4), torch.rand(4) + .5
    scale, bias = torch.rand(4), torch.rand(4)
    proto = input_layer(x.shape) + layer('bn', 'BatchNorm', ['data'], ['bn'], 'batch_norm_param { eps: 0.001 }')
    proto += layer('scale', 'Scale', ['bn'], ['out'], 'scale_param { axis: 1 num_axes: 1 bias_term: true }')
    for factor in (0., 2.):
        weights = layer('bn', 'BatchNorm', [], [], blob(mean) + blob(variance) + blob(torch.tensor([factor])))
        weights += layer('scale', 'Scale', [], [], blob(scale) + blob(bias))
        multiplier = 0. if factor == 0 else 1. / factor
        ref = (x - (mean * multiplier).reshape(1, 4, 1, 1)) / torch.sqrt((variance * multiplier).reshape(1, 4, 1, 1) + .001)
        ref = ref * scale.reshape(1, 4, 1, 1) + bias.reshape(1, 4, 1, 1)
        cases.append(('bn_scale%d' % factor, proto, weights, x, [ref], []))
    weight = torch.rand(4, 4, 3, 3) - .5
    conv_bias = torch.rand(4) - .5
    proto = input_layer(x.shape) + layer('conv', 'Convolution', ['data'], ['conv'],
        'convolution_param { num_output: 4 kernel_size: 3 pad: 1 bias_term: true }')
    proto += layer('bn', 'BatchNorm', ['conv'], ['bn'], 'batch_norm_param { eps: 0.001 }')
    proto += layer('scale', 'Scale', ['bn'], ['out'], 'scale_param { axis: 1 bias_term: true }')
    weights = layer('conv', 'Convolution', [], [], blob(weight) + blob(conv_bias))
    weights += layer('bn', 'BatchNorm', [], [], blob(mean) + blob(variance) + blob(torch.tensor([2.])))
    weights += layer('scale', 'Scale', [], [], blob(scale) + blob(bias))
    ref = (F.conv2d(x, weight, conv_bias, padding=1) - (mean / 2).reshape(1, 4, 1, 1)) / torch.sqrt((variance / 2).reshape(1, 4, 1, 1) + .001)
    ref = ref * scale.reshape(1, 4, 1, 1) + bias.reshape(1, 4, 1, 1)
    cases.append(('conv_bn_scale', proto, weights, x, [ref], []))
    proto = input_layer(x.shape) + layer('bn', 'BN', ['data'], ['out'])
    weights = layer('bn', 'BN', [], [], blob(scale) + blob(bias))
    cases.append(('bn_affine', proto, weights, x, [x * scale.reshape(1, 4, 1, 1) + bias.reshape(1, 4, 1, 1)], []))
    for shape in ((1, 4), (1, 4, 7)):
        value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) / 7 - 1
        affine_scale = torch.tensor([-1.5, 0., .25, 2.])
        affine_bias = torch.tensor([.5, -.75, 1., -.25])
        broadcast_shape = (1, 4) + (1,) * (len(shape) - 2)
        proto = input_layer(shape) + layer('bn', 'BN', ['data'], ['out'])
        weights = layer('bn', 'BN', [], [], blob(affine_scale) + blob(affine_bias))
        ref = value * affine_scale.reshape(broadcast_shape) + affine_bias.reshape(broadcast_shape)
        cases.append(('bn_affine_rank%d' % len(shape), proto, weights, value, [ref], []))
    # shape override must affect execution, including an inplace write and a
    # branch that still reads the original input
    proto = input_layer((1, 4, 3, 3)) + layer('relu', 'ReLU', ['data'], ['relu'])
    proto += layer('leaky', 'ReLU', ['relu'], ['relu'], 'relu_param { negative_slope: 0.1 }')
    proto += layer('cat', 'Concat', ['data', 'relu'], ['out'], 'concat_param { axis: 1 }')
    cases.append(('shape_override', proto, '', x, [torch.cat([x, x.relu()], 1)], ['inputshape=[1,4,7,8]']))
    proto = 'input: "data" input_shape { dim: 1 dim: 4 dim: 7 dim: 8 }\n'
    proto += layer('dropout', 'Dropout', ['data'], ['out'], 'dropout_param { dropout_ratio: 0.75 }')
    # invalid training-only connections must not be imported into the TEST graph
    proto += layer('training', 'NotACaffeOperator', ['training_data'], ['training_output'], 'include { phase: TRAIN }')
    cases.append(('net_input_test_phase', proto, '', x, [x], []))
    proto = layer('memory', 'MemoryData', [], ['data'],
        'memory_data_param { batch_size: 1 channels: 4 height: 7 width: 8 }')
    proto += layer('relu', 'ReLU', ['data'], ['out'])
    cases.append(('memory_input', proto, '', x, [x.relu()], []))
    proto = input_layer(x.shape) + layer('relu', 'ReLU', ['data'], ['relu'])
    proto += layer('sum', 'Eltwise', ['data', 'relu'], ['out'], 'eltwise_param { operation: SUM coeff: 0.25 coeff: 1.5 }')
    cases.append(('eltwise', proto, '', x, [x * .25 + x.relu() * 1.5], []))
    for operation, ref in [('PROD', x * x.relu()), ('MAX', torch.maximum(x, x.relu()))]:
        proto = input_layer(x.shape) + layer('relu', 'ReLU', ['data'], ['relu'])
        proto += layer('op', 'Eltwise', ['data', 'relu'], ['out'], 'eltwise_param { operation: %s }' % operation)
        cases.append(('eltwise_' + operation, proto, '', x, [ref], []))
    for operation, params, reference in [
        ('SUM', 'coeff: 0.25 coeff: -0.5 coeff: 1.5', (x * .25 + x.relu() * -.5) + (-x) * 1.5),
        ('PROD', '', (x * x.relu()) * (-x)),
        ('MAX', '', torch.maximum(torch.maximum(x, x.relu()), -x))]:
        proto = input_layer(x.shape) + layer('relu', 'ReLU', ['data'], ['relu'])
        proto += layer('negative', 'Power', ['data'], ['negative'], 'power_param { scale: -1 }')
        proto += layer('eltwise', 'Eltwise', ['data', 'relu', 'negative'], ['out'],
            'eltwise_param { operation: %s %s }' % (operation, params))
        cases.append(('eltwise_three_' + operation, proto, '', x, [reference], []))
    for operation, params, reference in [
        ('SUM', 'coeff: -0.5', x * -.5), ('PROD', '', x), ('MAX', '', x)]:
        proto = input_layer(x.shape) + layer('eltwise', 'Eltwise', ['data'], ['out'],
            'eltwise_param { operation: %s %s }' % (operation, params))
        cases.append(('eltwise_single_' + operation, proto, '', x, [reference], []))
    # source arithmetic changes type before downstream shape preparation
    # exercise that boundary with independent bounded inputs and a spatial head
    first = torch.linspace(.1, .8, 4 * 6 * 8).reshape(1, 4, 6, 8)
    second = torch.linspace(.9, .2, 4 * 6 * 8).reshape(1, 4, 6, 8)
    proto = input_layer(first.shape)
    proto += layer('second_input', 'Input', [], ['second'],
                   'input_param { shape { dim: 1 dim: 4 dim: 6 dim: 8 } }')
    proto += layer('weighted', 'Eltwise', ['data', 'second'], ['weighted'],
                   'eltwise_param { operation: SUM coeff: 0.25 coeff: 0.75 }')
    proto += layer('exp', 'Exp', ['weighted'], ['exponential'],
                   'exp_param { base: 2 scale: 0.5 shift: 0.25 }')
    proto += layer('log', 'Log', ['exponential'], ['logarithm'],
                   'log_param { base: 2 scale: 0.5 shift: 1 }')
    proto += layer('pool', 'Pooling', ['logarithm'], ['pooled'],
                   'pooling_param { pool: AVE kernel_size: 2 stride: 2 }')
    proto += layer('flatten', 'Flatten', ['pooled'], ['out'], 'flatten_param { axis: 1 }')
    weighted = first * .25 + second * .75
    exponential = torch.pow(2., weighted * .5 + .25)
    logarithm = torch.log2(exponential * .5 + 1.)
    expected = F.avg_pool2d(logarithm, 2, 2).flatten(1)
    cases.append(('arithmetic_pool_flatten', proto, '', [first, second], [expected], []))
    # repeated source rewrites must preserve inplace versions and branch consumers
    value = torch.linspace(.1, .9, 48).reshape(1, 2, 4, 6)
    proto = input_layer(value.shape)
    proto += layer('exp_first', 'Exp', ['data'], ['stage'], 'exp_param { base: 2 scale: 0.5 }')
    proto += layer('exp_second', 'Exp', ['stage'], ['stage'], 'exp_param { base: 0.5 scale: 0.25 shift: 0.1 }')
    proto += layer('log_first', 'Log', ['stage'], ['logged'], 'log_param { scale: 0.5 shift: 1 }')
    proto += layer('log_second', 'Log', ['logged'], ['logged'], 'log_param { base: 2 shift: 1 }')
    proto += layer('square', 'Power', ['logged'], ['powered'], 'power_param { power: 2 scale: 0.5 shift: 1 }')
    proto += layer('cube', 'Power', ['powered'], ['powered'], 'power_param { power: 3 scale: 0.25 shift: 0.5 }')
    proto += layer('merge', 'Eltwise', ['powered', 'stage'], ['out'],
                   'eltwise_param { operation: SUM coeff: 0.25 coeff: 0.75 }')
    stage = torch.pow(.5, torch.pow(2., value * .5) * .25 + .1)
    logged = torch.log2(torch.log(stage * .5 + 1.) + 1.)
    powered = ((logged * .5 + 1.).square() * .25 + .5).pow(3)
    cases.append(('arithmetic_repeated_inplace_branch', proto, '', value, [powered * .25 + stage * .75], []))
    # distinct matches of stable softplus and comparison retain floating output
    value = torch.linspace(-20., 20., 48).reshape(1, 2, 4, 6)
    proto = input_layer(value.shape)
    proto += layer('softplus_first', 'BNLL', ['data'], ['soft'])
    proto += layer('softplus_second', 'BNLL', ['soft'], ['soft'])
    proto += layer('threshold_first', 'Threshold', ['soft'], ['mask'], 'threshold_param { threshold: 1 }')
    proto += layer('threshold_second', 'Threshold', ['mask'], ['mask'], 'threshold_param { threshold: 0.5 }')
    proto += layer('multiply', 'Eltwise', ['soft', 'mask'], ['masked'], 'eltwise_param { operation: PROD }')
    proto += layer('flatten', 'Flatten', ['masked'], ['out'], 'flatten_param { axis: 1 }')
    soft = F.softplus(F.softplus(value))
    mask = ((soft > 1.).float() > .5).float()
    cases.append(('arithmetic_repeated_softplus_threshold', proto, '', value, [(soft * mask).flatten(1)], []))
    proto = input_layer(x.shape) + layer('softmax', 'Softmax', ['data'], ['out'], 'softmax_param { axis: -1 }')
    cases.append(('softmax_negative_axis', proto, '', x, [F.softmax(x, dim=-1)], []))
    # exercise omitted parameter messages and explicit source-only fields in
    # level2 rewrites. Negative inputs distinguish leaky ReLU from ReLU
    for name, kind, params, ref in [
        ('relu_default', 'ReLU', '', F.relu(x)),
        ('relu_zero_slope', 'ReLU', 'relu_param { negative_slope: 0 engine: CAFFE }', F.relu(x)),
        ('relu_leaky', 'ReLU', 'relu_param { negative_slope: 0.2 engine: CAFFE }', F.leaky_relu(x, .2)),
        ('elu_default', 'ELU', '', F.elu(x)),
        ('softmax_default', 'Softmax', '', F.softmax(x, dim=1)),
        ('softmax_engine', 'Softmax', 'softmax_param { axis: 2 engine: CAFFE }', F.softmax(x, dim=2)),
        ('softmax_parameter_scope', 'Softmax',
         'softmax_param { axis: 2 } concat_param { axis: 1 } slice_param { slice_dim: 3 }', F.softmax(x, dim=2)),
        ('sigmoid_engine', 'Sigmoid', 'sigmoid_param { engine: CAFFE }', x.sigmoid()),
        ('tanh_engine', 'TanH', 'tanh_param { engine: CAFFE }', x.tanh())]:
        proto = input_layer(x.shape) + layer('activation', kind, ['data'], ['out'], params)
        cases.append((name, proto, '', x, [ref], []))
    proto = input_layer(x.shape) + layer('slice', 'Slice', ['data'], ['a', 'b'], 'slice_param { axis: 1 slice_point: 1 }')
    cases.append(('slice_outputs', proto, '', x, [x[:, :1], x[:, 1:]], []))
    proto = input_layer(x.shape) + layer('slice', 'Slice', ['data'], ['a', 'b', 'c'],
        'slice_param { axis: 1 slice_point: 1 slice_point: 3 }')
    proto += layer('relu', 'ReLU', ['b'], ['b'])
    cases.append(('slice_three_inplace', proto, '', x, [x[:, :1], x[:, 3:], x[:, 1:3].relu()], []))
    proto = input_layer(x.shape) + layer('split', 'Split', ['data'], ['a', 'b'])
    cases.append(('split_outputs', proto, '', x, [x, x], []))
    proto = input_layer(x.shape) + layer('slice', 'Slice', ['data'], ['a', 'b'], 'slice_param { axis: 1 }')
    proto += layer('silence', 'Silence', ['a'], [])
    cases.append(('silence', proto, '', x, [x[:, 2:]], []))
    for kind, params, ref in [
        ('Power', 'power_param { power: 2 scale: 0.5 shift: 1 }', (1 + .5 * x).square()),
        ('ELU', 'elu_param { alpha: 0.3 }', F.elu(x, .3)),
        ('ReLU6', '', F.relu6(x)),
        ('AbsVal', '', x.abs()),
        ('Sigmoid', '', x.sigmoid()),
        ('TanH', '', x.tanh()),
        ('BNLL', '', F.softplus(x)),
        ('Threshold', 'threshold_param { threshold: 0.2 }', (x > .2).float()),
        ('Interp', 'interp_param { height: 11 width: 13 }', F.interpolate(x, size=(11, 13), mode='bilinear', align_corners=True)),
        ('ShuffleChannel', 'shuffle_channel_param { group: 2 }', x.reshape(1, 2, 2, 7, 8).transpose(1, 2).reshape_as(x)),
        ('MVN', 'mvn_param { normalize_variance: true across_channels: false eps: 0.001 }',
         (x - x.mean((-2, -1), keepdim=True)) / (x.var((-2, -1), unbiased=False, keepdim=True).sqrt() + .001)),
        ('LRN', 'lrn_param { local_size: 3 alpha: 0.2 beta: 0.75 norm_region: ACROSS_CHANNELS }',
         F.local_response_norm(x, 3, alpha=.2, beta=.75, k=1.))]:
        proto = input_layer(x.shape) + layer('op', kind, ['data'], ['out'], params)
        cases.append((kind, proto, '', x, [ref], []))
    for power in (1, 2, 3, -1):
        value = torch.linspace(.25, 2., x.numel()).reshape_as(x)
        value[..., ::2] *= -1
        proto = input_layer(value.shape) + layer('power', 'Power', ['data'], ['out'],
            'power_param { power: %d }' % power)
        cases.append(('power_signed_%d' % power, proto, '', value, [value.pow(power)], []))
    for axis, field in [(2, 'axis'), (-1, 'axis'), (2, 'concat_dim')]:
        proto = input_layer(x.shape) + layer('relu', 'ReLU', ['data'], ['relu'])
        proto += layer('cat', 'Concat', ['data', 'relu'], ['out'],
            'concat_param { %s: %d }' % (field, axis))
        cases.append(('concat_%s_%d' % (field, axis), proto, '', x,
            [torch.cat([x, x.relu()], axis)], []))
    for axis, field in [(2, 'axis'), (-1, 'axis'), (2, 'slice_dim')]:
        proto = input_layer(x.shape) + layer('slice', 'Slice', ['data'], ['a', 'b'],
            'slice_param { %s: %d slice_point: 2 }' % (field, axis))
        reference = list(torch.split(x, [2, x.shape[axis] - 2], dim=axis))
        cases.append(('slice_%s_%d' % (field, axis), proto, '', x, reference, []))
    for scale_train in (False, True):
        for ratio in (0., .25, .75):
            params = 'dropout_param { dropout_ratio: %s scale_train: %s }' % (ratio, str(scale_train).lower())
            proto = input_layer(x.shape) + layer('dropout', 'Dropout', ['data'], ['out'], params)
            reference = x if scale_train else x * (1. - ratio)
            cases.append(('dropout_%d_%s' % (scale_train, ratio), proto, '', x, [reference], []))
    for axis, tiles in [(0, 1), (1, 1), (1, 3), (2, 2), (-1, 2)]:
        proto = input_layer(x.shape) + layer('op', 'Tile', ['data'], ['out'],
            'tile_param { axis: %d tiles: %d }' % (axis, tiles))
        sizes = [1] * x.ndim
        sizes[axis] = tiles
        cases.append(('tile_%d_%d' % (axis, tiles), proto, '', x, [x.repeat(*sizes)], []))
    for kind in ['Exp', 'Log']:
        value = x.abs() + .25 if kind == 'Log' else x
        for base in [-1., .5, 2.]:
            for scale, shift in [(1., 0.), (.5, .75)]:
                params = '%s_param { base: %s scale: %s shift: %s }' % (kind.lower(), base, scale, shift)
                proto = input_layer(value.shape) + layer('op', kind, ['data'], ['out'], params)
                ref = value * scale + shift
                if kind == 'Exp':
                    ref = ref.exp() if base == -1 else base ** ref
                else:
                    ref = ref.log() if base == -1 else ref.log() / torch.tensor(base).log()
                cases.append(('%s_%s_%s' % (kind, base, scale), proto, '', value, [ref], []))
    bnll_input = torch.linspace(-100, 100, x.numel()).reshape_as(x)
    proto = input_layer(x.shape) + layer('op', 'BNLL', ['data'], ['out'])
    cases.append(('BNLL_large', proto, '', bnll_input, [F.softplus(bnll_input)], []))
    value = torch.tensor([-.5, 0., .125, .25, .5, 1.]).reshape(1, 1, 2, 3)
    for threshold in (0., .25):
        proto = input_layer(value.shape) + layer('op', 'Threshold', ['data'], ['out'],
            'threshold_param { threshold: %s }' % threshold)
        cases.append(('Threshold_boundary_%s' % threshold, proto, '', value,
            [(value > threshold).float()], []))
    proto = input_layer(x.shape) + layer('reshape', 'Reshape', ['data'], ['flat'],
        'reshape_param { shape { dim: 0 dim: 4 dim: -1 } }')
    proto += layer('permute', 'Permute', ['flat'], ['out'], 'permute_param { order: 0 order: 2 order: 1 }')
    cases.append(('reshape_permute', proto, '', x, [x.reshape(1, 4, -1).permute(0, 2, 1)], []))
    for name, params, dims in [
        ('permute_default', '', (0, 1, 2, 3)),
        ('permute_partial', 'permute_param { order: 0 order: 3 }', (0, 3, 1, 2))]:
        proto = input_layer(x.shape) + layer('permute', 'Permute', ['data'], ['out'], params)
        cases.append((name, proto, '', x, [x.permute(dims)], []))
    proto = input_layer(x.shape) + layer('reshape', 'Reshape', ['data'], ['out'],
        'reshape_param { axis: 2 num_axes: 2 shape { dim: -1 } }')
    cases.append(('reshape_partial', proto, '', x, [x.reshape(1, 4, 56)], []))
    proto = input_layer(x.shape) + layer('reshape', 'Reshape', ['data'], ['out'],
        'reshape_param { axis: -1 num_axes: 0 shape { dim: 1 } }')
    cases.append(('reshape_insert_axis', proto, '', x, [x.reshape(1, 4, 7, 8, 1)], []))
    proto = input_layer(x.shape) + layer('flatten', 'Flatten', ['data'], ['out'],
        'flatten_param { axis: 1 end_axis: 2 }')
    cases.append(('flatten_partial', proto, '', x, [x.flatten(1, 2)], []))
    value = torch.rand(1, 2, 6, 8)
    proto = input_layer(value.shape) + layer('reorg', 'Reorg', ['data'], ['out'], 'reorg_param { stride: 2 }')
    # legacy ncnn Reorg mode 0 orders channels before spatial offsets
    ref = value.reshape(1, 2, 3, 2, 4, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 8, 3, 4)
    cases.append(('reorg', proto, '', value, [ref], []))
    for stride in (1, 3):
        value = torch.arange(216.).reshape(1, 2, 9, 12) / 216
        proto = input_layer(value.shape) + layer('reorg', 'Reorg', ['data'], ['out'],
            'reorg_param { stride: %d }' % stride)
        ref = value.reshape(1, 2, 9 // stride, stride, 12 // stride, stride)
        ref = ref.permute(0, 1, 3, 5, 2, 4).reshape(1, 2 * stride * stride, 9 // stride, 12 // stride)
        cases.append(('reorg_stride%d' % stride, proto, '', value, [ref], []))
    for group in (1, 3, 6):
        value = torch.arange(120.).reshape(1, 6, 4, 5) / 120
        proto = input_layer(value.shape) + layer('shuffle', 'ShuffleChannel', ['data'], ['out'],
            'shuffle_channel_param { group: %d }' % group)
        ref = value.reshape(1, group, 6 // group, 4, 5).transpose(1, 2).reshape_as(value)
        cases.append(('shuffle_group%d' % group, proto, '', value, [ref], []))
    proto = input_layer(x.shape) + layer('target_input', 'Input', [], ['target'],
        'input_param { shape { dim: 1 dim: 4 dim: 3 dim: 4 } }')
    proto += layer('crop', 'Crop', ['data', 'target'], ['out'], 'crop_param { axis: 2 offset: 1 offset: 2 }')
    # The reference contributes only its shape, but model inputs are preserved.
    cases.append(('crop', proto, '', [x, torch.zeros(1, 4, 3, 4)], [x[:, :, 1:4, 2:6]], []))
    for name, value, target_shape, params, reference in [
        ('crop_rank2', torch.arange(8.).reshape(1, 8), (1, 4),
         'axis: 1 offset: 2', torch.arange(2., 6.).reshape(1, 4)),
        ('crop_rank3', torch.arange(32.).reshape(1, 4, 8), (1, 2, 4),
         'axis: 1 offset: 1 offset: 2', torch.arange(32.).reshape(1, 4, 8)[:, 1:3, 2:6])]:
        proto = input_layer(value.shape) + layer('target_input', 'Input', [], ['target'],
            'input_param { shape { %s } }' % ' '.join('dim: %d' % d for d in target_shape))
        proto += layer('crop', 'Crop', ['data', 'target'], ['out'], 'crop_param { %s }' % params)
        cases.append((name, proto, '', [value, torch.zeros(target_shape)], [reference], []))
    # PSPNet Caffe: shrink is floor((size-1)/factor)+1; zoom scales the
    # intervals between corner pixels. Negative padding crops both dimensions
    for name, params, cropped, size in [
        ('interp_zoom', 'zoom_factor: 2', x, (13, 15)),
        ('interp_shrink', 'shrink_factor: 2', x, (4, 4)),
        ('interp_shrink_zoom', 'shrink_factor: 2 zoom_factor: 3', x, (10, 10)),
        ('interp_crop', 'pad_beg: -1 pad_end: -2 height: 5 width: 6', x[:, :, 1:-2, 1:-2], (5, 6)),
        ('interp_crop_zoom', 'pad_beg: -1 pad_end: -1 zoom_factor: 2', x[:, :, 1:-1, 1:-1], (9, 11)),
        ('interp_singleton', 'height: 1 width: 1', x, (1, 1))]:
        proto = input_layer(x.shape) + layer('interp', 'Interp', ['data'], ['out'], 'interp_param { %s }' % params)
        cases.append((name, proto, '', x, [F.interpolate(cropped, size=size, mode='bilinear', align_corners=True)], []))
    results = []
    for name, proto, weights, value, reference, options in cases:
        if args.filter and not re.search(args.filter, name):
            continue
        Path(name + '.prototxt').write_text(proto)
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                 input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
        Path(name + '.caffemodel').write_bytes(encoded)
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'] + options)
            axes = [0, None] if name.startswith('scale_second_input') else None
            pytorch, actual = inference(prefix, value, axes)
            assert len(actual) == len(pytorch) == len(reference)
            for i, ref in enumerate(reference):
                if name.startswith('Threshold'):
                    assert pytorch[i].dtype == ref.dtype == torch.float32
                    assert actual[i].dtype == torch.float32
                results.append(compare(prefix + '/torch/%d' % i, pytorch[i], ref, 0))
                # ncnn omits precisely the leading batch dimension
                results.append(compare(prefix + '/ncnn/%d' % i, actual[i], ref[0], fp16))
    # invalid inputs must fail as Caffe input, without falling back to ONNX
    errors = [('missing', None, None), ('malformed', 'layer {', b''),
              ('missing_weights', input_layer(x.shape), None),
              ('malformed_weights', input_layer(x.shape), b'\xff'),
              ('missing_bottom', input_layer(x.shape) + layer('broken', 'ReLU', ['absent'], ['out']), b''),
              ('duplicate_input', input_layer(x.shape) + input_layer(x.shape), b''),
              ('missing_conv_weights', input_layer(x.shape) + layer('conv', 'Convolution', ['data'], ['out'],
                  'convolution_param { num_output: 4 kernel_size: 3 }'), b''),
              ('unsupported', input_layer(x.shape) + layer('unsupported_layer', 'NotACaffeOperator', ['data'], ['out']), b'')]
    for name, axis, tiles in [('tile_zero', 1, 0), ('tile_invalid_axis', 4, 2), ('tile_batch_repeat', 0, 2)]:
        errors.append((name, input_layer(x.shape) + layer('op', 'Tile', ['data'], ['out'],
            'tile_param { axis: %d tiles: %d }' % (axis, tiles)), b''))
    for kind, base in [('Exp', 0), ('Exp', -2), ('Log', 0), ('Log', -2), ('Log', 1)]:
        errors.append(('%s_invalid_base_%s' % (kind, base),
            input_layer(x.shape) + layer('op', kind, ['data'], ['out'],
                '%s_param { base: %s }' % (kind.lower(), base)), b''))
    diagnostic_cases = [
        ('conv_axis', 'Convolution', 'convolution_param { num_output: 4 kernel_size: 1 axis: 0 }', 'axis=0 is unsupported'),
        ('conv_group', 'Convolution', 'convolution_param { num_output: 4 kernel_size: 1 group: 3 }', 'divisible by group'),
        ('pool_method', 'Pooling', 'pooling_param { pool: STOCHASTIC kernel_size: 2 }', 'expected MAX(0) or AVE(1)'),
        ('pool_stride', 'Pooling', 'pooling_param { kernel_size: 2 stride: 0 }', 'require positive input/kernel/stride'),
        ('linear_axis', 'InnerProduct', 'inner_product_param { num_output: 2 axis: 0 }', 'expected a non-batch axis'),
        ('linear_weights', 'InnerProduct', 'inner_product_param { num_output: 2 }', 'requires 2 weight blobs, got 0'),
        ('batchnorm_weights', 'BatchNorm', '', 'expected 2 or 3 statistic blobs, got 0'),
        ('scale_axis', 'Scale', 'scale_param { axis: 5 }', 'axis'),
    ]
    for name, kind, params, reason in diagnostic_cases:
        errors.append((name, input_layer(x.shape) + layer('invalid_op', kind, ['data'], ['out'], params), b''))
    for name, proto, weights in errors:
        if proto is not None:
            Path(name + '.prototxt').write_text(proto)
        if weights is not None:
            Path(name + '.caffemodel').write_bytes(weights)
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert 'caffe' in result.stderr.lower(), (name, result.stderr)
        if name == 'unsupported':
            assert 'unsupported_layer' in result.stderr and 'NotACaffeOperator' in result.stderr
            assert result.stderr.count('unsupported operator type') == 1, result.stderr
            assert 'invalid weights or unknown input shape' not in result.stderr, result.stderr
        for case_name, kind, params, reason in diagnostic_cases:
            if name == case_name:
                assert 'invalid_op' in result.stderr and 'caffe.' + kind in result.stderr, result.stderr
                assert reason in result.stderr, result.stderr
                assert result.stderr.count('caffe layer invalid_op') == 1, result.stderr
        results.append(dict(name=name, passed=True))
    directory = Path('models.测试 with spaces')
    directory.mkdir(exist_ok=True)
    path = directory / 'network.v1.prototxt'
    path.write_text(input_layer(x.shape) + layer('relu', 'ReLU', ['data'], ['out']))
    path.with_suffix('.caffemodel').write_bytes(b'')
    prefix = 'path_case'
    run([pnnx, str(path), 'optlevel=2', 'fp16=0',
         'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
         'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
         'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
    pytorch, actual = inference(prefix, x)
    assert len(pytorch) == len(actual) == 1
    results.append(compare(prefix + '/torch', pytorch[0], x.relu(), 0))
    results.append(compare(prefix + '/ncnn', actual[0], x[0].relu(), 0))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results),
                  skipped=sum(bool(args.filter and not re.search(args.filter, c[0])) for c in cases))
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--protoc', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--filter', help='Run matching model names for debugging; omitted cases are reported as skipped')
    exit(0 if test(parser.parse_args()) else 1)
