# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from test_utils import blob, input_layer, layer, inference
from test_utils import compare, run


def test_consumers(pnnx, protoc, schema):
    weight = torch.tensor([[-1., .5, 2.], [.25, -2., 1.5], [1., -.25, -1.]])
    results = []
    for batched in (False, True):
        x = torch.tensor([0, 2], dtype=torch.int32)
        if batched:
            x = x.unsqueeze(0)
        value = weight[x.long()]
        base = input_layer(x.shape) + layer('embed', 'Embed', ['data'], ['embedding'],
            'embed_param { input_dim: 3 num_output: 3 bias_term: false }')
        weights = layer('embed', 'Embed', [], [], blob(weight))
        tail = layer('reduce', 'Reduction', ['embedding'], ['out'], 'reduction_param { axis: 1 }')
        cases = [('reduce', tail, '', value.sum(tuple(range(1, value.ndim))), False)]
        # preserve the source batch axis through pointwise layers and Split
        # before Reduction inspects it
        for kind, params, expected in [
            ('ReLU', '', value.relu()),
            ('Dropout', '', value),
            ('Split', '', value),
            ('ELU', 'elu_param { alpha: 0.5 }', torch.where(value >= 0, value, .5 * (value.exp() - 1))),
            ('Sigmoid', '', value.sigmoid()),
            ('TanH', '', value.tanh()),
            ('AbsVal', '', value.abs()),
            ('ReLU6', '', value.clamp(0, 6)),
            ('Power', 'power_param { power: 0.5 shift: 2 }', (value + 2).sqrt()),
            ('Softmax', 'softmax_param { axis: 1 }', value.softmax(1)),
        ]:
            tops = ['intermediate', 'unused'] if kind == 'Split' else ['intermediate']
            tail = layer('pointwise', kind, ['embedding'], tops, params)
            if kind == 'Split':
                tail += layer('silence', 'Silence', ['unused'], [])
            tail += layer('reduce', 'Reduction', ['intermediate'], ['out'], 'reduction_param { axis: 1 }')
            cases.append(('reduce_' + kind, tail, '', expected.sum(tuple(range(1, value.ndim))), False))

        channels = value.shape[1]
        broadcast = [1] * value.ndim
        broadcast[1] = channels
        for shared in (False, True):
            slope = torch.linspace(.1, .6, 1 if shared else channels)
            expected = value.clamp(min=0) + value.clamp(max=0) * slope.reshape([1] * value.ndim if shared else broadcast)
            tail = layer('prelu', 'PReLU', ['embedding'], ['out'],
                'prelu_param { channel_shared: %s }' % str(shared).lower())
            cases.append(('prelu_shared%d' % shared, tail, layer('prelu', 'PReLU', [], [], blob(slope)), expected, batched))
        mean = torch.linspace(.1, .3, channels)
        variance = torch.linspace(.5, 1.5, channels)
        for factor in (0., 2.):
            scale = 0. if factor == 0. else 1. / factor
            expected = (value - (mean * scale).reshape(broadcast)) / (variance * scale + .001).sqrt().reshape(broadcast)
            tail = layer('bn', 'BatchNorm', ['embedding'], ['out'], 'batch_norm_param { eps: 0.001 }')
            data = blob(mean) + blob(variance) + blob(torch.tensor([factor]))
            cases.append(('batch_norm_factor%d' % factor, tail, layer('bn', 'BatchNorm', [], [], data), expected, batched))
        dims = tuple(range(1, value.ndim))
        centered = value - value.mean(dims, keepdim=True)
        for normalize_variance in (False, True):
            expected = centered
            if normalize_variance:
                expected = centered / (centered.square().mean(dims, keepdim=True).sqrt() + .001)
            tail = layer('mvn', 'MVN', ['embedding'], ['out'],
                'mvn_param { across_channels: true normalize_variance: %s eps: 0.001 }' % str(normalize_variance).lower())
            cases.append(('mvn_variance%d' % normalize_variance, tail, '', expected, batched))
        tail = layer('normalize', 'Normalize', ['embedding'], ['out'],
            'norm_param { across_spatial: true channel_shared: true eps: 0.001 }')
        expected = value / (value.square().sum(dims, keepdim=True) + .001).sqrt() * 2
        cases.append(('normalize', tail, layer('normalize', 'Normalize', [], [], blob(torch.tensor([2.]))), expected, batched))

        for name, tail, data, expected, output_batched in cases:
            name = 'embed_consumer_%d_%s' % (batched, name)
            Path(name + '.prototxt').write_text(base + tail)
            encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                     input=(weights + data).encode(), stdout=subprocess.PIPE, check=True).stdout
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, x, [0 if batched else None])
                assert len(pytorch) == len(actual) == 1
                results.append(compare(prefix + '/torch', pytorch[0], expected, 0))
                results.append(compare(prefix + '/ncnn', actual[0], expected[0] if output_batched else expected, fp16))
    return results


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    schema = Path(args.schema).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    weight, bias = torch.rand(7, 5), torch.rand(5)
    results = []
    for batched in (False, True):
        for use_bias in (False, True):
            # Caffe's float-valued integer indices become explicit int32 values
            # at the exported model boundary. Both paths see the same indices
            x = torch.tensor([0, 6, 3, 0], dtype=torch.int32)
            if batched:
                x = x.unsqueeze(0)
            ref = weight[x.long()] + (bias if use_bias else 0.)
            name = 'embed%d_%d' % (batched, use_bias)
            params = 'embed_param { input_dim: 7 num_output: 5 bias_term: %s }' % str(use_bias).lower()
            proto = input_layer(x.shape) + layer('embed', 'Embed', ['data'], ['out'], params)
            weights = layer('embed', 'Embed', [], [], blob(weight) + (blob(bias) if use_bias else ''))
            encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                     input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, x, [0 if batched else None])
                assert len(pytorch) == len(actual) == 1
                results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
                results.append(compare(prefix + '/ncnn', actual[0], ref[0] if batched else ref, fp16))
    for batched in (False, True):
        for routing in ('shared', 'split', 'nested_split'):
            x = torch.tensor([0, 6, 3, 0], dtype=torch.int32)
            if batched:
                x = x.unsqueeze(0)
            name = 'embed_%s_%d' % (routing, batched)
            proto = input_layer(x.shape)
            bottoms = ['data', 'data']
            if routing != 'shared':
                proto += layer('split', 'Split', ['data'], ['left', 'right'])
                bottoms = ['left', 'right']
            if routing == 'nested_split':
                proto += layer('split_again', 'Split', ['right'], ['right0', 'right1'])
                bottoms = ['left', 'right0', 'right1']
            weights = ''
            references = []
            for i, bottom in enumerate(bottoms):
                table = weight + i
                use_bias = i % 2 == 0
                params = 'embed_param { input_dim: 7 num_output: 5 bias_term: %s }' % str(use_bias).lower()
                proto += layer('embed%d' % i, 'Embed', [bottom], ['out%d' % i], params)
                weights += layer('embed%d' % i, 'Embed', [], [], blob(table) + (blob(bias) if use_bias else ''))
                references.append(table[x.long()] + (bias if use_bias else 0.))
            encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                     input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, x, [0 if batched else None])
                assert len(pytorch) == len(actual) == len(references)
                for i, ref in enumerate(references):
                    results.append(compare(prefix + '/torch/%d' % i, pytorch[i], ref, 0))
                    results.append(compare(prefix + '/ncnn/%d' % i, actual[i], ref[0] if batched else ref, fp16))
    # an identity-looking floating-point computation must not silently become
    # an integer-storage path. Reject mixed consumers before adapting the input
    for routing in ('computed', 'mixed', 'mixed_split'):
        name = 'embed_invalid_' + routing
        proto = input_layer((4,))
        bottom = 'data'
        if routing == 'computed':
            proto += layer('floating', 'Power', ['data'], ['computed'], 'power_param { scale: 1 }')
            bottom = 'computed'
        elif routing == 'mixed':
            proto += layer('floating', 'ReLU', ['data'], ['float_out'])
        else:
            proto += layer('split', 'Split', ['data'], ['indices', 'floats'])
            proto += layer('floating', 'ReLU', ['floats'], ['float_out'])
            bottom = 'indices'
        proto += layer('embed', 'Embed', [bottom], ['out'], 'embed_param { input_dim: 7 num_output: 5 bias_term: false }')
        weights = layer('embed', 'Embed', [], [], blob(weight))
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                 input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(encoded)
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=0'], capture_output=True, text=True)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert 'indices' in result.stderr and 'floating' in result.stderr, (name, result.stderr)
        results.append(dict(name=name, passed=True))
    results += test_consumers(pnnx, protoc, schema)
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--protoc', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
