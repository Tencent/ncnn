# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import subprocess

import ncnn
import numpy as np
import torch


def blob(value):
    return 'blobs { shape { %s } %s }' % (
        ' '.join('dim: %d' % d for d in value.shape),
        ' '.join('data: %.9g' % d for d in value.flatten().tolist()))


def layer(name, kind, bottom, top, params=''):
    return 'layer { name: "%s" type: "%s" %s %s %s }\n' % (
        name, kind, ' '.join('bottom: "%s"' % b for b in bottom),
        ' '.join('top: "%s"' % t for t in top), params)


def input_layer(shape):
    return layer('input', 'Input', [], ['data'], 'input_param { shape { %s } }' %
                 ' '.join('dim: %d' % d for d in shape))


def inputs(shapes):
    return ''.join(layer('input%d' % i, 'Input', [], ['data%d' % i],
        'input_param { shape { %s } }' % ' '.join('dim: %d' % d for d in shape))
        for i, shape in enumerate(shapes))


def ncnn_inference(param, weight, x):
    with ncnn.Net() as net:
        net.opt.num_threads = 1
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param(str(param)) == 0
        assert net.load_model(str(weight)) == 0
        assert len(net.input_names()) == 1
        assert len(net.output_names()) == 1
        with net.create_extractor() as ex:
            assert ex.input(net.input_names()[0], ncnn.Mat(x[0].numpy()).clone()) == 0
            ret, out = ex.extract(net.output_names()[0])
            assert ret == 0
            if out.dims == 0:
                return torch.empty(0)
            return torch.from_numpy(np.array(out).copy())


def inference(prefix, x, batch_axes=None):
    inputs = list(x) if isinstance(x, (tuple, list)) else [x]
    if batch_axes is None:
        batch_axes = [0] * len(inputs)
    spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        result = module.Model().eval()(*inputs)
    pytorch = list(result) if isinstance(result, (tuple, list)) else [result]
    with ncnn.Net() as net:
        net.opt.num_threads = 1
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_fp16_packed = False
        assert net.load_param(prefix + '.ncnn.param') == 0
        assert net.load_model(prefix + '.ncnn.bin') == 0
        assert len(net.input_names()) == len(inputs)
        actual = []
        with net.create_extractor() as ex:
            for name, value, axis in zip(net.input_names(), inputs, batch_axes):
                tensor = value[0] if axis == 0 else value
                assert ex.input(name, ncnn.Mat(tensor.numpy()).clone()) == 0
            for name in net.output_names():
                ret, out = ex.extract(name)
                assert ret == 0
                actual.append(torch.from_numpy(np.array(out).copy()) if out.dims else torch.empty(0))
    return pytorch, actual


def run(command):
    print('+ ' + ' '.join(str(x) for x in command), flush=True)
    subprocess.run(command, check=True)


def compare(name, actual, reference, fp16):
    assert actual.shape == reference.shape, (name, actual.shape, reference.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), name
    atol, rtol = (1e-3, 1e-2) if fp16 else (1e-5, 1e-4)
    error = (actual - reference).abs()
    passed = bool(torch.all(error <= atol + rtol * reference.abs()))
    result = dict(name=name, shape=list(actual.shape), max_abs_error=error.max().item() if error.numel() else 0.,
                  atol=atol, rtol=rtol, passed=passed)
    print(json.dumps(result), flush=True)
    return result
