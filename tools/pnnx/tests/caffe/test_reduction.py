# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from test_utils import input_layer, layer, inference
from test_utils import compare, run


def reference(x, operation, axis, coeff):
    # Caffe reduces every dimension from axis onward, then applies coeff
    # NumPy is independent of the generated Torch and native reduction graphs
    value = x.numpy()
    axis = axis if axis >= 0 else value.ndim + axis
    axes = tuple(range(axis, value.ndim))
    if operation == 'ASUM':
        value = np.abs(value)
    elif operation == 'SUMSQ':
        value = np.square(value)
    if operation == 'MEAN':
        value = np.mean(value, axis=axes)
    else:
        value = np.sum(value, axis=axes)
    return torch.from_numpy(np.asarray(value * np.float32(coeff)))


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    # include both signs and avoid dependence on a random generator version
    x = torch.linspace(-1.3, 2.1, 60).reshape(1, 3, 4, 5)
    cases = [('default', x, 'SUM', 0, 1., '')]
    for operation in ('SUM', 'ASUM', 'SUMSQ', 'MEAN'):
        for axis in (0, 1, 2, -1):
            for coeff in (1., -.25):
                name = '%s_axis%d_coeff%s' % (operation, axis, str(coeff).replace('.', '_'))
                params = 'reduction_param { operation: %s axis: %d coeff: %s }' % (operation, axis, coeff)
                cases.append((name, x, operation, axis, coeff, params))
        value = x.flatten()[:7]
        params = 'reduction_param { operation: %s axis: -1 coeff: 0.5 }' % operation
        cases.append((operation + '_vector', value, operation, -1, .5, params))

    # A rank-zero reduction is a known scalar, including through several
    # pointwise layers. Missing source shapes must still be diagnosed below.
    scalar_ops = [
        ('ReLU', '', torch.relu),
        ('Sigmoid', '', torch.sigmoid),
        ('TanH', '', torch.tanh),
        ('AbsVal', '', torch.abs),
        ('ELU', '', F.elu),
        ('ReLU6', '', F.relu6),
        ('Power', 'power_param { power: 2 scale: 0.5 shift: 1 }', lambda v: (v * .5 + 1) ** 2),
        ('Power', 'power_param { power: 0.5 scale: 0.5 shift: 1 }', lambda v: (v * .5 + 1).sqrt()),
        ('Exp', '', torch.exp),
        ('Log', 'log_param { shift: 1 }', lambda v: (v + 1).log()),
        ('BNLL', '', F.softplus),
        ('Threshold', 'threshold_param { threshold: 0 }', lambda v: (v > 0).float()),
        ('Dropout', '', lambda v: v),
    ]
    scalar_tails = {}
    for index, (kind, params, fn) in enumerate(scalar_ops):
        for sign in (-1, 1):
            name = 'scalar_%s_%d_sign%d' % (kind, index, sign)
            scalar_tails[name] = (layer('pointwise', kind, ['reduced'], ['out'], params), fn)
            cases.append((name, x * (.01 * sign), 'SUM', 0, 1., ''))
    name = 'scalar_chain_split_eltwise'
    tail = layer('relu', 'ReLU', ['reduced'], ['positive'])
    tail += layer('split', 'Split', ['positive'], ['a', 'b'])
    tail += layer('dropout', 'Dropout', ['a'], ['c'])
    tail += layer('sum', 'Eltwise', ['b', 'c'], ['out'])
    scalar_tails[name] = (tail, lambda v: v.relu() * 2)
    cases.append((name, x, 'SUM', 0, 1., ''))

    results = []
    for name, value, operation, axis, coeff, params in cases:
        proto = input_layer(value.shape) + layer('reduce', 'Reduction', ['data'],
            ['reduced' if name in scalar_tails else 'out'], params)
        ref = reference(value, operation, axis, coeff)
        if name in scalar_tails:
            tail, fn = scalar_tails[name]
            proto += tail
            ref = fn(ref)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, native = inference(prefix, value, [None if value.ndim == 1 else 0])
            assert len(pytorch) == len(native) == 1
            results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
            native_ref = ref
            if ref.ndim == 0:
                # ncnn has no rank-zero Mat; scalar reductions use one element
                assert native[0].shape == (1,), native[0].shape
                native_ref = ref.reshape(1)
            elif ref.ndim > 1:
                assert ref.shape[0] == 1
                native_ref = ref[0]
            results.append(compare(prefix + '/ncnn', native[0], native_ref, fp16))

    for name, shape, params in [
        ('invalid_axis', (1, 3, 4, 5), 'reduction_param { axis: 4 }'),
        ('unsupported_batch', (2, 3, 4, 5), 'reduction_param { axis: 0 }')]:
        proto = input_layer(shape) + layer('reduce', 'Reduction', ['data'], ['out'], params)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0 and 'Reduction' in result.stderr, result.stderr
        results.append(dict(name=name + '/diagnostic', passed=True))
    for index, (kind, params, _) in enumerate(scalar_ops):
        name = 'unknown_shape_%s_%d' % (kind, index)
        proto = 'input: "data"\n' + layer('pointwise', kind, ['data'], ['out'], params)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt'], capture_output=True, text=True)
        assert result.returncode != 0 and 'input shape is unknown' in result.stderr, result.stderr
        results.append(dict(name=name + '/diagnostic', passed=True))
    unknown_split = 'input: "data"\n' + layer('split', 'Split', ['data'], ['a', 'b'])
    unknown_split += layer('relu', 'ReLU', ['a'], ['out'])
    unknown_eltwise = 'input: "unknown"\n' + input_layer(x.shape)
    unknown_eltwise += layer('reduce', 'Reduction', ['data'], ['reduced'])
    unknown_eltwise += layer('sum', 'Eltwise', ['reduced', 'unknown'], ['out'])
    for name, proto in [('unknown_shape_split', unknown_split), ('unknown_shape_eltwise', unknown_eltwise)]:
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt'], capture_output=True, text=True)
        assert result.returncode != 0 and 'shape is unknown' in result.stderr, result.stderr
        results.append(dict(name=name + '/diagnostic', passed=True))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
