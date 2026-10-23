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

from test_utils import input_layer, layer, inference
from test_utils import compare, run


def reference(x, top_k, values, axis):
    # Caffe uses std::greater<pair<value,index>>: equal values prefer larger
    # indices. With an explicit axis, out_max_val returns only values; without
    # an axis it returns an index row followed by a value row, both float32
    # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/argmax_layer.cpp
    data = x.numpy()
    if axis is None:
        rows = data.reshape(data.shape[0], -1)
    else:
        axis %= data.ndim
        moved = np.moveaxis(data, axis, -1)
        rows = moved.reshape(-1, moved.shape[-1])
    indices, maxima = [], []
    for row in rows:
        pairs = sorted(((float(value), index) for index, value in enumerate(row)), reverse=True)[:top_k]
        maxima.append([pair[0] for pair in pairs])
        indices.append([pair[1] for pair in pairs])
    indices = np.asarray(indices, dtype=np.float32)
    maxima = np.asarray(maxima, dtype=np.float32)
    if axis is None:
        shape = [1] * max(3, data.ndim)
        shape[0], shape[1], shape[2] = data.shape[0], 2 if values else 1, top_k
        result = np.stack((indices, maxima), axis=1) if values else indices
        return torch.from_numpy(result.reshape(shape))
    result = maxima if values else indices
    result = result.reshape(*moved.shape[:-1], top_k)
    return torch.from_numpy(np.moveaxis(result, -1, axis).copy())


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    x = torch.tensor([-2., 4., 1., 4., 0., -1., 2., 4., 3., 1., 4., -2., 3., 4., 1., 0., 2., 4., -3., 2., 4., 0., -1., 4.]).reshape(1, 3, 2, 4)
    cases = [('default', x, 1, False, None, '')]
    for top_k in (1, 3, x.numel()):
        for values in (False, True):
            cases.append(('flat_k%d_values%d' % (top_k, values), x, top_k, values, None, None))
    # aligned flattened lengths exclude allocation padding in the unchanged
    # native ArgMax. Unaligned lengths are checked as conversion diagnostics
    negative = torch.tensor([-4., -1., -2., -3., -1., -5., -8., -1., -6., -7., -2., -1.]).reshape(1, 3, 1, 4)
    cases.append(('negative_indices', negative, 1, False, None, None))
    cases.append(('negative_pairs', negative, 3, True, None, None))
    for axis in (0, 1, 2, -1):
        cases.append(('axis_values%d' % axis, x, 1, True, axis, None))
    vector = torch.tensor([2., -1., 2., 1., 2., 0., -3., 2.]).reshape(1, 8, 1, 1)
    for top_k in (1, 2, 8):
        for values in (False, True):
            cases.append(('vector_k%d_values%d' % (top_k, values), vector, top_k, values, 1, None))
    results = []
    for name, value, top_k, values, axis, params in cases:
        if params is None:
            params = 'argmax_param { top_k: %d out_max_val: %s %s }' % (
                top_k, str(values).lower(), '' if axis is None else 'axis: %d' % axis)
        proto = input_layer(value.shape) + layer('argmax', 'ArgMax', ['data'], ['out'], params)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        ref = reference(value, top_k, values, axis)
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, native = inference(prefix, value)
            assert len(pytorch) == len(native) == 1
            assert pytorch[0].dtype == native[0].dtype == torch.float32
            results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
            results.append(compare(prefix + '/ncnn', native[0], ref[0], fp16))
            # indices and tie ordering are exact; float tolerances cannot hide
            # an off-by-one index or reversed index/value output rows
            if not values:
                assert torch.equal(pytorch[0], ref)
                assert torch.equal(native[0], ref[0])
            elif axis is None:
                assert torch.equal(pytorch[0][:, 0], ref[:, 0])
                assert torch.equal(native[0][0], ref[0, 0])

    diagnostics = []
    for name, shape, params in [
        ('unaligned_flat', (1, 3, 2, 3), 'argmax_param { top_k: 18 }'),
        ('unaligned_negative', (1, 3, 1, 3), 'argmax_param { top_k: 1 }'),
        ('unaligned_axis', (1, 6, 1, 1), 'argmax_param { axis: 1 top_k: 2 out_max_val: true }'),
        ('invalid_topk', (1, 3, 2, 3), 'argmax_param { top_k: 0 }'),
        ('axis_indices_multiple_slices', (1, 3, 2, 3), 'argmax_param { axis: 1 }'),
        ('axis_topk_multiple_slices', (1, 3, 2, 3), 'argmax_param { axis: 1 top_k: 2 out_max_val: true }'),
        ('batch_two', (2, 3, 2, 3), '')]:
        proto = input_layer(shape) + layer('argmax', 'ArgMax', ['data'], ['out'], params)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0 and 'ArgMax' in result.stderr, result.stderr
        if name.startswith('unaligned'):
            assert 'allocation padding' in result.stderr and 'argmax' in result.stderr, result.stderr
        diagnostics.append(dict(name=name + '/diagnostic', passed=True))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0,
                  diagnostics=diagnostics, diagnostics_passed=len(diagnostics))
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
