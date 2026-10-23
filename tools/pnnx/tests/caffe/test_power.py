# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from test_utils import input_layer, layer, inference
from test_utils import compare, run


def compare_power(name, actual, expected, fp16):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    # finite input may produce NaN (negative fractional powers) or infinity
    # (zero negative powers). Check their classification and infinity signs
    # explicitly; never allow a NaN error to pass a tolerance comparison
    assert torch.equal(torch.isnan(actual), torch.isnan(expected)), name
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected)), name
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected)), name
    finite = torch.isfinite(expected)
    return compare(name, actual[finite], expected[finite], fp16)


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    signed = torch.tensor([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.]).reshape(1, 1, 3, 3)
    positive = torch.tensor([.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.]).reshape(1, 1, 3, 3)
    cases = []
    for power in (0., 1., 2., 3., 4., 5., -1., -2., -3.):
        for scale, shift in ((1., 0.), (-.5, .25), (0., -2.)):
            cases.append((signed, power, scale, shift))
    for power in (.5, 1.5, -.5):
        cases.append((positive, power, 1., 0.))
        cases.append((positive, power, .5, .25))
        cases.append((signed, power, 1., 0.))

    results = []
    for i, (value, power, scale, shift) in enumerate(cases):
        name = 'power_%d' % i
        params = 'power_param { power: %.9g scale: %.9g shift: %.9g }' % (power, scale, shift)
        proto = input_layer(value.shape) + layer('power', 'Power', ['data'], ['out'], params)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        # independent NumPy reference, preserving Caffe's float32 arithmetic
        with np.errstate(divide='ignore', invalid='ignore'):
            base = np.float32(shift) + np.float32(scale) * value.numpy()
            ref = torch.from_numpy(np.power(base, np.float32(power)))
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, native = inference(prefix, value)
            assert len(pytorch) == len(native) == 1
            results.append(compare_power(prefix + '/torch', pytorch[0], ref, 0))
            results.append(compare_power(prefix + '/ncnn', native[0], ref[0], fp16))

    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
