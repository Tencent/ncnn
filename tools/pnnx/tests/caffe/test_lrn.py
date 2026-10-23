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


def reference(x, within, size, alpha, beta, k):
    # BVLC Caffe's spatial LRN averages squared windows and applies a Power
    # layer with shift=1; k is only used by the across-channel implementation
    # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lrn_layer.cpp
    value = x.numpy()
    squared = np.square(value)
    result = np.empty_like(value)
    pad = size // 2
    for n in range(value.shape[0]):
        for c in range(value.shape[1]):
            for h in range(value.shape[2]):
                for w in range(value.shape[3]):
                    if within:
                        window = squared[n, c, max(0, h - pad):h + pad + 1, max(0, w - pad):w + pad + 1]
                        denominator = 1. + alpha * float(window.sum()) / (size * size)
                    else:
                        window = squared[n, max(0, c - pad):c + pad + 1, h, w]
                        denominator = k + alpha * float(window.sum()) / size
                    result[n, c, h, w] = value[n, c, h, w] * denominator ** (-beta)
    return torch.from_numpy(result)


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    results = []
    for shape in ((1, 3, 2, 4), (1, 1, 1, 1)):
        x = torch.linspace(-1.2, 1.8, int(np.prod(shape))).reshape(shape)
        for within in (False, True):
            region = 'WITHIN_CHANNEL' if within else 'ACROSS_CHANNELS'
            for size in (1, 3, 5):
                for case, (alpha, beta, k) in enumerate(((.3, .75, 1.), (1., .5, 2.5), (0., 1.2, 3.))):
                    name = '%s_c%d_size%d_case%d' % (region, shape[1], size, case)
                    params = 'lrn_param { norm_region: %s local_size: %d alpha: %s beta: %s k: %s }' % (region, size, alpha, beta, k)
                    proto = input_layer(shape) + layer('lrn', 'LRN', ['data'], ['out'], params)
                    Path(name + '.prototxt').write_text(proto)
                    Path(name + '.caffemodel').write_bytes(b'')
                    ref = reference(x, within, size, alpha, beta, k)
                    for fp16 in (0, 1):
                        prefix = name + '_fp%d' % fp16
                        run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                             'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                             'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                             'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                        pytorch, native = inference(prefix, x)
                        assert len(pytorch) == len(native) == 1
                        results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
                        results.append(compare(prefix + '/ncnn', native[0], ref[0], fp16))

    for region in ('WITHIN_CHANNEL', 'ACROSS_CHANNELS'):
        name = region + '_even_size'
        proto = input_layer((1, 3, 2, 4)) + layer('lrn', 'LRN', ['data'], ['out'],
            'lrn_param { norm_region: %s local_size: 2 }' % region)
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0 and 'LRN' in result.stderr, result.stderr
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
