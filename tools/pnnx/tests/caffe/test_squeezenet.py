# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from test_utils import compare, ncnn_inference, run


def test(args):
    pnnx = shutil.which(args.pnnx)
    caffe2ncnn = shutil.which(args.caffe2ncnn)
    assert pnnx, 'pnnx executable is required'
    assert caffe2ncnn, 'legacy caffe2ncnn executable is required'
    source = Path(args.model_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('prototxt', 'caffemodel'):
        shutil.copyfile(source / ('squeezenet_v1.1.' + suffix), output / ('squeezenet_v1.1.' + suffix))
    os.chdir(output)
    torch.set_num_threads(1)
    # a fixed synthetic BGR image already at the example's 227 x 227 resize size
    # follow examples/squeezenet.cpp: BGR channel order, subtract 104/117/123,
    # and no division by 255. All three paths receive this exact tensor
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 3, 227, 227)).float()
    x -= torch.tensor([104., 117., 123.]).reshape(1, 3, 1, 1)
    np.save('input.npy', x.numpy())
    run([caffe2ncnn, 'squeezenet_v1.1.prototxt', 'squeezenet_v1.1.caffemodel', 'legacy.param', 'legacy.bin'])
    reference = ncnn_inference('legacy.param', 'legacy.bin', x)
    assert reference.shape == (1000,), reference.shape
    results = []
    for fp16 in (0, 1):
        prefix = 'squeezenet_fp%d' % (16 if fp16 else 32)
        run([pnnx, 'squeezenet_v1.1.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
             'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
             'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
             'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
        actual = ncnn_inference(prefix + '.ncnn.param', prefix + '.ncnn.bin', x)
        # the unified exporter retains the singleton spatial dimensions;
        # legacy caffe2ncnn global pooling emits a vector instead
        assert actual.shape == (1000, 1, 1), actual.shape
        actual = actual[:, 0, 0]
        spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.no_grad():
            pytorch = module.Model().eval()(x)
        assert isinstance(pytorch, torch.Tensor)
        assert pytorch.shape == (1, 1000, 1, 1), pytorch.shape
        pytorch = pytorch[0, :, 0, 0]
        results.append(compare(prefix + '/ncnn_vs_legacy', actual, reference, fp16))
        results.append(compare(prefix + '/torch_vs_legacy', pytorch, reference, 0))
        results.append(compare(prefix + '/ncnn_vs_torch', actual, pytorch, fp16))
    report = dict(input='input.npy; seed=0; BGR 227x227 minus [104,117,123]',
                  results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--caffe2ncnn', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
