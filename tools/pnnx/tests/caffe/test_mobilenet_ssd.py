# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from test_utils import ncnn_inference
from test_utils import compare, run


def test(args):
    pnnx = shutil.which(args.pnnx)
    caffe2ncnn = shutil.which(args.caffe2ncnn)
    assert pnnx and caffe2ncnn, 'pnnx and legacy caffe2ncnn are required'
    source = Path(args.model_dir).resolve()
    # pin the external fixture to the revision documented in CAFFE.md
    hashes = {
        'deploy.prototxt': '2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608',
        'mobilenet_iter_73000.caffemodel': '52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7',
        'images/000001.jpg': '3f0bdad67d01aa452929683b74a124a2926b6bce534c85f3ee0f00e20eeacab0'
    }
    for name, expected in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'deploy.prototxt', output / 'mobilenet_ssd.prototxt')
    shutil.copyfile(source / 'mobilenet_iter_73000.caffemodel', output / 'mobilenet_ssd.caffemodel')
    # follow the upstream demo.py preprocessing, retaining BGR order
    image = cv2.imread(str(source / 'images' / '000001.jpg'))
    assert image is not None, 'MobileNet-SSD sample image 000001.jpg is required'
    value = (cv2.resize(image, (300, 300)) - 127.5) * .007843
    x = torch.from_numpy(value.astype(np.float32).transpose(2, 0, 1).copy()).unsqueeze(0)
    os.chdir(output)
    np.save('input.npy', x.numpy())
    torch.set_num_threads(1)
    # legacy caffe2ncnn does not import net-level inputs. Express the same input
    # as an Input layer only for that baseline; pnnx consumes the original file
    proto = Path('mobilenet_ssd.prototxt').read_text()
    legacy_proto, count = re.subn(r'input:\s*"data"\s*input_shape\s*\{([^}]*)\}',
        r'layer { name: "input" type: "Input" top: "data" input_param { shape {\1} } }', proto, count=1)
    assert count == 1
    Path('legacy.prototxt').write_text(legacy_proto)
    run([caffe2ncnn, 'legacy.prototxt', 'mobilenet_ssd.caffemodel', 'legacy.param', 'legacy.bin'])
    reference = ncnn_inference('legacy.param', 'legacy.bin', x)
    assert reference.ndim == 2 and reference.shape[1] == 6 and reference.shape[0] > 0, reference.shape
    results = []
    for fp16 in (0, 1):
        prefix = 'mobilenet_ssd_fp%d' % fp16
        run([pnnx, 'mobilenet_ssd.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
             'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
             'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
             'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
        actual = ncnn_inference(prefix + '.ncnn.param', prefix + '.ncnn.bin', x)
        spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with torch.no_grad():
            pytorch = module.Model().eval()(x)
        assert pytorch.ndim == 4 and pytorch.shape[:2] == (1, 1) and pytorch.shape[-1] == 7
        assert torch.equal(pytorch[0, 0, :, 0], torch.zeros(pytorch.shape[2]))
        pytorch = pytorch[0, 0, :, 1:]
        # the native layer and exported helper sort retained detections by score
        # compare the complete ordered result; do not discard low-score rows
        assert actual.shape == pytorch.shape == reference.shape
        assert torch.equal(actual[:, 0], reference[:, 0])
        assert torch.equal(pytorch[:, 0], reference[:, 0])
        results.append(compare(prefix + '/ncnn_vs_legacy', actual, reference, fp16))
        results.append(compare(prefix + '/torch_vs_legacy', pytorch, reference, 0))
        results.append(compare(prefix + '/ncnn_vs_torch', actual, pytorch, fp16))
    report = dict(input='upstream images/000001.jpg; BGR resize 300x300; (x-127.5)*0.007843',
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
