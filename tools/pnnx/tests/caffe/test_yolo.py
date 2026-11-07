# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch

from test_utils import inputs
from test_utils import layer, inference
from test_utils import compare, run


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    results = []
    diagnostics_passed = 0
    for v3 in (False, True):
        kind = 'Yolov3DetectionOutput' if v3 else 'YoloDetectionOutput'
        parameter = 'yolov3_detection_output_param' if v3 else 'yolo_detection_output_param'
        for heads in (1, 2):
            values = [torch.zeros(1, 7, 1, 1) for i in range(heads)]
            score = 1. / 3 if v3 else .25
            label = 1.
            if heads == 2:
                # the second head predicts the same box but a different class
                # with higher confidence. Legacy YOLO uses class-agnostic NMS
                values[1][0, 4, 0, 0] = 2.
                values[1][0, 6, 0, 0] = 2.
                label = 2.
                score = 1 / (1 + math.exp(-2) * (1 + math.exp(-2))) if v3 else 1 / (1 + math.exp(-2)) ** 2
            params = 'num_classes: 2 num_box: 1 confidence_threshold: 0.1 nms_threshold: 0.3 '
            if v3:
                params += 'biases: 4 biases: 2 ' + 'mask: 0 anchors_scale: 10 ' * heads
            else:
                params += 'biases: 0.4 biases: 0.2 '
            proto = inputs([v.shape for v in values]) + layer('yolo', kind,
                ['data%d' % i for i in range(heads)], ['out'], '%s { %s }' % (parameter, params))
            name = kind + '_heads%d' % heads
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(b'')
            ref = torch.tensor([[label, score, .3, .4, .7, .6]])
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, values)
                assert len(pytorch) == len(actual) == 1
                assert torch.equal(pytorch[0][:, 0], ref[:, 0])
                assert torch.equal(actual[0][:, 0], ref[:, 0])
                results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
                results.append(compare(prefix + '/ncnn', actual[0], ref, fp16))
                # empty detections are a documented unchanged-runtime limitation,
                # not a supported numerical path. Verify the conversion warning
                warning = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16],
                                         capture_output=True, text=True)
                assert warning.returncode == 0, warning.stderr
                assert 'warning:' in warning.stderr and kind in warning.stderr, warning.stderr
                assert 'empty' in warning.stderr.lower(), warning.stderr
                diagnostics_passed += 1
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0,
                  diagnostics_passed=diagnostics_passed)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
