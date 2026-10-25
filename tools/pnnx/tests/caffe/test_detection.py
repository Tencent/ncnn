# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
from unittest import mock

import torch

from test_utils import inputs, layer, inference
from test_utils import compare, run


def prior_reference(height, width, image_h, image_w, min_sizes, max_sizes,
                    ratios, variance, step_h, step_w, offset, clip):
    # scalar enumeration independent of the exported vectorized Torch helper
    boxes = []
    for row in range(height):
        for col in range(width):
            cx, cy = (col + offset) * step_w, (row + offset) * step_h
            for index, size in enumerate(min_sizes):
                sizes = [(size, size)]
                if max_sizes:
                    side = math.sqrt(size * max_sizes[index])
                    sizes.append((side, side))
                sizes += [(size * math.sqrt(r), size / math.sqrt(r)) for r in ratios]
                for bw, bh in sizes:
                    box = [(cx - bw / 2) / image_w, (cy - bh / 2) / image_h,
                           (cx + bw / 2) / image_w, (cy + bh / 2) / image_h]
                    boxes.extend([max(0., min(1., v)) for v in box] if clip else box)
    return torch.tensor([[boxes, variance * (len(boxes) // 4)]])


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    # single centered square: center=(5,5), side=2 in a 10x10 image
    prior = torch.tensor([[[.4, .4, .6, .6], [.1, .1, .1, .1]]])
    proto = inputs([(1, 1, 1, 1), (1, 3, 10, 10)])
    proto += layer('prior', 'PriorBox', ['data0', 'data1'], ['out'],
                   'prior_box_param { min_size: 2 variance: 0.1 }')
    cases = [('prior', proto, [torch.zeros(1, 1, 1, 1), torch.zeros(1, 3, 10, 10)], prior, prior[0])]
    prior_cases = [
        ('prior_grid_multiple_sizes', 'min_size: 2 min_size: 4 max_size: 3 max_size: 5',
         [2, 4], [3, 5], [], [.1] * 4, 6, 12, 3., 4., .5, False),
        ('prior_ratio_no_flip', 'min_size: 2 aspect_ratio: 2 flip: false',
         [2], [], [2], [.1] * 4, 6, 12, 3., 4., .5, False),
        ('prior_ratio_flip', 'min_size: 2 aspect_ratio: 2 flip: true',
         [2], [], [2, .5], [.1] * 4, 6, 12, 3., 4., .5, False),
        ('prior_ratio_dedup', 'min_size: 2 aspect_ratio: 1 aspect_ratio: 2 aspect_ratio: 2 aspect_ratio: 0.5',
         [2], [], [2, .5], [.1] * 4, 6, 12, 3., 4., .5, False),
        ('prior_square_override_clip', 'min_size: 20 img_size: 10 step: 4 offset: 0 clip: true variance: 0.2',
         [20], [], [], [.2] * 4, 10, 10, 4., 4., 0., True),
        ('prior_rect_override_variance', 'min_size: 2 img_h: 8 img_w: 20 step_h: 2 step_w: 3 '
         'offset: 0.25 variance: 0.1 variance: 0.2 variance: 0.3 variance: 0.4',
         [2], [], [], [.1, .2, .3, .4], 8, 20, 2., 3., .25, False),
    ]
    for name, params, mins, maxs, ratios, variance, ih, iw, sh, sw, offset, clip in prior_cases:
        proto = inputs([(1, 1, 2, 3), (1, 3, 6, 12)])
        proto += layer('prior', 'PriorBox', ['data0', 'data1'], ['out'], 'prior_box_param { %s }' % params)
        expected = prior_reference(2, 3, ih, iw, mins, maxs, ratios, variance, sh, sw, offset, clip)
        cases.append((name, proto, [torch.zeros(1, 1, 2, 3), torch.zeros(1, 3, 6, 12)], expected, expected[0]))
    # zero localization offsets leave priors unchanged. The third box duplicates
    # the first and has lower confidence, so NMS removes it. The second box is
    # disjoint. Confidence scores are separated from threshold and each other
    priors = torch.tensor([[[.1, .1, .3, .3, .6, .6, .8, .8, .1, .1, .3, .3], [.1] * 12]])
    scores = torch.tensor([[.05, .95, .2, .8, .3, .7]])
    proto = inputs([(1, 12), (1, 6), (1, 2, 12)])
    proto += layer('detection', 'DetectionOutput', ['data0', 'data1', 'data2'], ['out'],
        'detection_output_param { num_classes: 2 code_type: CENTER_SIZE confidence_threshold: 0.5 '
        'keep_top_k: 10 nms_param { nms_threshold: 0.3 top_k: 10 } }')
    expected = torch.tensor([[[[0., 1., .95, .1, .1, .3, .3], [0., 1., .8, .6, .6, .8, .8]]]])
    cases.append(('detection', proto, [torch.zeros(1, 12), scores, priors], expected, expected[0, 0, :, 1:]))
    # confidence comparison is strict: a score exactly at the threshold is
    # discarded. Reordered confidence makes output order observable as well
    scores = torch.tensor([[.5, .5, .05, .95, .2, .8]])
    expected = torch.tensor([[[[0., 1., .95, .6, .6, .8, .8], [0., 1., .8, .1, .1, .3, .3]]]])
    cases.append(('confidence_boundary', proto, [torch.zeros(1, 12), scores, priors], expected, expected[0, 0, :, 1:]))
    risk_proto = inputs([(1, 4), (1, 2), (1, 2, 4)])
    risk_proto += layer('detection', 'DetectionOutput', ['data0', 'data1', 'data2'], ['out'],
        'detection_output_param { num_classes: 2 code_type: CENTER_SIZE confidence_threshold: 0.5 '
        'keep_top_k: 10 nms_param { nms_threshold: 0.3 top_k: 10 } }')
    expected = torch.tensor([[[[0., 1., .9, .1, .2, .5, .6]]]])
    cases.append(('detection_unnormalized_confidence', risk_proto,
                  [torch.zeros(1, 4), torch.tensor([[.8, .9]]),
                   torch.tensor([[[.1, .2, .5, .6], [.1] * 4]])], expected, expected[0, 0, :, 1:]))
    for name, score in [('no_detection_zero', 0.), ('no_detection_negative', -1.)]:
        expected = torch.full((1, 1, 1, 7), -1.)
        expected[0, 0, 0, 0] = 0.
        cases.append((name, proto, [torch.zeros(1, 12), torch.full((1, 6), score), priors], expected, torch.empty(0)))

    def detection_case(name, boxes, variances, locations, confidence, rows, options):
        count, classes = len(boxes), len(confidence[0])
        model = inputs([(1, count * 4), (1, count * classes), (1, 2, count * 4)])
        model += layer('detection', 'DetectionOutput', ['data0', 'data1', 'data2'], ['out'],
            'detection_output_param { num_classes: %d code_type: CENTER_SIZE %s }' % (classes, options))
        values = [torch.tensor(locations).reshape(1, -1), torch.tensor(confidence).reshape(1, -1),
                  torch.tensor([boxes, variances]).reshape(1, 2, -1)]
        if rows:
            ref = torch.tensor(rows).reshape(1, 1, -1, 7)
            # Caffe emits class-major rows; native ncnn emits globally sorted rows
            native = torch.tensor(sorted(rows, key=lambda row: -row[2]))[:, 1:]
        else:
            ref = torch.full((1, 1, 1, 7), -1.)
            ref[0, 0, 0, 0] = 0.
            native = torch.empty(0)
        cases.append((name, model, values, ref, native))

    # center (.3,.4) moves to (.32,.36), width doubles, height halves
    detection_case('detection_nonzero_decode', [[.1, .2, .5, .6]], [[.1, .2, .3, .4]],
                   [[.5, -.5, math.log(2) / .3, math.log(.5) / .4]], [[.1, .9]],
                   [[0., 1., .9, -.08, .26, .72, .46]],
                   'confidence_threshold: 0.5 keep_top_k: 10 nms_param { top_k: 10 nms_threshold: 0.3 }')
    boxes = [[0., 0., .25, .25], [.5, .5, .75, .75], [0., 0., .25, .25]]
    rows = [[0., 1., .8] + boxes[0], [0., 1., .7] + boxes[1], [0., 2., .9] + boxes[2]]
    confidence = [[.05, .8, .15], [.1, .7, .2], [.05, .05, .9]]
    for name, options, expected in [
        ('detection_multiclass', 'keep_top_k: 10 nms_param { top_k: 10 }', rows),
        ('detection_top_k_one', 'keep_top_k: 10 nms_param { top_k: 1 }', [rows[0], rows[2]]),
        ('detection_keep_top_k_one', 'keep_top_k: 1 nms_param { top_k: 10 }', [rows[2]]),
        ('detection_default_limits', '', rows),
        ('detection_top_k_zero', 'nms_param { top_k: 0 }', []),
        ('detection_keep_top_k_zero', 'keep_top_k: 0', []),
    ]:
        detection_case(name, boxes, [[.1] * 4] * 3, [[0.] * 4] * 3, confidence, expected,
                       'confidence_threshold: 0.5 ' + options)
    # exact binary fractions give IoU exactly .5: equality must not suppress
    boxes = [[0., 0., 1., 1.], [0., 0., .5, 1.]]
    detection_case('detection_nms_equality', boxes, [[.1] * 4] * 2, [[0.] * 4] * 2,
                   [[.1, .9], [.2, .8]], [[0., 1., .9] + boxes[0], [0., 1., .8] + boxes[1]],
                   'confidence_threshold: 0.5 keep_top_k: 10 nms_param { top_k: 10 nms_threshold: 0.5 }')
    results = []
    runtime_warnings = 0
    for name, proto, values, ref_torch, ref_ncnn in cases:
        if args.filter and not re.search(args.filter, name):
            continue
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            if name == 'detection_unnormalized_confidence':
                result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16],
                                        capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                assert 'warning:' in result.stderr and 'DetectionOutput' in result.stderr, result.stderr
                assert 'normalized class probabilities' in result.stderr and 'unsupported' in result.stderr, result.stderr
                runtime_warnings += 1
                continue
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, actual = inference(prefix, values)
            assert len(pytorch) == len(actual) == 1
            results.append(compare(prefix + '/torch', pytorch[0], ref_torch, 0))
            results.append(compare(prefix + '/ncnn', actual[0], ref_ncnn, fp16))
            if not name.startswith('prior') and ref_ncnn.numel():
                # Exercise the generated helper too: its output batch metadata
                # must describe native [R,6] detections without an extra axis.
                spec = importlib.util.spec_from_file_location(prefix + '_ncnn', prefix + '_ncnn.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with mock.patch.object(module.torch, 'rand', side_effect=values):
                    generated = module.test_inference()
                results.append(compare(prefix + '/ncnn_python', generated, ref_ncnn, fp16))
            if not name.startswith('prior'):
                # batch IDs and class indices are discrete, not tolerance based
                assert torch.equal(pytorch[0][..., :2], ref_torch[..., :2])
                if actual[0].numel():
                    assert torch.equal(actual[0][:, 0], ref_ncnn[:, 0])
    # legacy serialization ignored these options; reject unsupported semantics
    # explicitly instead of treating the old converter's silent output as truth
    diagnostics_passed = 0
    diagnostics_skipped = 0
    for name, options in [
        ('reject_code_type_default', ''),
        ('reject_code_type_corner', 'code_type: CORNER'),
        ('reject_share_location', 'code_type: CENTER_SIZE share_location: false'),
        ('reject_background', 'code_type: CENTER_SIZE background_label_id: 1'),
        ('reject_variance_encoded', 'code_type: CENTER_SIZE variance_encoded_in_target: true'),
        ('reject_eta', 'code_type: CENTER_SIZE nms_param { eta: 0.5 }'),
    ]:
        if args.filter and not re.search(args.filter, name):
            diagnostics_skipped += 1
            continue
        model = inputs([(1, 4), (1, 2), (1, 2, 4)])
        model += layer('unsupported_detection', 'DetectionOutput', ['data0', 'data1', 'data2'], ['out'],
                       'detection_output_param { num_classes: 2 %s }' % options)
        Path(name + '.prototxt').write_text(model)
        Path(name + '.caffemodel').write_bytes(b'')
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0, name
        assert 'unsupported_detection' in result.stderr and 'DetectionOutput' in result.stderr, result.stderr
        print(name + ': diagnostic passed')
        diagnostics_passed += 1
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  failed=sum(not r['passed'] for r in results),
                  skipped=sum(bool(args.filter and not re.search(args.filter, c[0])) for c in cases),
                  diagnostics_passed=diagnostics_passed, diagnostics_skipped=diagnostics_skipped,
                  runtime_warnings_passed=runtime_warnings)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--filter', help='Run matching fixture names')
    exit(0 if test(parser.parse_args()) else 1)
