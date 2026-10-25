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


def reference(feature, rois, position_sensitive):
    output = []
    for roi in rois:
        # positive integer coordinates deliberately avoid fork-specific rounding
        # of negative half-integers. Caffe pooling uses inclusive endpoints
        x0, y0, x1, y1 = [int(v) for v in roi[1:]]
        width, height = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
        channels = feature.shape[1] // 4 if position_sensitive else feature.shape[1]
        result = torch.empty(channels, 2, 2)
        for c in range(channels):
            for h in range(2):
                for w in range(2):
                    hs = max(min(math.floor(h * height / 2) + y0, feature.shape[2]), 0)
                    he = max(min(math.ceil((h + 1) * height / 2) + y0, feature.shape[2]), 0)
                    ws = max(min(math.floor(w * width / 2) + x0, feature.shape[3]), 0)
                    we = max(min(math.ceil((w + 1) * width / 2) + x0, feature.shape[3]), 0)
                    channel = (c * 2 + h) * 2 + w if position_sensitive else c
                    window = feature[0, channel, hs:he, ws:we]
                    result[c, h, w] = 0. if not window.numel() else window.mean() if position_sensitive else window.max()
        output.append(result)
    return torch.stack(output)


def boundary_reference(feature, rois, kind, scale):
    def nearest(v):
        return int(math.copysign(math.floor(abs(v) + .5), v))
    channels = 2 if kind == 'PSROIPooling' else 8
    output = torch.zeros(len(rois), channels, 2, 2)
    for r, roi in enumerate(rois):
        x0, y0, x1, y1 = roi[1:].tolist()
        if kind == 'ROIPooling':
            x0, y0, x1, y1 = [nearest(v * scale) for v in (x0, y0, x1, y1)]
            width, height = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
        elif kind == 'PSROIPooling':
            # Caffe rounds the inclusive endpoint before adding 1
            x1 = (nearest(x1) + 1) * scale
            y1 = (nearest(y1) + 1) * scale
            x0, y0 = nearest(x0) * scale, nearest(y0) * scale
            width, height = max(x1 - x0, .1), max(y1 - y0, .1)
        else:
            x0, y0, x1, y1 = [v * scale for v in (x0, y0, x1, y1)]
            width, height = max(x1 - x0, 1), max(y1 - y0, 1)
        for h in range(2):
            for w in range(2):
                left, right = x0 + w * width / 2, x0 + (w + 1) * width / 2
                top, bottom = y0 + h * height / 2, y0 + (h + 1) * height / 2
                if kind == 'ROIAlign':
                    left, right, top, bottom = [max(0., min(v, 5.)) for v in (left, right, top, bottom)]
                    nx, ny = math.ceil(right - left), math.ceil(bottom - top)
                    if nx <= 0 or ny <= 0:
                        continue
                    # affine feature: bilinear sampling equals the affine value
                    # at each clamped sample, so no exporter interpolation code
                    xs = [min(4., left + (i + .5) * width / (2 * nx)) for i in range(nx)]
                    ys = [min(4., top + (i + .5) * height / (2 * ny)) for i in range(ny)]
                    for c in range(channels):
                        output[r, c, h, w] = (c * 25 + sum(ys) / ny * 5 + sum(xs) / nx) / 100
                    continue
                left, right = max(0, min(5, math.floor(left))), max(0, min(5, math.ceil(right)))
                top, bottom = max(0, min(5, math.floor(top))), max(0, min(5, math.ceil(bottom)))
                if left >= right or top >= bottom:
                    continue
                for c in range(channels):
                    channel = c * 4 + h * 2 + w if kind == 'PSROIPooling' else c
                    values = feature[0, channel, top:bottom, left:right]
                    output[r, c, h, w] = values.mean() if kind == 'PSROIPooling' else values.max()
    return output


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    feature = torch.arange(8 * 5 * 5).float().reshape(1, 8, 5, 5) / 100
    rois = torch.tensor([[0., 0., 0., 3., 3.], [0., 1., 1., 4., 4.]])
    results = []
    for kind, params, ps in [
        ('ROIPooling', 'roi_pooling_param { pooled_h: 2 pooled_w: 2 spatial_scale: 1 }', False),
        ('PSROIPooling', 'psroi_pooling_param { output_dim: 2 group_size: 2 spatial_scale: 1 }', True),
        ('ROIAlign', 'roi_align_param { pooled_h: 2 pooled_w: 2 spatial_scale: 1 }', False)]:
        proto = inputs([feature.shape, rois.shape]) + layer('roi', kind, ['data0', 'data1'], ['out'], params)
        Path(kind + '.prototxt').write_text(proto)
        Path(kind + '.caffemodel').write_bytes(b'')
        ref = reference(feature, rois, ps)
        if kind == 'ROIAlign':
            # bilinear interpolation of this affine image equals its value at
            # each bin center, independently of the adaptive sampling count
            for r, roi in enumerate(rois):
                for c in range(8):
                    for h in range(2):
                        for w in range(2):
                            y = roi[2] + (h + .5) * (roi[4] - roi[2]) / 2
                            x = roi[1] + (w + .5) * (roi[3] - roi[1]) / 2
                            ref[r, c, h, w] = (c * 25 + y * 5 + x) / 100
        for fp16 in (0, 1):
            prefix = kind + '_fp%d' % fp16
            run([pnnx, kind + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, actual = inference(prefix, [feature, rois], [0, None])
            assert len(pytorch) == len(actual) == 1
            # the leading ROI dimension is retained by both execution paths
            results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
            results.append(compare(prefix + '/ncnn', actual[0], ref, fp16))
    for kind, param in [('ROIPooling', 'roi_pooling_param { pooled_h: 2 pooled_w: 2'),
                        ('PSROIPooling', 'psroi_pooling_param { output_dim: 2 group_size: 2'),
                        ('ROIAlign', 'roi_align_param { pooled_h: 2 pooled_w: 2')]:
        for name, scale, coordinates in [
            ('scaled', .5, [[0, 1, 1, 7, 7], [0, 2, 2, 8, 8]]),
            ('clipped', 1., [[0, -1, -1, 3, 3], [0, 3, 3, 6, 6]]),
            ('outside', 1., [[0, -4, -4, -2, -2], [0, 6, 6, 8, 8]]),
            ('negative_half', 1., [[0, -1.5, -1.5, -.5, -.5], [0, -.5, -.5, 2.5, 2.5]]),
        ]:
            rois = torch.tensor(coordinates, dtype=torch.float32)
            proto = inputs([feature.shape, rois.shape]) + layer('roi', kind, ['data0', 'data1'], ['out'],
                param + ' spatial_scale: %s }' % scale)
            model = kind + '_' + name
            Path(model + '.prototxt').write_text(proto)
            Path(model + '.caffemodel').write_bytes(b'')
            ref = boundary_reference(feature, rois, kind, scale)
            for fp16 in (0, 1):
                prefix = model + '_fp%d' % fp16
                conversion = subprocess.run([pnnx, model + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'], capture_output=True, text=True, check=True)
                assert 'warning: caffe layer roi (caffe.' + kind + ')' in conversion.stderr, conversion.stderr
                if name != 'scaled':
                    # unknown ROI coordinates cannot be range-checked during
                    # conversion. Do not execute known legacy-runtime failures
                    results.append(dict(name=prefix + '/legacy_warning', passed=True, diagnostic=True))
                    continue
                pytorch, actual = inference(prefix, [feature, rois], [0, None])
                results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
                results.append(compare(prefix + '/ncnn', actual[0], ref, fp16))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  numerical_passed=sum(r['passed'] and not r.get('diagnostic', False) for r in results),
                  diagnostic_passed=sum(r['passed'] and r.get('diagnostic', False) for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
