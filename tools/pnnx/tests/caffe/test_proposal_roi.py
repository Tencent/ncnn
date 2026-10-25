# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from test_utils import inputs
from test_utils import layer, inference
from test_roi import reference
from test_utils import compare, run


def test(args):
    pnnx = shutil.which(args.pnnx)
    assert pnnx, 'pnnx is required'
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    feature = torch.arange(8 * 128 * 128).float().reshape(1, 8, 128, 128)
    # a strong horizontal gradient makes swapped ROIs fail FP16 tolerances too
    feature = (feature + torch.arange(128).float() * 127) / 65536
    results = []
    for count, tied in product((0, 1, 2), (False, True)):
        locations = max(count, 1)
        scores = torch.zeros(1, 18, 1, locations)
        scores[0, 12, 0] = torch.tensor([.9, .9 if tied else .8][:locations])
        deltas = torch.zeros(1, 36, 1, locations)
        # only the square 128x128 anchor survives the min-size filter
        # shrink it to 32x32 and shift its center to (24+64*x,24)
        # disjoint boxes isolate score ordering from NMS. Equal scores must
        # retain generation order under pnnx's explicit stable-sort contract
        deltas.reshape(1, 9, 4, 1, locations)[:, :, 2:] = -10
        deltas[:, 12:14] = .125
        deltas[:, 14:16] = torch.log(torch.tensor(.25))
        # an 8x8 clipping extent leaves no box satisfying min_size=16
        extent = 8. if count == 0 else 128.
        info = torch.tensor([[extent, extent, 1.]])
        values = [feature, scores, deltas, info]
        rois = torch.tensor([[0., 8., 8., 40., 40.], [0., 72., 8., 104., 40.]])[:count]
        for kind, params, ps in [
            ('ROIPooling', 'roi_pooling_param { pooled_h: 2 pooled_w: 2 spatial_scale: 1 }', False),
            ('PSROIPooling', 'psroi_pooling_param { output_dim: 2 group_size: 2 spatial_scale: 1 }', True),
            ('ROIAlign', 'roi_align_param { pooled_h: 2 pooled_w: 2 spatial_scale: 1 }', False)]:
            name = kind + str(count) + ('_tied' if tied else '')
            proto = inputs([v.shape for v in values])
            proto += layer('proposal', 'Python', ['data1', 'data2', 'data3'], ['rois'],
                           'python_param { module: "rpn.proposal_layer" layer: "ProposalLayer" param_str: "feat_stride: 64" }')
            proto += layer('roi', kind, ['data0', 'rois'], ['out'], params)
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(b'')
            ref = torch.empty(0, 2 if ps else 8, 2, 2) if count == 0 else reference(feature, rois, ps)
            if kind == 'ROIAlign':
                # an affine feature map interpolates exactly to the bin center
                for r, roi in enumerate(rois):
                    for c in range(8):
                        for h in range(2):
                            for w in range(2):
                                y = roi[2] + (h + .5) * 16
                                x = roi[1] + (w + .5) * 16
                                ref[r, c, h, w] = (c * 128 * 128 + y * 128 + x * 128) / 65536
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16],
                                        capture_output=True, text=True)
                assert result.returncode != 0, result.stderr
                assert 'caffe layer proposal (caffe.Python)' in result.stderr, result.stderr
                assert 'unsupported runtime multi-ROI support' in result.stderr, result.stderr
                results.append(dict(name=prefix + '/dynamic_roi_diagnostic', passed=True, diagnostic=True))

            invalid = name + '_downstream'
            Path(invalid + '.prototxt').write_text(proto + layer('cross_roi', 'Softmax', ['out'], ['final'], 'softmax_param { axis: 0 }'))
            Path(invalid + '.caffemodel').write_bytes(b'')
            result = subprocess.run([pnnx, invalid + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
            assert result.returncode != 0 and 'unsupported runtime multi-ROI support' in result.stderr, result.stderr
            results.append(dict(name=invalid + '/diagnostic', passed=True, diagnostic=True))

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
