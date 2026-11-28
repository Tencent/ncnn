# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
from itertools import product
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
    scores = torch.zeros(1, 18, 1, 1)
    scores[0, 9:, 0, 0] = torch.arange(1, 10).float() / 10
    values = [scores, torch.zeros(1, 36, 1, 1), torch.tensor([[32., 32., 1.]])]
    # all nine default anchors at this single location exceed the 32x32 image;
    # clipping makes each [0,0,31,31]. NMS retains only the highest score, 0.9
    reference = [torch.tensor([[0., 0., 0., 31., 31.]]), torch.tensor([[.9]])]
    cases = [('clipped', values, reference)]
    # clipping to an 8x8 image makes every box smaller than min_size=16
    # an empty candidate set is a valid result, not a failed allocation
    empty_values = [values[0], values[1], torch.tensor([[8., 8., 1.]])]
    cases.append(('empty', empty_values, [torch.empty(0, 5), torch.empty(0, 1)]))
    scores = torch.zeros(1, 18, 1, 2)
    scores[0, 12, 0] = torch.tensor([.9, .8])
    deltas = torch.zeros(1, 36, 1, 2)
    deltas.reshape(1, 9, 4, 1, 2)[:, :, 2:] = -10
    deltas[:, 12, 0] = torch.tensor([.125, 2.9 / 128])
    deltas[:, 13] = .125
    deltas[:, 14:16] = torch.log(torch.tensor(.125))
    # exactly two valid 16x16 proposals: [16,16,32,32] and
    # [18.9,16,34.9,32]. Exclusive IoU=13.1/18.9 < .7; Caffe's
    # inclusive IoU=14.1/19.9 > .7. The latter keeps only score .9
    # all other anchors fail the independent min-size filter
    values = [scores, deltas, torch.tensor([[128., 128., 1.]])]
    reference = [torch.tensor([[0., 16., 16., 32., 32.]]), torch.tensor([[.9]])]
    cases.append(('inclusive_nms', values, reference))
    # this stable tie order is a pnnx two-export contract, not a claim about
    # every Caffe fork or NumPy sorting implementation
    for name, displacement in [('tie_disjoint', 32.), ('tie_overlap', 2.)]:
        tie_scores = scores.clone()
        tie_scores[0, 12, 0] = .9
        tie_deltas = deltas.clone()
        tie_deltas[:, 12, 0, 1] = displacement / 128
        boxes = [[0., 16., 16., 32., 32.]]
        if displacement == 32.:
            boxes.append([0., 48., 16., 64., 32.])
        reference = [torch.tensor(boxes), torch.full((len(boxes), 1), .9)]
        cases.append((name, [tie_scores, tie_deltas, values[2]], reference))
    results = []
    for (case, values, reference), count in product(cases, (1, 2)):
        name = 'proposal_%s_%d' % (case, count)
        native = [reference[0][:, None, 1:], reference[1][:, None]]
        proto = inputs([v.shape for v in values])
        proto += layer('proposal', 'Python', ['data0', 'data1', 'data2'], ['rois', 'scores'][:count],
            'python_param { module: "rpn.proposal_layer" layer: "ProposalLayer" param_str: "\'feat_stride\': 16" }')
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(b'')
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            conversion = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'], capture_output=True, text=True, check=True)
            assert 'caffe layer proposal (caffe.Python)' in conversion.stderr, conversion.stderr
            assert 'exclusive NMS' in conversion.stderr and 'empty candidates may fail' in conversion.stderr, conversion.stderr
            for text in Path(prefix + '.ncnn.param').read_text().splitlines():
                if text.startswith('Proposal '):
                    assert ' 6=' not in text and ' 7=' not in text, text
            if case != 'clipped':
                # runtime-value-dependent legacy limitations are warnings, not
                # numerical compatibility claims against the Caffe reference
                results.append(dict(name=prefix + '/legacy_warning', passed=True, diagnostic=True))
                continue
            pytorch, actual = inference(prefix, values)
            assert len(pytorch) == len(actual) == count
            for i in range(count):
                results.append(compare(prefix + '/torch/%d' % i, pytorch[i], reference[i], 0))
                results.append(compare(prefix + '/ncnn/%d' % i, actual[i], native[i], fp16))
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
