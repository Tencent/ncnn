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
import torch.nn.functional as F

from test_utils import inputs
from test_utils import blob, inference, layer
from test_roi import reference
from test_utils import compare, run


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    schema = Path(args.schema).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    feature = torch.arange(4 * 128 * 128).float().reshape(1, 4, 128, 128)
    feature = (feature + torch.arange(128).float() * 127) / 65536
    rois = torch.tensor([[0., 8., 8., 40., 40.], [0., 72., 8., 104., 40.]])
    results = []
    for kind, explicit in product(('ROIPooling', 'PSROIPooling', 'ROIAlign'), (False, True)):
        channels = 1 if kind == 'PSROIPooling' else 4
        width = channels * 4
        weights = [torch.rand(6, width) - .5, torch.rand(6) - .5,
                   torch.rand(3, 6) - .5, torch.rand(3) - .5,
                   torch.rand(4, 6) - .5, torch.rand(4) - .5]
        proto = inputs([feature.shape, (1, 18, 1, 2), (1, 36, 1, 2), (1, 3)])
        proto += layer('proposal', 'Python', ['data1', 'data2', 'data3'], ['rois'],
                       'python_param { module: "rpn.proposal_layer" layer: "ProposalLayer" param_str: "feat_stride: 64" }')
        if kind == 'PSROIPooling':
            params = 'psroi_pooling_param { output_dim: 1 group_size: 2 spatial_scale: 1 }'
        else:
            field = 'roi_align_param' if kind == 'ROIAlign' else 'roi_pooling_param'
            params = field + ' { pooled_h: 2 pooled_w: 2 spatial_scale: 1 }'
        proto += layer('roi', kind, ['data0', 'rois'], ['pooled'], params)
        proto += layer('roi_relu', 'ReLU', ['pooled'], ['pooled'])
        if explicit:
            proto += layer('roi_prob', 'Softmax', ['pooled'], ['pooled'], 'softmax_param { axis: 1 }')
            proto += layer('flatten', 'Flatten', ['pooled'], ['flat'], 'flatten_param { axis: 1 end_axis: -1 }')
        proto += layer('fc1', 'InnerProduct', ['flat' if explicit else 'pooled'], ['hidden'],
                       'inner_product_param { num_output: 6 axis: 1 }')
        proto += layer('relu', 'ReLU', ['hidden'], ['hidden'])
        proto += layer('class', 'InnerProduct', ['hidden'], ['logits'], 'inner_product_param { num_output: 3 axis: 1 }')
        proto += layer('prob', 'Softmax', ['logits'], ['prob'], 'softmax_param { axis: 1 }')
        proto += layer('box', 'InnerProduct', ['hidden'], ['box'], 'inner_product_param { num_output: 4 axis: 1 }')
        model = ''.join(layer(name, 'InnerProduct', [], [], blob(weights[i * 2]) + blob(weights[i * 2 + 1]))
                        for i, name in enumerate(('fc1', 'class', 'box')))
        name = kind + ('_explicit' if explicit else '_implicit')
        Path(name + '.prototxt').write_text(proto)
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                 input=model.encode(), stdout=subprocess.PIPE, check=True).stdout
        Path(name + '.caffemodel').write_bytes(encoded)
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16],
                                    capture_output=True, text=True)
            assert result.returncode != 0, result.stderr
            assert 'caffe layer proposal (caffe.Python)' in result.stderr, result.stderr
            assert 'unsupported runtime multi-ROI support' in result.stderr, result.stderr
            results.append(dict(name=prefix + '/dynamic_roi_diagnostic', passed=True, diagnostic=True))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
                  numerical_passed=sum(r['passed'] and not r.get('diagnostic', False) for r in results),
                  diagnostic_passed=sum(r['passed'] and r.get('diagnostic', False) for r in results),
                  failed=sum(not r['passed'] for r in results), skipped=0)
    Path('results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report['failed'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--protoc', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output-dir', required=True)
    exit(0 if test(parser.parse_args()) else 1)
