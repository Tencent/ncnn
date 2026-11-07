# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from test_utils import blob, input_layer, layer, inference
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
    weight, bias = torch.rand(7, 5), torch.rand(5)
    results = []
    for batched in (False, True):
        for use_bias in (False, True):
            # Caffe's float-valued integer indices become explicit int32 values
            # at the exported model boundary. Both paths see the same indices
            x = torch.tensor([0, 6, 3, 0], dtype=torch.int32)
            if batched:
                x = x.unsqueeze(0)
            ref = weight[x.long()] + (bias if use_bias else 0.)
            name = 'embed%d_%d' % (batched, use_bias)
            params = 'embed_param { input_dim: 7 num_output: 5 bias_term: %s }' % str(use_bias).lower()
            proto = input_layer(x.shape) + layer('embed', 'Embed', ['data'], ['out'], params)
            weights = layer('embed', 'Embed', [], [], blob(weight) + (blob(bias) if use_bias else ''))
            encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                     input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, x, [0 if batched else None])
                assert len(pytorch) == len(actual) == 1
                results.append(compare(prefix + '/torch', pytorch[0], ref, 0))
                results.append(compare(prefix + '/ncnn', actual[0], ref[0] if batched else ref, fp16))
    for batched in (False, True):
        for routing in ('shared', 'split', 'nested_split'):
            x = torch.tensor([0, 6, 3, 0], dtype=torch.int32)
            if batched:
                x = x.unsqueeze(0)
            name = 'embed_%s_%d' % (routing, batched)
            proto = input_layer(x.shape)
            bottoms = ['data', 'data']
            if routing != 'shared':
                proto += layer('split', 'Split', ['data'], ['left', 'right'])
                bottoms = ['left', 'right']
            if routing == 'nested_split':
                proto += layer('split_again', 'Split', ['right'], ['right0', 'right1'])
                bottoms = ['left', 'right0', 'right1']
            weights = ''
            references = []
            for i, bottom in enumerate(bottoms):
                table = weight + i
                use_bias = i % 2 == 0
                params = 'embed_param { input_dim: 7 num_output: 5 bias_term: %s }' % str(use_bias).lower()
                proto += layer('embed%d' % i, 'Embed', [bottom], ['out%d' % i], params)
                weights += layer('embed%d' % i, 'Embed', [], [], blob(table) + (blob(bias) if use_bias else ''))
                references.append(table[x.long()] + (bias if use_bias else 0.))
            encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                     input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, actual = inference(prefix, x, [0 if batched else None])
                assert len(pytorch) == len(actual) == len(references)
                for i, ref in enumerate(references):
                    results.append(compare(prefix + '/torch/%d' % i, pytorch[i], ref, 0))
                    results.append(compare(prefix + '/ncnn/%d' % i, actual[i], ref[0] if batched else ref, fp16))
    # an identity-looking floating-point computation must not silently become
    # an integer-storage path. Reject mixed consumers before adapting the input
    for routing in ('computed', 'mixed', 'mixed_split'):
        name = 'embed_invalid_' + routing
        proto = input_layer((4,))
        bottom = 'data'
        if routing == 'computed':
            proto += layer('floating', 'Power', ['data'], ['computed'], 'power_param { scale: 1 }')
            bottom = 'computed'
        elif routing == 'mixed':
            proto += layer('floating', 'ReLU', ['data'], ['float_out'])
        else:
            proto += layer('split', 'Split', ['data'], ['indices', 'floats'])
            proto += layer('floating', 'ReLU', ['floats'], ['float_out'])
            bottom = 'indices'
        proto += layer('embed', 'Embed', [bottom], ['out'], 'embed_param { input_dim: 7 num_output: 5 bias_term: false }')
        weights = layer('embed', 'Embed', [], [], blob(weight))
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                 input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(encoded)
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=0'], capture_output=True, text=True)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert 'indices' in result.stderr and 'floating' in result.stderr, (name, result.stderr)
        results.append(dict(name=name, passed=True))
    report = dict(results=results, passed=sum(r['passed'] for r in results),
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
