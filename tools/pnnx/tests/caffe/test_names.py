# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from test_utils import blob, inference, layer
from test_utils import compare, run


def input_layer(name, shape):
    return layer('input_' + name, 'Input', [], [name],
                 'input_param { shape { %s } }' % ' '.join('dim: %d' % d for d in shape))


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    schema = Path(args.schema).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3) - 3
    cases = []

    # distinct module names must retain their own weights after Python export
    # include Python keywords, nn.Module attributes and a leading digit
    names = ['a/b', 'a_b', 'a.b', 'a:b', 'forward', 'training', 'modules', '1conv', 'class', 'a/b_weight']
    proto = input_layer('data', x.shape)
    weights = ''
    references = []
    for i, name in enumerate(names):
        proto += layer(name, 'Convolution', ['data'], ['out%d' % i],
                       'convolution_param { num_output: 1 kernel_size: 1 bias_term: true }')
        weights += layer(name, 'Convolution', [], [],
                         blob(torch.full((1, 1, 1, 1), float(i + 1))) + blob(torch.tensor([float(i)])))
        references.append(x * (i + 1) + i)
    cases.append(('module_names', proto, weights, [x], references))

    # both branch values remain live at the concatenation. Reusing one Python
    # variable for a/b and a_b would silently concatenate the same value twice
    proto = input_layer('data', x.shape)
    proto += layer('relu', 'ReLU', ['data'], ['a/b'])
    proto += layer('negate', 'Power', ['data'], ['a_b'], 'power_param { scale: -1 }')
    proto += layer('concat', 'Concat', ['a/b', 'a_b'], ['out'], 'concat_param { axis: 1 }')
    cases.append(('operand_names', proto, '', [x], [torch.cat((x.relu(), -x), dim=1)]))

    # colliding input names used to produce duplicate Python argument names
    proto = input_layer('a/b', x.shape) + input_layer('a_b', x.shape)
    proto += layer('add', 'Eltwise', ['a/b', 'a_b'], ['out'], 'eltwise_param { operation: SUM }')
    cases.append(('input_names', proto, '', [x, x + 2], [x * 2 + 2]))

    # other legal Caffe names must remain valid identifiers and valid PNNX text
    proto = input_layer('1 input', x.shape)
    proto += layer('relu-with-space', 'ReLU', ['1 input'], ['a-b'])
    proto += layer('negate', 'Power', ['1 input'], ['a b'], 'power_param { scale: -1 }')
    proto += layer('concat', 'Concat', ['a-b', 'a b'], ['class'], 'concat_param { axis: 1 }')
    cases.append(('identifier_characters', proto, '', [x], [torch.cat((x.relu(), -x), dim=1)]))

    results = []
    for name, proto, weights, values, references in cases:
        Path(name + '.prototxt').write_text(proto)
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                                 input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
        Path(name + '.caffemodel').write_bytes(encoded)
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                 'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                 'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                 'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
            pytorch, actual = inference(prefix, values)
            assert len(pytorch) == len(actual) == len(references)
            for i, reference in enumerate(references):
                results.append(compare(prefix + '/torch/%d' % i, pytorch[i], reference, 0))
                results.append(compare(prefix + '/ncnn/%d' % i, actual[i], reference[0], fp16))
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
