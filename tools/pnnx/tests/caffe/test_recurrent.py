# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import ncnn
import numpy as np
import torch

from test_utils import blob, input_layer, layer, compare, run


def reference(x, weight, bias, recurrent):
    # independent step equations and original IFOG storage order, following
    # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_unit_layer.cpp
    # one-input legacy inference starts with zero state and continues thereafter
    hidden = bias.size // 4
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)
    sequence = []
    for t, value in enumerate(x):
        gates = weight @ value + bias + recurrent @ h
        i, f, o, g = np.split(gates, 4)
        i = 1 / (1 + np.exp(-i))
        f = 1 / (1 + np.exp(-f))
        o = 1 / (1 + np.exp(-o))
        c = f * c + i * np.tanh(g)
        h = o * np.tanh(c)
        sequence.append(h.copy())
    return torch.from_numpy(np.stack(sequence))


def inference(prefix, x, batched):
    spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        model = module.Model().eval()
        args = (x,)
        result = model(*args)
        # repeated inference must start from zero state, not carry hidden state
        torch.testing.assert_close(model(*args), result, atol=0, rtol=0)
    with ncnn.Net() as net:
        net.opt.num_threads = 1
        net.opt.use_vulkan_compute = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param(prefix + '.ncnn.param') == 0
        assert net.load_model(prefix + '.ncnn.bin') == 0
        assert len(net.input_names()) == 1
        assert len(net.output_names()) == 1
        previous = None
        for repeat in range(2):
            with net.create_extractor() as ex:
                # ncnn's sequence is T,I; remove only the known singleton N axis
                value = x[:, 0] if batched else x
                assert ex.input(net.input_names()[0], ncnn.Mat(value.numpy()).clone()) == 0
                ret, output = ex.extract(net.output_names()[0])
                assert ret == 0
                actual = torch.from_numpy(np.array(output).copy())
            if previous is not None:
                torch.testing.assert_close(actual, previous, atol=0, rtol=0)
            previous = actual
    return result, actual


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    schema = Path(args.schema).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    torch.manual_seed(13)
    input_size, hidden_size = 3, 4
    weight = torch.rand(4 * hidden_size, input_size) - .5
    bias = torch.rand(4 * hidden_size) - .5
    recurrent = torch.rand(4 * hidden_size, hidden_size) - .5
    weights = layer('lstm', 'LSTM', [], [], blob(weight) + blob(bias) + blob(recurrent))
    encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent), '--encode=caffe.NetParameter', str(schema)],
                             input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
    results = []
    for batched in (False, True):
        for length in (1, 5):
            x = torch.rand(length, input_size) * 2 - 1
            ref = reference(x.numpy(), weight.numpy(), bias.numpy(), recurrent.numpy())
            if batched:
                x = x.unsqueeze(1)
            name = 'lstm_%d_%d' % (batched, length)
            proto = input_layer(x.shape) + layer('lstm', 'LSTM', ['data'], ['out'],
                'recurrent_param { num_output: %d }' % hidden_size)
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                prefix = name + '_fp%d' % fp16
                run([pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16,
                     'pnnxparam=' + prefix + '.pnnx.param', 'pnnxbin=' + prefix + '.pnnx.bin',
                     'pnnxpy=' + prefix + '_pnnx.py', 'ncnnparam=' + prefix + '.ncnn.param',
                     'ncnnbin=' + prefix + '.ncnn.bin', 'ncnnpy=' + prefix + '_ncnn.py'])
                pytorch, native = inference(prefix, x, batched)
                results.append(compare(prefix + '/torch', pytorch, ref.unsqueeze(1) if batched else ref, 0))
                results.append(compare(prefix + '/ncnn', native, ref, fp16))

    # continuation cannot be represented by the unchanged native LSTM protocol
    # check both supported input ranks and both mask ranks before any export
    for data_rank in (2, 3):
        for mask_rank in (1, 2):
            shape = (5, input_size) if data_rank == 2 else (5, 1, input_size)
            mask_shape = (5,) if mask_rank == 1 else (5, 1)
            name = 'unsupported_continuation_%d_%d' % (data_rank, mask_rank)
            proto = input_layer(shape)
            proto += layer('continuation_input', 'Input', [], ['cont'],
                           'input_param { shape { %s } }' % ' '.join('dim: %d' % d for d in mask_shape))
            proto += layer('lstm', 'LSTM', ['data', 'cont'], ['out'], 'recurrent_param { num_output: 4 }')
            Path(name + '.prototxt').write_text(proto)
            Path(name + '.caffemodel').write_bytes(encoded)
            for fp16 in (0, 1):
                # fresh directory avoids confusing stale successful exports with
                # artifacts from this rejected conversion
                with tempfile.TemporaryDirectory(prefix=name + '_', dir=output) as rejected:
                    rejected = Path(rejected)
                    paths = {key: rejected / key for key in
                             ('pnnxparam', 'pnnxbin', 'pnnxpy', 'ncnnparam', 'ncnnbin', 'ncnnpy')}
                    command = [pnnx, name + '.prototxt', 'optlevel=2', 'fp16=%d' % fp16]
                    command += ['%s=%s' % (key, path) for key, path in paths.items()]
                    result = subprocess.run(command, capture_output=True, text=True)
                    assert result.returncode != 0, name
                    assert 'lstm' in result.stderr and 'caffe.LSTM' in result.stderr, result.stderr
                    assert 'continuation' in result.stderr and 'unsupported' in result.stderr, result.stderr
                    assert not any(path.exists() for path in paths.values()), list(rejected.iterdir())
                results.append(dict(name=name + '_fp%d/diagnostic' % fp16, passed=True))

    # exposed state and larger sequence batches remain unsupported
    for name, proto in [
        ('continuation_shape', input_layer((5, 1, input_size)) +
         layer('continuation_input', 'Input', [], ['cont'], 'input_param { shape { dim: 4 dim: 1 } }') +
         layer('lstm', 'LSTM', ['data', 'cont'], ['out'], 'recurrent_param { num_output: 4 }')),
        ('exposed_state', input_layer((5, 1, input_size)) +
         layer('lstm', 'LSTM', ['data'], ['out'], 'recurrent_param { num_output: 4 expose_hidden: true }')),
        ('batch_two', input_layer((5, 2, input_size)) +
         layer('lstm', 'LSTM', ['data'], ['out'], 'recurrent_param { num_output: 4 }')),
        ('invalid_rank', input_layer((1, 1, 5, input_size)) +
         layer('lstm', 'LSTM', ['data'], ['out'], 'recurrent_param { num_output: 4 }')),
        ('invalid_hidden_size', input_layer((5, input_size)) +
         layer('lstm', 'LSTM', ['data'], ['out'], 'recurrent_param { num_output: 0 }')),
        ('invalid_weight_size', input_layer((5, input_size + 1)) +
         layer('lstm', 'LSTM', ['data'], ['out'], 'recurrent_param { num_output: 4 }'))]:
        Path(name + '.prototxt').write_text(proto)
        Path(name + '.caffemodel').write_bytes(encoded)
        result = subprocess.run([pnnx, name + '.prototxt', 'optlevel=2'], capture_output=True, text=True)
        assert result.returncode != 0 and 'LSTM' in result.stderr, result.stderr
        reasons = {'invalid_rank': 'expected rank2 or rank3',
                   'invalid_hidden_size': 'must be positive',
                   'invalid_weight_size': 'weight blob sizes'}
        if name in reasons:
            assert reasons[name] in result.stderr and 'lstm' in result.stderr, result.stderr
        results.append(dict(name=name + '/diagnostic', passed=True))

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
