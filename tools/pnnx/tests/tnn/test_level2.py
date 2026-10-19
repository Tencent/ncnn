# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
from pathlib import Path
import struct
import subprocess

import ncnn
import torch


def test(args):
    pnnx = str(Path(args.pnnx).resolve())
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(1)
    x = torch.linspace(-4, 4, 24).reshape(1, 3, 2, 4)
    cases = [
        ('softmax_channel', ['SoftmaxCaffe softmax 1 1 data out 1'], x.softmax(1)),
        ('softmax_last', ['SoftmaxCaffe softmax 1 1 data out -1'], x.softmax(-1)),
        ('concat_softmax', ['Concat concat 2 1 data data joined 1',
                            'SoftmaxCaffe softmax 1 1 joined out 1'],
         torch.cat((x, x), 1).softmax(1)),
    ]
    for name, layers, reference in cases:
        # load_tnn reads a single input, blob names, outputs, then layer records.
        lines = ['1 4 1 4206624772', 'data 4 1 3 2 4 0',
                 'data joined out', 'out', str(len(layers))] + layers
        Path(name + '.tnnproto').write_text(''.join('"%s ,"\n' % line for line in lines))
        # Empty weight-resource and constant-resource sections, each with magic.
        Path(name + '.tnnmodel').write_bytes(struct.pack('<IIII', 0xfabc0004, 0, 0xfabc0004, 0))
        for fp16 in (0, 1):
            prefix = name + '_fp%d' % fp16
            subprocess.run([pnnx, name + '.tnnproto', 'optlevel=2', 'fp16=%d' % fp16,
                            'pnnxparam=' + prefix + '.pnnx.param',
                            'pnnxbin=' + prefix + '.pnnx.bin',
                            'pnnxpy=' + prefix + '_pnnx.py',
                            'ncnnparam=' + prefix + '.ncnn.param',
                            'ncnnbin=' + prefix + '.ncnn.bin',
                            'ncnnpy=' + prefix + '_ncnn.py'], check=True)
            spec = importlib.util.spec_from_file_location(prefix, prefix + '_pnnx.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with torch.no_grad():
                actual = module.Model().eval()(x)
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert actual.device == reference.device
            assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
            with ncnn.Net() as net:
                net.opt.num_threads = 1
                net.opt.use_vulkan_compute = False
                net.opt.use_fp16_storage = False
                net.opt.use_fp16_arithmetic = False
                assert net.load_param(prefix + '.ncnn.param') == 0
                assert net.load_model(prefix + '.ncnn.bin') == 0
                with net.create_extractor() as ex:
                    # TNN does not mark an implicit batch axis; preserve all dims,
                    # matching the generated ncnn Python adapter.
                    assert ex.input(net.input_names()[0], ncnn.Mat(x.numpy(), batch_index=233).clone()) == 0
                    ret, result = ex.extract(net.output_names()[0])
                    assert ret == 0
                    actual_ncnn = torch.from_numpy(result.numpy(batch_index=233).copy())
                assert actual_ncnn.shape == reference.shape and actual_ncnn.dtype == reference.dtype
                assert actual_ncnn.device == reference.device
                assert torch.allclose(actual_ncnn, reference, atol=1e-5, rtol=1e-4)
            print('%s: PyTorch and ncnn passed' % prefix)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--output-dir', required=True)
    test(parser.parse_args())
