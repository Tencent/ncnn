# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def test(args):
    pnnx = shutil.which(args.pnnx)
    protoc = shutil.which(args.protoc)
    assert pnnx and protoc, 'pnnx and protoc are required'
    if not Path('/dev/full').exists():
        print('SKIP: /dev/full is unavailable')
        return
    schema = Path(args.schema).resolve()
    with tempfile.TemporaryDirectory(prefix='pnnx-export-errors-') as directory:
        directory = Path(directory)
        proto = '''input: "data" input_shape { dim: 1 dim: 1 dim: 2 dim: 2 }
layer { name: "conv" type: "Convolution" bottom: "data" top: "out"
convolution_param { num_output: 1 kernel_size: 1 bias_term: false } }
'''
        weights = '''layer { name: "conv" type: "Convolution"
blobs { shape { dim: 1 dim: 1 dim: 1 dim: 1 } data: 2 } }
'''
        (directory / 'model.prototxt').write_text(proto)
        encoded = subprocess.run([protoc, '--proto_path=' + str(schema.parent),
                                  '--encode=caffe.NetParameter', str(schema)],
                                 input=weights.encode(), stdout=subprocess.PIPE, check=True).stdout
        (directory / 'model.caffemodel').write_bytes(encoded)
        outputs = {'pnnxparam': 'model.pnnx.param', 'pnnxbin': 'model.pnnx.bin',
                   'pnnxpy': 'model_pnnx.py', 'ncnnparam': 'model.ncnn.param',
                   'ncnnbin': 'model.ncnn.bin', 'ncnnpy': 'model_ncnn.py'}
        for failed in [None] + list(outputs):
            options = [key + '=' + ('/dev/full' if key == failed else value)
                       for key, value in outputs.items()]
            result = subprocess.run([pnnx, 'model.prototxt', 'optlevel=2', 'fp16=0'] + options,
                                    cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if failed is None:
                assert result.returncode == 0, result.stderr.decode()
            else:
                assert result.returncode != 0, failed + ': unexpectedly succeeded'
                assert b'incomplete' in result.stderr, result.stderr.decode()
        optional = directory / 'optional'
        optional.mkdir()
        options = [key + '=' + str(optional / value) for key, value in outputs.items()]
        result = subprocess.run([pnnx, 'model.prototxt', 'optlevel=2', 'fp16=0',
                                 'pnnxonnx=' + str(optional)] + options,
                                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0, result.stderr.decode()
        assert b'Optional Caffe ONNX export failed' in result.stderr, result.stderr.decode()
        for value in outputs.values():
            assert (optional / value).stat().st_size > 0, value
        print('PASS: normal export, six output write failures and optional ONNX failure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnnx', required=True)
    parser.add_argument('--protoc', required=True)
    parser.add_argument('--schema', required=True)
    test(parser.parse_args())
