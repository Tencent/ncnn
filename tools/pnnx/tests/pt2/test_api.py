# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import pathlib
import subprocess
import sys
from unittest import mock

import torch
from common import rewrite_archive


class IntegerModel(torch.nn.Module):
    def forward(self, x):
        return x + 3


def rejects(convert_module, path, **kwargs):
    try:
        convert_module.convert(path, **kwargs)
    except subprocess.CalledProcessError as error:
        return error.returncode != 2
    return False


def test():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'python'))
    convert_module = importlib.import_module('pnnx.utils.convert')
    convert_module.EXEC_PATH = '../../src/pnnx'

    net = torch.nn.Linear(4, 3)
    net.eval()
    torch.manual_seed(0)
    x = torch.rand(1, 4)
    a = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, 'test_pt2_api.pt2')
    converted = convert_module.convert('test_pt2_api.pt2', fp16=False)
    b = converted(x)
    if not torch.allclose(a, b, 2e-5, 2e-5):
        return False

    # Both API argument paths must accept the explicit optional-backend status.
    integer = IntegerModel().eval()
    indices = torch.tensor([[2**40, 1, -4]], dtype=torch.int64)
    torch.export.save(torch.export.export(integer, (indices,)), 'test_pt2_api_integer.pt2')
    for inputs in (None, [indices]):
        converted = convert_module.convert('test_pt2_api_integer.pt2', inputs=inputs, fp16=False)
        actual = converted(indices)
        if actual.dtype != torch.int64 or not torch.equal(actual, integer(indices)):
            return False

    # A stale importable module must never turn a real subprocess failure into
    # success, even when stderr resembles an optional-backend diagnostic.
    for status in (1, 255, -9):
        error = subprocess.CalledProcessError(status, ['pnnx'], output='failure output',
                                            stderr='pt2 ncnn: unsupported stale diagnostic')
        for inputs in (None, [x]):
            with mock.patch.object(convert_module.subprocess, 'run', side_effect=error):
                try:
                    convert_module.convert('test_pt2_api.pt2', inputs=inputs)
                except subprocess.CalledProcessError as caught:
                    if caught is not error:
                        return False
                else:
                    return False

    stale = pathlib.Path('test_pt2_api_stale_pnnx.py')
    stale.write_text("raise AssertionError('stale output was imported')\n")
    rewrite_archive('test_pt2_api.pt2', 'test_pt2_api_bad.pt2', lambda p, e: p['opset_version'].update(aten=999))
    if not rejects(convert_module, 'test_pt2_api_bad.pt2', pnnxpy=str(stale)):
        return False

    # Reading valid weights must not clear an earlier graph metadata error.
    rewrite_archive('test_pt2_api.pt2', 'test_pt2_api_bad_guards.pt2', lambda p, e: p.update(guards_code=None))
    if not rejects(convert_module, 'test_pt2_api_bad_guards.pt2', inputs=[x], pnnxpy=str(stale)):
        return False

    # Failure to write either pnnx or an otherwise supported backend is a hard
    # error, even when a usable pnnx module from a previous run exists.
    directory = pathlib.Path('test_pt2_api_output_directory')
    directory.mkdir(exist_ok=True)
    for output in ('pnnxparam', 'pnnxbin', 'pnnxpy', 'ncnnparam', 'ncnnbin', 'ncnnpy'):
        if not rejects(convert_module, 'test_pt2_api.pt2', **{output: str(directory)}):
            return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
