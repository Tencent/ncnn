# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import pathlib
import sys

import torch
from packaging import version


class IntegerModel(torch.nn.Module):
    def forward(self, x):
        return x + 3


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

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

    # Both API argument paths preserve integer pnnx computation.
    integer = IntegerModel().eval()
    indices = torch.tensor([[2**40, 1, -4]], dtype=torch.int64)
    torch.export.save(torch.export.export(integer, (indices,)), 'test_pt2_api_integer.pt2')
    for inputs in (None, [indices]):
        converted = convert_module.convert('test_pt2_api_integer.pt2', inputs=inputs, fp16=False)
        actual = converted(indices)
        if actual.dtype != torch.int64 or not torch.equal(actual, integer(indices)):
            return False

    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
