# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
import importlib
import math
from pathlib import Path
import subprocess

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x * 10
        y = y * 13
        y = y.to(dtype=x.dtype, memory_format=torch.contiguous_format)
        x = x.to(device='cpu', dtype=torch.int, copy=True)
        x = x + 1
        y = y - 2
        z = x.to(y.device)
        return x, y, z


class ToLayout(nn.Module):
    def __init__(self, memory_format, copy, read_layout, sliced, nested, size, dtype=torch.float32):
        super().__init__()
        self.memory_format = memory_format
        self.copy = copy
        self.read_layout = read_layout
        self.sliced = sliced
        self.nested = nested
        self.dtype = dtype
        self.size = size

    def forward(self, x):
        if self.sliced:
            x = x[:, ::2]
        value = x.to(dtype=self.dtype, copy=self.copy, memory_format=self.memory_format)
        if self.nested:
            value = value + 1
        if self.read_layout:
            return torch.as_strided(value, (self.size,), (1,), 0)
        return value


def test_layout():
    frontends = ['torchscript']
    if version.parse(torch.__version__) >= version.parse('2.11'):
        frontends.append('pt2')
    for case, memory_format, copy, sliced, shape in (
        ('copy', None, True, True, (3, 4)),
        ('channels_last', torch.channels_last, False, False, (1, 2, 2, 3)),
        ('channels_last_3d', torch.channels_last_3d, False, False, (1, 2, 2, 2, 3)),
    ):
        x = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        size = x.numel() // (2 if sliced else 1)
        for read_layout in (False, True):
            for nested in (False, True):
                for dtype in (torch.float32, torch.float16):
                    net = ToLayout(memory_format, copy, read_layout, sliced, nested, size, dtype).eval()
                    reference = ToLayout(memory_format, copy, read_layout, sliced, nested, size).eval()(x)
                    expected = net(x)
                    for frontend in frontends:
                        name = 'test_Tensor_to_' + '_'.join((case, str(read_layout), str(nested), str(dtype).split('.')[-1], frontend))
                        if frontend == 'torchscript':
                            torch.jit.trace(net, x).save(name + '.pt')
                            source = torch.jit.load(name + '.pt')(x)
                            path = name + '.pt'
                            args = ['inputshape=[' + ','.join(map(str, shape)) + ']']
                        else:
                            path = name + '.pt2'
                            torch.export.save(torch.export.export(net, (x,)), path)
                            source = torch.export.load(path).module()(x)
                            args = []
                        assert source.shape == expected.shape and source.dtype == expected.dtype
                        assert torch.equal(source, expected)
                        result = subprocess.run(['../src/pnnx', path, 'fp16=0'] + args, capture_output=True, text=True)
                        if frontend == 'pt2' and read_layout:
                            # The pt2 frontend requires a proven storage layout;
                            # it does not yet infer one through Tensor.to.
                            assert result.returncode != 0 and 'layout query source is not statically known' in result.stderr
                            continue
                        assert result.returncode == 0, result.stderr
                        actual = importlib.import_module(name + '_pnnx').Model().eval()(x)
                        assert isinstance(actual, torch.Tensor)
                        assert actual.shape == reference.shape and actual.dtype == reference.dtype
                        assert torch.equal(actual, reference), name
                        param = Path(name + '.pnnx.param').read_text()
                        assert ('Tensor.to ' in param) == read_layout, name
    return True


def test():
    if not test_layout():
        return False
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.randint(10, (1, 13), dtype=torch.int)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_Tensor_to.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_to.pt inputshape=[3,16],[1,13]i32")

    # pnnx inference
    import test_Tensor_to_pnnx
    b = test_Tensor_to_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_Tensor_to_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_Tensor_to_export.pt2 fp16=0")

    # pnnx inference
    import test_Tensor_to_export_pnnx
    c = test_Tensor_to_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype or not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
