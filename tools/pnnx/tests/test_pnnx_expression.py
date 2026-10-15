# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w0 = nn.Parameter(torch.rand(12, 15))
        self.w1 = nn.Parameter(torch.rand(12, 15))
        self.w2 = nn.Parameter(torch.rand(12, 15))
        self.w3 = nn.Parameter(torch.rand(12, 15))
        self.w4 = nn.Parameter(torch.rand(12, 15))
        self.w5 = nn.Parameter(torch.rand(12, 15))
        self.c0 = nn.Parameter(torch.ones(1))
        self.c1 = nn.Parameter(torch.ones(3) + 0.2)

    def forward(self, x):
        c10, c11, _ = torch.unbind(self.c1)
        x0 = x * 10 + self.c0 - c11
        x = x + self.w0 + x0
        x = x - self.w1 + x0.float()
        x = x * self.w2 + x0
        x = x / self.w3 + x0
        x = x // self.w4 + x0
        if version.parse(torch.__version__) >= version.parse('2.0'):
            x = x % self.w5 + x0
        else:
            x = torch.fmod(x, self.w5) + x0
        y = x.int()
        return x, y & 3, y | 3, y ^ 3, y << 3, y >> 3

class ScalarDivision(nn.Module):
    def __init__(self, dtype):
        super(ScalarDivision, self).__init__()
        self.register_buffer('vector', torch.tensor([3.0], dtype=dtype))

    def forward(self, x):
        return torch.div(3.0, x), self.vector / x, x / 3.0, x / self.vector, x / x, torch.div(3.0, x * x + 1)


def test_scalar_division():
    import importlib.util
    import subprocess

    for dtype, suffix in ((torch.float16, 'f16'), (torch.float32, 'f32'), (torch.float64, 'f64')):
        net = ScalarDivision(dtype).eval()
        x = torch.tensor([[5, -5, 3, -3], [0.25, -0.25, 1, -1]], dtype=dtype)
        name = 'test_pnnx_expression_div_' + suffix
        torch.jit.trace(net, x).save(name + '.pt')
        subprocess.run(['../src/pnnx', name + '.pt', 'inputshape=[2,4]' + suffix], check=True)
        spec = importlib.util.spec_from_file_location(name, name + '_pnnx.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = module.Model().eval()(x)
        expected = (torch.div(3.0, x), torch.div(3.0, x), torch.div(x, 3.0),
                    torch.div(x, 3.0), torch.div(x, x), torch.div(3.0, x * x + 1))
        if len(expected) != len(actual):
            return False
        for a, b in zip(expected, actual):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False
    return True


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 15)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_expression.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_expression.pt inputshape=[12,15]")

    # pnnx inference
    import test_pnnx_expression_pnnx
    b = test_pnnx_expression_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return test_scalar_division()

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
