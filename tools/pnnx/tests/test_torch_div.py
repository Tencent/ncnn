# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x * 8 - 4
        y = y * 2 + 1
        a = torch.div(x, y)
        b = torch.div(x, y, rounding_mode='floor')
        c = torch.div(x, y, rounding_mode='trunc')
        d = torch.div(x, 2.5, rounding_mode='floor')
        e = torch.div(x, 2.5, rounding_mode='trunc')
        f = x.div(-y, rounding_mode='floor')
        g = x.div(-y, rounding_mode='trunc')
        h = torch.sin(torch.div(x, y, rounding_mode='floor')) * 2
        i = torch.sin(torch.div(x, y, rounding_mode='trunc')) * 2
        return a, b, c, d, e, f, g, h, i

class IntegerModel(nn.Module):
    def __init__(self, bitwise):
        super(IntegerModel, self).__init__()
        self.bitwise = bitwise

    def forward(self, x, y):
        a = torch.div(x, y, rounding_mode='trunc')
        b = torch.div(x, 2, rounding_mode='trunc')
        c = torch.div(a, 2, rounding_mode='trunc')
        d = torch.div(x, y, rounding_mode='floor')
        e = torch.trunc(x / y)
        if self.bitwise:
            return a, b, c, d, e, a & 1, c | 2
        return a, b, c, d, e

def test_integer():
    import importlib.util
    import os

    for i, (xtype, ytype) in enumerate([(torch.int32, torch.int32), (torch.int64, torch.int64), (torch.int32, torch.int64), (torch.int32, torch.float32), (torch.float32, torch.int64)]):
        x = torch.tensor([[-7, -5, -1], [1, 5, 7]], dtype=xtype)
        y = torch.tensor([[2, -2, 2], [-2, 2, -2]], dtype=ytype)
        net = IntegerModel(xtype != torch.float32 and ytype != torch.float32).eval()
        expected = net(x, y)
        name = 'test_torch_div_integer_%d' % i
        mod = torch.jit.trace(net, (x, y))
        mod.save(name + '.pt')
        dtype = {torch.int32: 'i32', torch.int64: 'i64', torch.float32: 'f32'}
        os.system('../src/pnnx %s.pt inputshape=[2,3]%s,[2,3]%s fp16=0' % (name, dtype[xtype], dtype[ytype]))
        spec = importlib.util.spec_from_file_location(name, name + '_pnnx.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = module.Model()(x, y)
        for a, b in zip(expected, actual):
            if a.dtype != b.dtype or not torch.equal(a, b):
                return False
    return True

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save('test_torch_div.pt')

    # torchscript to pnnx
    import os
    os.system('../src/pnnx test_torch_div.pt inputshape=[3,16],[3,16] fp16=0')

    # pnnx inference
    import test_torch_div_pnnx
    b = test_torch_div_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-6, 1e-6):
            return False
    return test_integer()

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
