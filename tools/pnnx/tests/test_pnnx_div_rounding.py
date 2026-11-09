# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os

import torch
from packaging import version


class Model(torch.nn.Module):
    def forward(self, x, y):
        return (torch.div(x, y),
                torch.div(x, y, rounding_mode='floor'),
                torch.div(x, y, rounding_mode='trunc'),
                torch.div(x, 1, rounding_mode='floor'),
                torch.div(x, 1, rounding_mode='trunc'),
                torch.div(x, 1),
                torch.div(3, y, rounding_mode='trunc'),
                torch.floor_divide(x, y))


class Shape(torch.nn.Module):
    def forward(self, x):
        half = torch.div(x.size(2), 2, rounding_mode='trunc')
        half_floor = torch.div(x.size(2), 2, rounding_mode='floor')
        return (x[:, :, :half], x.reshape(x.size(0), -1),
                x[:, :, :half_floor], x.reshape(x.size(0), x.size(1) * 2, half_floor))


def test():
    for name, dtype in [('float', torch.float32), ('half', torch.float16),
                        ('bf16', torch.bfloat16), ('int', torch.int32), ('long', torch.int64)]:
        model = Model().eval()
        x = torch.tensor([[-9, -7, -3, -1], [1, 3, 7, 9]], dtype=dtype)
        y = torch.tensor([[2, -2, 2, -2], [2, -2, 2, -2]], dtype=dtype)
        if x.is_floating_point():
            x = x + 0.5
        expected = model(x, y)
        prefix = 'test_pnnx_div_rounding_' + name

        mod = torch.jit.trace(model, (x, y))
        mod.save(prefix + '.pt')
        types = {'float': 'f32', 'half': 'f16', 'bf16': 'bf16', 'int': 'i32', 'long': 'i64'}
        shape = '[2,4]' + types[name]
        os.system('../src/pnnx ' + prefix + '.pt inputshape=' + shape + ',' + shape + ' fp16=0')
        converted = importlib.import_module(prefix + '_pnnx').Model().eval()
        actual = converted(x, y)
        for a, b in zip(actual, expected):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False

        if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
            continue

        program = torch.export.export(model, (x, y))
        torch.export.save(program, prefix + '_export.pt2')
        os.system('../src/pnnx ' + prefix + '_export.pt2 fp16=0')
        converted = importlib.import_module(prefix + '_export_pnnx').Model().eval()
        actual = converted(x, y)
        for a, b in zip(actual, expected):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False

    model = Shape().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 2, 6)
    x2 = torch.rand(1, 3, 8)
    mod = torch.jit.trace(model, x)
    mod.save('test_pnnx_div_rounding_shape.pt')
    os.system('../src/pnnx test_pnnx_div_rounding_shape.pt inputshape=[1,2,6] inputshape2=[1,3,8] fp16=0')
    import test_pnnx_div_rounding_shape_pnnx
    converted = test_pnnx_div_rounding_shape_pnnx.Model().eval()
    for value in (x, x2):
        for a, b in zip(converted(value), model(value)):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
