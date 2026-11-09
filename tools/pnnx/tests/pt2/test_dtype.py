# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import subprocess

import torch
from packaging import version


class ExplicitDtype(torch.nn.Module):
    def forward(self, x):
        return (torch.linalg.vector_norm(x, dim=1, dtype=torch.float32),
                torch.cumsum(x, 1, dtype=torch.float32),
                torch.cumprod(x, 1, dtype=torch.float32))


class IntegerScan(torch.nn.Module):
    def forward(self, x):
        return x.cumsum(1, dtype=torch.int32), x.cumprod(1, dtype=torch.int32)


class BufferDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.125, -2.25, 3.5], dtype=dtype))

    def forward(self, x):
        return x + self.weight


class InplaceArithmetic(torch.nn.Module):
    def __init__(self, operation, view=False):
        super().__init__()
        self.operation = operation
        self.view = view

    def forward(self, x, y):
        result = x.clone()
        target = result[:, 1:] if self.view else result
        if self.operation == 'add':
            target.add_(y)
        elif self.operation == 'sub':
            target.sub_(y)
        elif self.operation == 'mul':
            target.mul_(y)
        elif self.operation == 'div':
            target.div_(y)
        elif self.operation == 'clamp':
            target.clamp_(min=y)
        elif self.operation == 'floor':
            target.div_(y, rounding_mode='floor')
        else:
            target.div_(y, rounding_mode='trunc')
        # A subsequent operation must see the rounded destination, not the
        # higher precision functional temporary. Also observe the base alias.
        return target, target.to(torch.float32) * 3, result


def check(name, model, inputs):
    prefix = 'test_pt2_dtype_' + name
    model.eval()
    expected = model(*inputs)
    program = torch.export.export(model, inputs)
    torch.export.save(program, prefix + '.pt2')
    generated = prefix + '_pnnx.py'
    if os.path.exists(generated):
        os.remove(generated)
    result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'], capture_output=True, text=True, timeout=90)
    if not os.path.isfile(generated) or result.returncode != 0:
        print(result.stderr)
        return False
    actual = importlib.import_module(prefix + '_pnnx').Model().eval()(*inputs)
    for a, b in zip(actual, expected):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            print(name, a, b)
            return False
    return True


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    for dtype in (torch.float16, torch.bfloat16):
        model = BufferDtype(dtype).eval()
        torch.manual_seed(0)
        x = torch.rand(2, 3, dtype=dtype)
        prefix = 'test_pt2_dtype_buffer_' + str(dtype).split('.')[-1]
        torch.export.save(torch.export.export(model, (x,)), prefix + '.pt2')
        subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'], check=True)
        actual = importlib.import_module(prefix + '_pnnx').test_inference()
        expected = model(x)
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            return False
    if not check('explicit', ExplicitDtype(), (torch.tensor([[1.25, 2.5, 3.75], [4.5, 5.25, 6.75]], dtype=torch.float16),)):
        return False
    if not check('integer_scan', IntegerScan(), (torch.tensor([[2147483647, 1], [65536, 65536]], dtype=torch.int64),)):
        return False
    if not check('float_integer_scan', IntegerScan(), (torch.tensor([[0.6, 0.6], [1.5, 1.5]], dtype=torch.float32),)):
        return False
    if not check('half_add_rounding', InplaceArithmetic('add'),
                 (torch.ones(2, 3, dtype=torch.float16), torch.full((1, 3), 0.0003, dtype=torch.float32))):
        return False
    for operation in ('add', 'sub', 'mul', 'div', 'clamp', 'floor', 'trunc'):
        for view in (False, True):
            destination_dtype = torch.float32 if operation == 'clamp' else torch.float16
            rhs_dtype = torch.float64 if operation == 'clamp' else torch.float32
            x = torch.tensor([[1.125, -2.25, 3.5], [-4.75, 5.125, -6.25]], dtype=destination_dtype)
            y = torch.tensor([[0.333333333333, 1.00000001, 1.777777777777]], dtype=rhs_dtype)
            if view:
                y = y[:, 1:]
            if not check(operation + ('_view' if view else ''), InplaceArithmetic(operation, view), (x, y)):
                return False
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
