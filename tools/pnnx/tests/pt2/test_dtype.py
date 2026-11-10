# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import subprocess

import torch
from packaging import version
from common import rewrite_archive


class ExplicitDtype(torch.nn.Module):
    def forward(self, x):
        return (torch.linalg.vector_norm(x, dim=1, dtype=torch.float32),
                torch.cumsum(x, 1, dtype=torch.float32),
                torch.cumprod(x, 1, dtype=torch.float32))


class IntegerScan(torch.nn.Module):
    def forward(self, x):
        return x.cumsum(1, dtype=torch.int32), x.cumprod(1, dtype=torch.int32)


class NeutralTensors(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        zero = torch.zeros((2, 3), dtype=self.dtype)
        one = torch.ones((2, 3), dtype=self.dtype)
        return x + zero, zero + x, x - zero, x * one, one * x, x / one, x ** one, torch.add(x, one, alpha=0)


class RMSNormScalar(torch.nn.Module):
    def forward(self, x, e):
        value = e.item()
        return torch.nn.functional.rms_norm(x, (3,), eps=0.1), x + value


class NumericCopy(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        result = x.clone()
        copied = torch.ops.aten.copy_.default(result, self.value)
        return result, copied


def runtime_epsilon(program, entries):
    nodes = program['graph_module']['graph']['nodes']
    scalar = next(n for n in nodes if n['target'] == 'torch.ops.aten.item.default')['outputs'][0]
    norm = next(n for n in nodes if n['target'] == 'torch.ops.aten.rms_norm.default')
    next(a for a in norm['inputs'] if a['name'] == 'eps')['arg'] = scalar


class BufferDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.125, -2.25, 3.5], dtype=dtype))

    def forward(self, x):
        return x + self.weight


class ComplexDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.register_buffer('weight', torch.tensor([1.125 + 2.25j, -3.5 + 4.75j, 5.125 - 6.25j], dtype=dtype))

    def forward(self, x, z, w):
        value = x + z
        return (w + self.weight, value, value.to(self.dtype), value.sum(1, dtype=self.dtype),
                z.to(self.dtype), z.sum(1, dtype=self.dtype),
                torch.complex(x, x), w.real, w.imag, torch.view_as_real(w))


class ComplexMutation(torch.nn.Module):
    def __init__(self, source, operation):
        super().__init__()
        self.source = source
        self.operation = operation

    def forward(self, x, z, w):
        result = torch.complex(x, x) if self.source == 'construct' else x + z
        target = result[:, 1:] if self.source == 'view' else result
        before = target.clone()
        if self.operation == 'add':
            target.add_(w)
        else:
            target.copy_(w)
        return before, target, result


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
        # A subsequent operation and the base alias must see the same value.
        return target, target.to(torch.float32) * 3, result


def check(name, model, inputs, reference=None):
    prefix = 'test_pt2_dtype_' + name
    model.eval()
    source = model(*inputs)
    if isinstance(source, torch.Tensor):
        source = (source,)
    program = torch.export.export(model, inputs)
    program.validate()
    torch.export.save(program, prefix + '.pt2')
    for exported in (program, torch.export.load(prefix + '.pt2')):
        exported.validate()
        outputs = exported.module()(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if len(outputs) != len(source) or any(a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b) for a, b in zip(outputs, source)):
            print(name, 'exported source differs', outputs, source)
            return False
    generated = prefix + '_pnnx.py'
    if os.path.exists(generated):
        os.remove(generated)
    result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'], capture_output=True, text=True, timeout=90)
    if not os.path.isfile(generated) or result.returncode != 0:
        print(result.stderr)
        return False
    inputs = tuple(x.float() if x.is_floating_point() else x.to(torch.complex64) if x.is_complex() else x for x in inputs)
    expected = (reference if reference is not None else model).float().eval()(*inputs)
    actual = importlib.import_module(prefix + '_pnnx').Model().float().eval()(*inputs)
    if isinstance(expected, torch.Tensor):
        actual, expected = (actual,), (expected,)
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            print(name, a, b)
            return False
    return True


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        model = BufferDtype(dtype).eval()
        torch.manual_seed(0)
        x = torch.rand(2, 3, dtype=dtype)
        prefix = 'test_pt2_dtype_buffer_' + str(dtype).split('.')[-1]
        torch.export.save(torch.export.export(model, (x,)), prefix + '.pt2')
        subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'], check=True)
        module = importlib.import_module(prefix + '_pnnx')
        actual = module.Model().float().eval()(x.float())
        expected = model.float()(x.float())
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            return False
        torch.manual_seed(0)
        expected = model(torch.rand(2, 3))
        if not torch.equal(module.test_inference(), expected):
            return False
    if not check('explicit', ExplicitDtype(), (torch.tensor([[1.25, 2.5, 3.75], [4.5, 5.25, 6.75]], dtype=torch.float16),)):
        return False
    for dtype in (torch.float16, torch.float32):
        if not check('neutral_' + str(dtype).split('.')[-1], NeutralTensors(dtype),
                     (torch.tensor([[1.25, -2.5, 3.75], [4.5, 5.25, -6.75]], dtype=torch.float16),)):
            return False
    for name, value, dtype in (
        ('integer', -2.75, torch.int32),
        ('half', 1.125, torch.float16),
        ('bool', -2, torch.bool),
        ('complex_real', 1 + 2j, torch.float32),
        ('integer_overflow', 1e20, torch.int32),
    ):
        if not check('copy_' + name, NumericCopy(value), (torch.ones(2, 3, dtype=dtype),)):
            return False
    if not check('rms_scalar', RMSNormScalar(), (torch.ones(2, 3), torch.tensor(0.1))):
        return False
    path = 'test_pt2_dtype_rms_runtime_epsilon.pt2'
    rewrite_archive('test_pt2_dtype_rms_scalar.pt2', path, runtime_epsilon)
    result = subprocess.run(['../../src/pnnx', path, 'fp16=0'], capture_output=True, text=True, timeout=90)
    if result.returncode <= 0 or 'RMSNorm epsilon must be constant' not in result.stderr:
        print(result.stderr)
        return False
    if not check('integer_scan', IntegerScan(), (torch.tensor([[2147483647, 1], [65536, 65536]], dtype=torch.int64),)):
        return False
    if not check('float_integer_scan', IntegerScan(), (torch.tensor([[0.6, 0.6], [1.5, 1.5]], dtype=torch.float32),)):
        return False
    if not check('half_add_float32', InplaceArithmetic('add'),
                 (torch.ones(2, 3, dtype=torch.float16), torch.full((1, 3), 0.0003, dtype=torch.float32))):
        return False
    x = torch.tensor([[16777216., 1., -16777216.], [1.125, -2.25, 3.5]], dtype=torch.float64)
    z = torch.tensor([[1j, 2j, 3j], [4j, 5j, 6j]], dtype=torch.complex64)
    w = torch.tensor([[1 + 2**-30 + 2j, 2 + 3j, 4 + 5j]], dtype=torch.complex128)
    if not check('complex', ComplexDtype(torch.complex128), (x, z, w), ComplexDtype(torch.complex64)):
        return False
    for source in ('mixed', 'construct', 'view'):
        for operation in ('add', 'copy'):
            rhs = w[:, 1:] if source == 'view' else w
            if not check('complex_' + source + '_' + operation, ComplexMutation(source, operation), (x, z, rhs)):
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
