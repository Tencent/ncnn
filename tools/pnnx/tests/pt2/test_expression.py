# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
import subprocess

import torch
from packaging import version


class Arithmetic(torch.nn.Module):
    def __init__(self, dtype=torch.float32, shape=(), alpha=None):
        super().__init__()
        self.register_buffer('weight', torch.tensor(2, dtype=dtype).reshape(shape))
        self.alpha = alpha

    def forward(self, x):
        if self.alpha is not None:
            return torch.add(x, self.weight, alpha=self.alpha)
        return x + self.weight


class ConstantReturns(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor(2.))
        self.register_buffer('b', torch.tensor(3.))

    def forward(self, x):
        return x + self.a, self.a, self.a + self.b, self.a.clone()


class Lifted(torch.nn.Module):
    def forward(self, x):
        return x * torch.tensor(2.)


class IntegerAttributes(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        values = (2**40 + 3, -2**40 - 3, 2**63 - 1, -2**63, 2**31, -2**31 - 1,
                  2**31 - 1, -2**31)
        self.count = len(values)
        for i, value in enumerate(values):
            self.register_buffer('v' + str(i), torch.tensor(value, dtype=torch.int64).reshape(shape))

    def forward(self, x):
        return tuple(x + getattr(self, 'v' + str(i)) for i in range(self.count))


class IntegerUnbind(torch.nn.Module):
    def __init__(self, return_constants=False):
        super().__init__()
        self.register_buffer('values', torch.tensor([2**40 + 3, -2**40 - 3, 2**63 - 1, -2**63, 5]))
        self.return_constants = return_constants

    def forward(self, x):
        values = self.values.unbind()
        results = tuple(x + value for value in values)
        if self.return_constants:
            return results + values
        return results


class Foldable(torch.nn.Module):
    def __init__(self, value, dtype=torch.int64):
        super().__init__()
        self.register_buffer('values', torch.tensor([value], dtype=dtype))

    def forward(self, x):
        return x + self.values.sum()


def check(name, model, inputs, *, torchscript=False, inline=False):
    prefix = 'test_pt2_expression_' + name
    model.eval()
    if torchscript:
        path = prefix + '.pt'
        torch.jit.trace(model, inputs).save(path)
    else:
        path = prefix + '.pt2'
        torch.export.save(torch.export.export(model, inputs), path)

    generated = Path(prefix + '_pnnx.py')
    generated.unlink(missing_ok=True)
    command = ['../../src/pnnx', path, 'fp16=0']
    if torchscript:
        dtypes = {torch.float32: 'f32', torch.float16: 'f16', torch.int32: 'i32', torch.int64: 'i64'}
        command.append('inputshape=' + ','.join(
            '[' + ','.join(map(str, x.shape)) + ']' + dtypes[x.dtype] for x in inputs))
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    # A backend dtype/rank rejection is allowed only after PNNX artifacts were
    # saved. Safe FP32 scalar fusion must also make it through ncnn conversion.
    if (not generated.is_file() or (result.returncode != 0 and
            (torchscript or inline or 'pt2 ncnn: unsupported' not in result.stderr))):
        print(name, result.stderr)
        return False

    ir = Path(prefix + '.pnnx.param').read_text()
    if inline and 'pnnx.Attribute' in ir:
        print(name, 'constant was not inlined', ir)
        return False
    spec = importlib.util.spec_from_file_location(prefix, generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().eval()
    for samples in (inputs, tuple(x + 1 for x in inputs)):
        actual, expected = converted(*samples), model(*samples)
        if isinstance(expected, torch.Tensor):
            actual, expected = (actual,), (expected,)
        if len(actual) != len(expected):
            print(name, 'output count differs')
            return False
        for a, b in zip(actual, expected):
            if not isinstance(a, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                print(name, 'output type, dtype, shape or value differs', a, b)
                return False
    return True


@torch.no_grad()
def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    f32 = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)
    i64 = (torch.zeros(2, 3, dtype=torch.int64),)
    i32 = (torch.ones(2, 3, dtype=torch.int32),)
    for name, model, inputs in (
        ('zero_dim', Arithmetic(), f32),
        ('alpha', Arithmetic(shape=(1,), alpha=0.5), f32),
    ):
        for ts in (False, True):
            if not check(name + ('_ts' if ts else ''), model, inputs, torchscript=ts, inline=True):
                return False

    for name, model, inputs in (
        ('half_promotion', Arithmetic(torch.float16), i32),
        ('float_promotion', Arithmetic(), i32),
        ('broadcast', Arithmetic(shape=(1,)), (torch.tensor(1.),)),
        ('zero_dim_input', Arithmetic(), (torch.tensor(1.),)),
        ('constant_returns', ConstantReturns(), f32),
        ('integer_attributes', IntegerAttributes((1,)), i64),
        ('integer_zero_dim', IntegerAttributes(()), i64),
        ('integer_unbind', IntegerUnbind(), i64),
        ('unbind_returns', IntegerUnbind(True), i64),
    ):
        if not check(name, model, inputs):
            return False

    if not check('lifted_clone', Lifted(), f32):
        return False

    # TorchScript shape inference stores constant subgraph results in the
    # foldable-constant zip, independently of Attribute and unbind scalarization.
    for i, value in enumerate((2**40 + 3, -2**40 - 3, 2**63 - 1, -2**63)):
        if not check('foldable_' + str(i), Foldable(value), i64, torchscript=True):
            return False
    if not check('foldable_promotion', Foldable(2., torch.float32), i32, torchscript=True):
        return False
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
