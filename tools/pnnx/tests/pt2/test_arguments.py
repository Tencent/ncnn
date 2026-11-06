# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

import torch
from common import rewrite_archive


class Arguments(torch.nn.Module):
    def forward(self, x):
        return torch.add(x, 2, alpha=3), torch.div(x, 2, rounding_mode='floor'), torch.div(x, 2, rounding_mode='trunc'), x.reshape(())


class Lists(torch.nn.Module):
    def forward(self, x):
        return torch.split(x, 100, dim=1)[0], torch.unbind(x, dim=0)[0]


class Nonfinite(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-float('inf'), max=float('inf')), x + float('nan')


class IntegerReduce(torch.nn.Module):
    def forward(self, x):
        return x.sum(1, dtype=torch.int32), x.prod(1, dtype=torch.int32), x.sum(1, dtype=torch.bool)


class ScalarThreshold(torch.nn.Module):
    def forward(self, x):
        torch._check(x.item() > 1.00000001)
        return x + 1


class SharedScalar(torch.nn.Module):
    def forward(self, x):
        value = x.item()
        torch._check(value > 1.00000001)
        return x + value


class NumericTensorArguments(torch.nn.Module):
    def forward(self, x):
        a = torch.ops.aten.add.Tensor(1, 2)
        b = torch.ops.aten.add.Tensor(True, True)
        c = torch.ops.aten.div.Tensor(3, 2)
        return x + b, a, b, c


def bad_arg(p, entries):
    p['graph_module']['graph']['nodes'][0]['inputs'].append({'name': 'nonsense', 'arg': {'as_int': 1}, 'kind': 2})


def test():
    torch.set_num_threads(1)
    model = Arguments().eval()
    x = torch.tensor(-3.5)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_args.pt2')
    os.system('../../src/pnnx test_pt2_arguments_args.pt2 fp16=0')

    import test_pt2_arguments_args_pnnx
    torch.manual_seed(0)
    sample = torch.rand(())
    for actual, expected in [
        (test_pt2_arguments_args_pnnx.Model().eval()(x), model(x)),
        (test_pt2_arguments_args_pnnx.test_inference(), model(sample)),
    ]:
        if len(actual) != len(expected):
            return False
        for a, b in zip(actual, expected):
            if a.shape != b.shape or not torch.allclose(a, b, 2e-5, 2e-5):
                return False

    model = Lists().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_one_list.pt2')
    os.system('../../src/pnnx test_pt2_arguments_one_list.pt2 fp16=0')

    import test_pt2_arguments_one_list_pnnx
    actual = test_pt2_arguments_one_list_pnnx.test_inference()
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        # Compare the tensor leaf of each single-element split/unbind result.
        if isinstance(a, tuple):
            a, = a
        if a.shape != b.shape or not torch.allclose(a, b, 2e-5, 2e-5):
            return False

    model = Nonfinite().eval()
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_nonfinite.pt2')
    os.system('../../src/pnnx test_pt2_arguments_nonfinite.pt2 fp16=0')

    import test_pt2_arguments_nonfinite_pnnx
    actual = test_pt2_arguments_nonfinite_pnnx.test_inference()
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if a.shape != b.shape:
            return False
    if not torch.allclose(actual[0], expected[0], 2e-5, 2e-5):
        return False
    if not torch.allclose(actual[1], expected[1], 2e-5, 2e-5, equal_nan=True):
        return False

    for name, change, message in [
        ('unknown_arg', bad_arg, 'argument'),
        ('unknown_op', lambda p, e: p['graph_module']['graph']['nodes'][0].update(target='torch.ops.aten.no_such_op.default'), 'schema'),
        ('overflow', lambda p, e: p['graph_module']['graph']['nodes'][0]['inputs'][1].update(arg={'as_int': 2147483648}), 'integer outside'),
    ]:
        path = 'test_pt2_arguments_' + name + '.pt2'
        rewrite_archive('test_pt2_arguments_args.pt2', path, change)
        result = subprocess.run(['../../src/pnnx', path, 'fp16=0'], capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or message not in result.stderr:
            print(result.stderr)
            return False

    model = IntegerReduce().eval()
    x = torch.tensor([[2147483647, 1], [65536, 65536]], dtype=torch.int64)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_integer_reduce.pt2')
    os.system('../../src/pnnx test_pt2_arguments_integer_reduce.pt2 fp16=0')

    import test_pt2_arguments_integer_reduce_pnnx
    actual = test_pt2_arguments_integer_reduce_pnnx.Model().eval()(x)
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            return False

    # Runtime checks and their unused scalar conditions are discarded on import.
    program = torch.export.export(ScalarThreshold().eval(), (torch.tensor(2., dtype=torch.float32),))
    torch.export.save(program, 'test_pt2_arguments_scalar_threshold.pt2')
    os.system('../../src/pnnx test_pt2_arguments_scalar_threshold.pt2 fp16=0')

    import test_pt2_arguments_scalar_threshold_pnnx
    converted = test_pt2_arguments_scalar_threshold_pnnx.Model().eval()
    for x in (torch.tensor(2.), torch.tensor(0.)):
        if not torch.equal(converted(x), x + 1):
            return False

    # Keep scalar values that are also used by inference.
    model = SharedScalar().eval()
    x = torch.tensor(2.)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_shared_scalar.pt2')
    os.system('../../src/pnnx test_pt2_arguments_shared_scalar.pt2 fp16=0')

    import test_pt2_arguments_shared_scalar_pnnx
    actual = test_pt2_arguments_shared_scalar_pnnx.Model().eval()(x)
    if actual.shape != x.shape or actual.dtype != x.dtype or not torch.equal(actual, model(x)):
        return False

    model = NumericTensorArguments().eval()
    torch.manual_seed(0)
    x = torch.rand(2)
    expected = model(x)
    program = torch.export.export(model, (x,))
    torch.export.save(program, 'test_pt2_arguments_numeric_tensor.pt2')
    os.system('../../src/pnnx test_pt2_arguments_numeric_tensor.pt2 fp16=0')

    import test_pt2_arguments_numeric_tensor_pnnx
    actual = test_pt2_arguments_numeric_tensor_pnnx.test_inference()
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if not isinstance(a, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
