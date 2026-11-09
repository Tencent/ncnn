# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from pathlib import Path
import subprocess

import torch
from packaging import version


class Copy(torch.nn.Module):
    def forward(self, target, source):
        a = torch.ops.aten.copy.default(target, source)
        b = target.clone()
        b.copy_(source)
        c = target.clone()
        c[:, 1:].copy_(source)
        # Functional copy on a slice returns the slice shape, not the base.
        d = torch.ops.aten.copy.default(target[:, 1:], source)
        return a, b, c, d, target + 1, source + 1


class Broadcast(torch.nn.Module):
    def forward(self, x, y):
        a = x.clone()
        a.copy_(y)
        return a, torch.ops.aten.copy.default(x, y)


class CopyResult(torch.nn.Module):
    def forward(self, x, y):
        base = x.clone()
        result = base[:, 1:].copy_(y)
        return base, result, result + 3


class DynamicCopy(torch.nn.Module):
    def forward(self, x, y):
        a = x.clone()
        a.copy_(y)
        return a


class OverlappingMutation(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        a = base[:, 1:3]
        result_a = a.mul_(2)
        b = base[:, 2:4]
        result_b = b.add_(3)
        return base, a, b, result_a, result_b


class DerivedMutationResult(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        result = base[:, 1:3].mul_(2)
        alias = result[:]
        base[:, 2:].add_(3)
        return base, alias


class ChainedMutationResult(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        result = base.relu_()
        result.add_(2)
        return base, result


class Fill(torch.nn.Module):
    def forward(self, x):
        a = torch.ops.aten.fill.Scalar(x, 3.5)
        b = x.clone()
        b[:, 1:].fill_(2.5)
        return a, x + 1, b


class TensorFill(torch.nn.Module):
    def forward(self, x, value):
        a = torch.ops.aten.fill.Tensor(x, value)
        b = x.clone()
        result = b[:, 1:].fill_(value)
        return a, x + 1, b, result, value


class ConstantTensorFill(torch.nn.Module):
    def forward(self, x):
        value = torch.tensor(2.5)
        a = torch.ops.aten.fill.Tensor(x, value)
        b = x.clone()
        b[:, 1:].fill_(value)
        return a, x + 1, b, value


class Scalar(torch.nn.Module):
    def forward(self, x):
        return x.reshape(())


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    for name, model, inputs in [
        ('derived_result', DerivedMutationResult(), (torch.randn(2, 4),)),
        ('chained_result', ChainedMutationResult(), (torch.randn(2, 4),)),
        ('overlapping', OverlappingMutation(), (torch.randn(2, 4),)),
        ('copy_result', CopyResult(), (torch.randn(2, 3), torch.randn(1, 2))),
        ('copy_original', Broadcast(), (torch.randn(2, 3), torch.randn(1, 3))),
        ('copy_broadcast', Copy(), (torch.randn(2, 3), torch.randn(2, 1))),
        ('copy_float', Copy(), (torch.randn(2, 3), torch.tensor([2], dtype=torch.int32))),
        ('copy_int', Copy(), (torch.zeros(2, 3, dtype=torch.int32), torch.tensor([2.75]))),
        ('fill', Fill(), (torch.randn(2, 3),)),
        ('fill_tensor', TensorFill(), (torch.randn(2, 3), torch.tensor(2.5))),
        ('fill_tensor_int', TensorFill(), (torch.zeros(2, 3, dtype=torch.int32), torch.tensor(2.75))),
        ('fill_tensor_int64', TensorFill(), (torch.zeros(2, 3, dtype=torch.int64), torch.tensor(2**40 + 1))),
        ('fill_tensor_half', TensorFill(), (torch.randn(2, 3, dtype=torch.float16), torch.tensor(2.501))),
        ('fill_tensor_complex', TensorFill(), (torch.zeros(2, 3, dtype=torch.complex64), torch.tensor(2.5 + 1.25j))),
        ('fill_tensor_constant', ConstantTensorFill(), (torch.randn(2, 3),)),
        ('scalar', Scalar(), (torch.tensor([2.]),)),
    ]:
        prefix = 'test_pt2_' + name
        torch.export.save(torch.export.export(model.eval(), inputs), prefix + '.pt2')
        Path(prefix + '_pnnx.py').unlink(missing_ok=True)
        result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'], capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            print(result.stderr)
            return False
        expected = model(*[x.clone() for x in inputs])
        actual_inputs = [x.clone() for x in inputs]
        actual = importlib.import_module(prefix + '_pnnx').Model().eval()(*actual_inputs)
        if isinstance(expected, torch.Tensor):
            expected, actual = (expected,), (actual,)
        if len(actual) != len(expected):
            return False
        for a, b in zip(actual, expected):
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                print(name, a, b)
                return False
        if any(not torch.equal(a, b) for a, b in zip(actual_inputs, inputs)):
            print(name, 'modified a caller input')
            return False
    for name, model, broadcast in [('slice', CopyResult(), False), ('root', DynamicCopy(), True)]:
        prefix = 'test_pt2_copy_dynamic_' + name
        batch = torch.export.Dim('batch', min=2, max=8)
        inputs = (torch.randn(2, 3), torch.randn(1, 3) if broadcast else torch.randn(2, 2))
        shapes = ({0: batch}, {} if broadcast else {0: batch})
        program = torch.export.export(model, inputs, dynamic_shapes=shapes)
        torch.export.save(program, prefix + '.pt2')
        args = ['inputshape=[2,3],[1,3]', 'inputshape2=[3,3],[1,3]'] if broadcast else [
            'inputshape=[2,3],[2,2]', 'inputshape2=[3,3],[3,2]']
        result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0', 'optlevel=2', *args], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            return False
        param = Path(prefix + '.pnnx.param').read_text()
        if param.count('Tensor.expand_as ') != int(broadcast):
            return False
        converted = importlib.import_module(prefix + '_pnnx').Model().eval()
        for n in (2, 3, 5):
            xs = (torch.randn(n, 3), torch.randn(1, 3) if broadcast else torch.randn(n, 2))
            expected = model(*[x.clone() for x in xs])
            actual = converted(*[x.clone() for x in xs])
            if isinstance(expected, torch.Tensor):
                expected, actual = (expected,), (actual,)
            if any(a.shape != b.shape or not torch.equal(a, b) for a, b in zip(expected, actual)):
                return False
    return True


if __name__ == '__main__':
    assert test()
