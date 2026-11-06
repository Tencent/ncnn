# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os
import subprocess

import torch


class Multi(torch.nn.Module):
    def forward(self, x, y):
        maximum, indices = x.max(1)
        parts = torch.split(x, [1, 2], dim=1)
        unbound = torch.unbind(y, 0)
        return maximum, indices, torch.cat(parts, 1), torch.stack(unbound, 0), x.mean(1), x.sum(1)


class Index(torch.nn.Module):
    def forward(self, x, idx):
        return x[:, idx], x[:, 1:3], x.select(1, 2)


class Shapes(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1).reshape(2, -1), x.transpose(1, 2), x.flatten(1), x.view(2, -1)


class ViewDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.view(self.dtype)


class IndexSlices(torch.nn.Module):
    def forward(self, x, idx):
        return x[:, idx], x[..., idx], x[idx, :, idx], x[:, idx, :, idx], x[:, [2, 0]], x[[2, 0], :, [1, 0]], torch.ops.aten.index.Tensor(x, [None, idx, None]), x[:, torch.maximum(idx, idx + 1)]


class NativeBatch(torch.nn.Module):
    def forward(self, x, mean, var, weight, bias):
        return torch.ops.aten.native_batch_norm.default(x, weight, bias, mean, var, False, 0.1, 1e-5)[0]


def test():
    torch.set_num_threads(1)
    for name, model, shapes in [
        ('multi', Multi(), ((2, 3), (2, 4))),
        ('shapes', Shapes(), ((2, 3, 4),)),
    ]:
        model.eval()
        torch.manual_seed(0)
        inputs = tuple(torch.rand(*shape) for shape in shapes)
        expected = model(*inputs)
        prefix = 'test_pt2_ops_' + name
        program = torch.export.export(model, inputs)
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

        actual = importlib.import_module(prefix + '_pnnx').test_inference()
        if len(actual) != len(expected):
            return False
        for a, b in zip(actual, expected):
            if a.shape != b.shape:
                return False
            if b.dtype == torch.int64:
                if a.dtype != b.dtype or not torch.equal(a, b):
                    return False
            elif not torch.allclose(a, b, 2e-5, 2e-5):
                return False

    for name, model, shape in [
        ('index', Index(), (2, 4)),
        ('index_slices', IndexSlices(), (3, 4, 5, 6)),
    ]:
        model.eval()
        torch.manual_seed(0)
        x = torch.rand(*shape)
        idx = torch.tensor([2, 0], dtype=torch.int64)
        expected = model(x, idx)
        prefix = 'test_pt2_ops_' + name
        program = torch.export.export(model, (x, idx))
        torch.export.save(program, prefix + '.pt2')
        os.system('../../src/pnnx ' + prefix + '.pt2 fp16=0')

        actual = importlib.import_module(prefix + '_pnnx').Model().eval()(x, idx)
        if len(actual) != len(expected):
            return False
        for a, b in zip(actual, expected):
            if a.shape != b.shape or not torch.equal(a, b):
                return False

    for name, dtype, x in [
        ('view_dtype_int32', torch.int32, torch.tensor([1., 2., 3.], dtype=torch.float32)),
        ('view_dtype_int16', torch.int16, torch.tensor([1.], dtype=torch.float32)),
    ]:
        program = torch.export.export(ViewDtype(dtype).eval(), (x,))
        path = 'test_pt2_ops_' + name + '.pt2'
        torch.export.save(program, path)
        result = subprocess.run(['../../src/pnnx', path, 'fp16=0'], capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or 'torch.ops.aten.view.dtype: dtype reinterpretation is unsupported' not in result.stderr:
            print(result.stderr)
            return False

    model = NativeBatch().eval()
    torch.manual_seed(0)
    inputs = (torch.rand(2, 3, 4), torch.rand(3), torch.rand(3) + 1, torch.rand(3), torch.rand(3))
    expected = model(*inputs)
    program = torch.export.export(model, inputs)
    torch.export.save(program, 'test_pt2_ops_native_batch.pt2')
    os.system('../../src/pnnx test_pt2_ops_native_batch.pt2 fp16=0')

    import test_pt2_ops_native_batch_pnnx
    actual = test_pt2_ops_native_batch_pnnx.Model().eval()(*inputs)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
        return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
