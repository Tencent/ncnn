# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

import torch
from packaging import version


class Dynamic(torch.nn.Module):
    def forward(self, x, y):
        z = x + y
        return z.reshape(x.shape[0], x.shape[1] * x.shape[2]), z.sum(1)


class Floor(torch.nn.Module):
    def forward(self, x):
        return x[:, :x.shape[0] // 2]


class Multiply(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])


class TensorArguments(torch.nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x + n, n - x, x * n, x / n, torch.ops.aten.add.Tensor(n, 2)


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.manual_seed(0)
    net = Dynamic().eval()
    x = torch.rand(3, 3, 4)
    y = torch.rand(3, 3, 4)
    dim = torch.export.Dim('batch', min=2, max=8)
    ep = torch.export.export(net, (x, y), dynamic_shapes=({0: dim}, {0: dim}))
    torch.export.save(ep, 'test_pt2_dynamic.pt2')
    os.system('../../src/pnnx test_pt2_dynamic.pt2 fp16=0 inputshape=[3,3,4],[3,3,4] inputshape2=[5,3,4],[5,3,4]')

    import test_pt2_dynamic_pnnx
    converted = test_pt2_dynamic_pnnx.Model().eval()
    for inputs in ((x, y), (torch.rand(5, 3, 4), torch.rand(5, 3, 4))):
        a = net(*inputs)
        b = converted(*inputs)
        if len(a) != len(b):
            return False
        for a0, b0 in zip(a, b):
            if a0.shape != b0.shape or not torch.allclose(a0, b0, 2e-5, 2e-5):
                return False

    for options, message in (
        ([], 'require concrete'),
        (['inputshape=[9,3,4],[9,3,4]'], 'above range'),
        (['inputshape=[3,3,4],[4,3,4]'], 'shared input symbol mismatch'),
    ):
        result = subprocess.run(['../../src/pnnx', 'test_pt2_dynamic.pt2', 'fp16=0', *options], capture_output=True, text=True)
        if result.returncode == 0 or message not in result.stderr:
            print(result.stderr)
            return False

    # PyTorch rejects the min=2 domain because (batch // 2) != 1 is required.
    net = Floor().eval()
    x = torch.rand(4, 12)
    dim = torch.export.Dim('floor_batch', min=4, max=8)
    ep = torch.export.export(net, (x,), dynamic_shapes=({0: dim},))
    torch.export.save(ep, 'test_pt2_dynamic_floor.pt2')
    os.system('../../src/pnnx test_pt2_dynamic_floor.pt2 fp16=0 inputshape=[4,12] inputshape2=[6,12]')

    import test_pt2_dynamic_floor_pnnx
    converted = test_pt2_dynamic_floor_pnnx.Model().eval()
    for x0 in (x, torch.rand(6, 12)):
        a = net(x0)
        b = converted(x0)
        if a.shape != b.shape or not torch.equal(a, b):
            return False

    net = TensorArguments().eval()
    x = torch.rand(3, 2)
    dim = torch.export.Dim('batch', min=2, max=8)
    ep = torch.export.export(net, (x,), dynamic_shapes=({0: dim},))
    torch.export.save(ep, 'test_pt2_dynamic_tensor_arguments.pt2')
    os.system('../../src/pnnx test_pt2_dynamic_tensor_arguments.pt2 fp16=0 inputshape=[3,2] inputshape2=[5,2]')

    import test_pt2_dynamic_tensor_arguments_pnnx
    converted = test_pt2_dynamic_tensor_arguments_pnnx.Model().eval()
    for x0 in (x, torch.rand(5, 2)):
        expected = net(x0)
        actual = converted(x0)
        if len(actual) != len(expected):
            return False
        for a, b in zip(actual, expected):
            if not isinstance(a, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                return False

    net = Multiply().eval()
    x = torch.rand(3, 3, 4)
    dim = torch.export.Dim('batch', min=2, max=8)
    ep = torch.export.export(net, (x,), dynamic_shapes=({0: dim},))
    torch.export.save(ep, 'test_pt2_dynamic_multiply.pt2')
    os.system('../../src/pnnx test_pt2_dynamic_multiply.pt2 fp16=0 inputshape=[3,3,4] inputshape2=[5,3,4]')

    import test_pt2_dynamic_multiply_pnnx
    converted = test_pt2_dynamic_multiply_pnnx.Model().eval()
    for x0 in (x, torch.rand(5, 3, 4)):
        a = net(x0)
        b = converted(x0)
        if a.shape != b.shape or not torch.equal(a, b):
            return False
    return True


if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
