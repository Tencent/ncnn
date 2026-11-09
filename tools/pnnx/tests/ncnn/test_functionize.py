# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
import subprocess
import tempfile

import ncnn
import torch


class Assignment(torch.nn.Module):
    def forward(self, x, y):
        z = x + 1
        z[:, :2] = y
        return z


class Chained(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        r = y[:, 1:].relu_()
        r[:, 1:].add_(1)
        return y


class FullSlice(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        y[:, :].add_(2)
        return y


class Rectangle(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        q[:, :, 1:5, 2:6] = src
        return q


class SelectSlice(torch.nn.Module):
    def forward(self, x, src):
        x = x + 1
        x[:, :, 3, :2] = src
        return x


class MultiAxis(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        q[:, 1:3, 1:5, 2:6] = src
        return q


class SelectChain(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        q[:, 1, 2, :2] = src
        return q


class NegativeAxis(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        q.select(-2, 3)[:, :, :2] = src
        return q


class SameAxis(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        q[:, :, 1:5, :][:, :, 1:3, 2:6] = src
        return q


class ObservedView(torch.nn.Module):
    def forward(self, x, src):
        q = x + 1
        view = q[:, :, 1:5, :]
        before = view + 2
        view[:, :, :, 2:6] = src
        return q, before, view


def run(binary, model, inputs, copies, crops, second=None, extra=()):
    torch.jit.trace(model, inputs, check_trace=False).save('model.pt')
    shape = lambda xs: ','.join('[' + ','.join(map(str, x.shape)) + ']' for x in xs)
    args = ['inputshape=' + shape(inputs)]
    if second is not None:
        args.append('inputshape2=' + shape(second))
    subprocess.run([binary, 'model.pt', 'optlevel=2', 'fp16=0', *args], check=True)
    param = Path('model.ncnn.param').read_text()
    assert 'Tensor.expand_as' not in param and 'Tensor.type_as' not in param
    for layer, expected in [('CopyTo', copies), ('Crop', crops)]:
        count = sum(line.split()[0] == layer for line in param.splitlines()[2:])
        assert count == expected, (type(model).__name__, layer, count, expected)
    with ncnn.Net() as net:
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param('model.ncnn.param') == 0
        assert net.load_model('model.ncnn.bin') == 0
        for xs in [inputs] + ([second] if second is not None else []) + list(extra):
            expected = model(*[x.clone() for x in xs])
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            with net.create_extractor() as ex:
                for i, x in enumerate(xs):
                    assert ex.input('in%d' % i, ncnn.Mat(x.numpy(), batch_index=233).clone()) == 0
                for i, y in enumerate(expected):
                    status, output = ex.extract('out%d' % i)
                    assert status == 0
                    actual = torch.from_numpy(output.numpy(batch_index=233)).clone()
                    assert actual.shape == y.shape
                    assert torch.allclose(actual, y, 1e-5, 1e-5)


def test():
    binary = str(Path('../../src/pnnx').resolve())
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        try:
            x = torch.arange(-4, 4).float().reshape(2, 4)
            x2 = torch.arange(-4, 8).float().reshape(3, 4)
            x3 = torch.arange(-4, 16).float().reshape(5, 4)
            run(binary, Assignment(), (x, torch.ones(2, 2)), 1, 0, (x2, torch.full((3, 2), 2.)),
                ((x3, torch.full((5, 2), 3.)),))
            run(binary, Chained(), (x,), 3, 3, (x2,), ((x3,),))
            run(binary, Assignment(), (x, torch.tensor([[2., 3.]])), 1, 0)
            run(binary, FullSlice(), (x,), 0, 0)

            torch.manual_seed(7767517)
            x = torch.randn(2, 3, 6, 8)
            x2 = torch.randn(3, 3, 7, 9)
            run(binary, Rectangle(), (x, torch.randn(2, 3, 4, 4)), 1, 0,
                (x2, torch.randn(3, 3, 4, 4)))
            run(binary, SelectSlice(), (x, torch.randn(2, 3, 2)), 1, 0,
                (x2, torch.randn(3, 3, 2)))
            run(binary, MultiAxis(), (x, torch.randn(2, 2, 4, 4)), 1, 0)
            run(binary, SelectChain(), (x, torch.randn(2, 2)), 1, 0)
            run(binary, NegativeAxis(), (x, torch.randn(2, 3, 2)), 1, 0)
            run(binary, SameAxis(), (x, torch.randn(2, 3, 2, 4)), 2, 1)
            run(binary, ObservedView(), (x, torch.randn(2, 3, 4, 4)), 1, 2)
        finally:
            os.chdir(cwd)
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
