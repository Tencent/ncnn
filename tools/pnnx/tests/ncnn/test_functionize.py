# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
from pathlib import Path
import shlex
import tempfile

import ncnn
import torch
from packaging import version


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


class EvalDropout(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.view(-1)
        result = torch.nn.functional.dropout(base, 0.5, False)
        result.add_(2)
        return base, view, result


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


class ViewMetadata(torch.nn.Module):
    __constants__ = ['operation']

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        base = x + 1
        view = base[..., 1:5]
        sibling = base[..., 2:6]
        before = view.clone()
        if self.operation == 'transpose':
            view.transpose_(0, -1)
        elif self.operation == 't':
            view.t_()
        elif self.operation == 'squeeze':
            view.squeeze_(1)
        else:
            view.unsqueeze_(1)
        result = view.add_(2)
        middle = sibling + 3
        base.mul_(3)
        return before, base, view, sibling, result, middle


class ViewMetadataReplay(torch.nn.Module):
    operation = 'transpose'

    def forward(self, x):
        base = x + 1
        view = base[:, 1:5]
        previous = view.add_(2)
        old = view[:, 1:3]
        before = old.clone()
        result = previous.transpose_(0, 1)
        result[:2].mul_(3)
        middle = old + 1
        base[:, 2:4].add_(4)
        return before, base, view, previous, result, old, middle


class InputViewMetadata(torch.nn.Module):
    operation = 'unsqueeze'

    def forward(self, x):
        view = x[0]
        before = view + 1
        result = view.unsqueeze_(0)
        return x, view, result, before


def run(binary, frontend, model, inputs, copies, crops, second=None, extra=(), name=None):
    name = 'test_functionize_%s_%s' % (name or type(model).__name__, frontend)
    model.eval()
    samples = tuple(x.clone() for x in inputs)
    if frontend == 'pt2':
        dynamic_shapes = None
        if second is not None:
            dynamic_shapes = tuple({i: torch.export.Dim.AUTO for i in range(x.dim()) if x.shape[i] != y.shape[i]}
                                   for x, y in zip(inputs, second))
        program = torch.export.export(model, samples, dynamic_shapes=dynamic_shapes)
        mutable_ops = [node.target._schema.name for node in program.graph.nodes if node.op == 'call_function'
                       and getattr(node.target, '_schema', None) is not None and node.target._schema.is_mutable]
        assert mutable_ops, (name, 'raw export lost the mutation')
        if hasattr(model, 'operation'):
            assert 'aten::' + model.operation + '_' in mutable_ops, (name, 'raw export lost the metadata mutation', mutable_ops)
        torch.export.save(program, name + '.pt2')
        path = name + '.pt2'
    else:
        torch.jit.trace(model, samples, check_trace=False).save(name + '.pt')
        path = name + '.pt'
    shape = lambda xs: ','.join('[' + ','.join(map(str, x.shape)) + ']' for x in xs)
    args = ['inputshape=' + shape(inputs)]
    if second is not None:
        args.append('inputshape2=' + shape(second))
    os.system(' '.join(shlex.quote(arg) for arg in [binary, path, 'fp16=0', *args]))
    spec = importlib.util.spec_from_file_location(name + '_pnnx', name + '_pnnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    converted = module.Model().eval()
    param = Path(name + '.ncnn.param').read_text()
    assert 'Tensor.expand_as' not in param and 'Tensor.type_as' not in param
    for layer, expected in [('CopyTo', copies), ('Crop', crops)]:
        count = sum(line.split()[0] == layer for line in param.splitlines()[2:])
        assert count == expected, (name, layer, count, expected)
    with ncnn.Net() as net:
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param(name + '.ncnn.param') == 0
        assert net.load_model(name + '.ncnn.bin') == 0
        for xs in [inputs] + ([second] if second is not None else []) + list(extra):
            expected = model(*[x.clone() for x in xs])
            actual = converted(*[x.clone() for x in xs])
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            if isinstance(actual, torch.Tensor):
                actual = (actual,)
            assert len(actual) == len(expected), name
            for i, (a, y) in enumerate(zip(actual, expected)):
                assert a.shape == y.shape and a.dtype == y.dtype, (name, 'pnnx', i, a.shape, y.shape)
                assert torch.allclose(a, y, 1e-5, 1e-5), (name, 'pnnx', i)
            with net.create_extractor() as ex:
                for i, x in enumerate(xs):
                    assert ex.input('in%d' % i, ncnn.Mat(x.numpy(), batch_index=233).clone()) == 0
                for i, y in enumerate(expected):
                    status, output = ex.extract('out%d' % i)
                    assert status == 0
                    actual = torch.from_numpy(output.numpy(batch_index=233)).clone()
                    assert actual.shape == y.shape, (name, 'ncnn', i, actual.shape, y.shape)
                    assert torch.allclose(actual, y, 1e-5, 1e-5), (name, 'ncnn', i)


def test():
    binary = str(Path('../../src/pnnx').resolve())
    frontends = ['torchscript']
    if version.parse(torch.__version__) >= version.parse('2.11'):
        frontends.append('pt2')
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        try:
            for frontend in frontends:
                x = torch.arange(-4, 4).float().reshape(2, 4)
                x2 = torch.arange(-4, 8).float().reshape(3, 4)
                x3 = torch.arange(-4, 16).float().reshape(5, 4)
                run(binary, frontend, Assignment(), (x, torch.ones(2, 2)), 1, 0, (x2, torch.full((3, 2), 2.)),
                    ((x3, torch.full((5, 2), 3.)),), name='assignment_dynamic')
                run(binary, frontend, Chained(), (x,), 3, 3, (x2,), ((x3,),))
                run(binary, frontend, Assignment(), (x, torch.tensor([[2., 3.]])), 1, 0, name='assignment_broadcast')
                run(binary, frontend, FullSlice(), (x,), 0, 0)
                run(binary, frontend, EvalDropout(), (x,), 0, 0)

                torch.manual_seed(7767517)
                x = torch.randn(2, 3, 6, 8)
                x2 = torch.randn(3, 3, 7, 9)
                run(binary, frontend, Rectangle(), (x, torch.randn(2, 3, 4, 4)), 1, 0,
                    (x2, torch.randn(3, 3, 4, 4)))
                run(binary, frontend, SelectSlice(), (x, torch.randn(2, 3, 2)), 1, 0,
                    (x2, torch.randn(3, 3, 2)))
                run(binary, frontend, MultiAxis(), (x, torch.randn(2, 2, 4, 4)), 1, 0)
                run(binary, frontend, SelectChain(), (x, torch.randn(2, 2)), 1, 0)
                run(binary, frontend, NegativeAxis(), (x, torch.randn(2, 3, 2)), 1, 0)
                run(binary, frontend, SameAxis(), (x, torch.randn(2, 3, 2, 4)), 2, 1)
                run(binary, frontend, ObservedView(), (x, torch.randn(2, 3, 4, 4)), 1, 2)

                for operation in ('transpose', 't', 'squeeze', 'unsqueeze'):
                    shape = (2, 1, 6) if operation == 'squeeze' else (2, 6)
                    x = torch.arange(-6, 6).float().reshape(shape)
                    run(binary, frontend, ViewMetadata(operation), (x,), 1, 4, name='view_' + operation)
                x = torch.arange(-6, 6).float().reshape(2, 6)
                run(binary, frontend, ViewMetadataReplay(), (x,), 4, 9)
                x = torch.arange(-4, 4).float().reshape(2, 4)
                run(binary, frontend, InputViewMetadata(), (x,), 0, 1)
        finally:
            os.chdir(cwd)
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
