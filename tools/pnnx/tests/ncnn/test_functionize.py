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


class HalfAllocationBoundary(torch.nn.Module):
    def forward(self, x):
        y = torch.nn.functional.avg_pool2d(x.half(), 2)
        y.relu_()
        return y.float()


class EvalDropout(torch.nn.Module):
    def forward(self, x):
        base = x + 1
        view = base.view(-1)
        result = torch.nn.functional.dropout(base, 0.5, False)
        result.add_(2)
        return base, view, result


class EvalDropoutModule(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        base = x + 1
        old = base[..., 1:3]
        before = old.clone()
        result = self.dropout(base)
        result.add_(2)
        return before, base, old, result


class ConditionalAlias(torch.nn.Module):
    __constants__ = ['kind', 'sliced']
    operation = 'transpose'

    def __init__(self, kind, sliced):
        super().__init__()
        self.kind = kind
        self.sliced = sliced

    def forward(self, x):
        base = x + 1
        source = base[..., 1:5] if self.sliced else base
        old = source[..., 1:3]
        before = old.clone()
        if self.kind == 'dropout2d':
            result = torch.nn.functional.dropout2d(source, 0.5, False)
        elif self.kind == 'dropout3d':
            result = torch.nn.functional.dropout3d(source, 0.5, False)
        elif self.kind == 'alpha_dropout':
            result = torch.nn.functional.alpha_dropout(source, 0.5, False)
        elif self.kind == 'feature_alpha_dropout':
            result = torch.nn.functional.feature_alpha_dropout(source, 0.5, False)
        elif self.kind == 'type_as':
            result = source.type_as(x)
        elif self.kind == 'to':
            result = source.to(dtype=torch.float32, copy=False)
        else:
            result = source.to(dtype=torch.float32, copy=True)
        # Unbatched dropout3d returns a view; mutate its 4D source and check that result stays live.
        target = source if self.kind == 'dropout3d' else result
        target.transpose_(-2, -1)
        target.add_(2)
        return before, base, source, result, old


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
        saved = torch.export.load(path).module()
    else:
        torch.jit.trace(model, samples, check_trace=False).save(name + '.pt')
        path = name + '.pt'
        saved = torch.jit.load(path)
    shape = lambda xs: ','.join('[' + ','.join(map(str, x.shape)) + ']' for x in xs)
    args = ['inputshape=' + shape(inputs)]
    if second is not None:
        args.append('inputshape2=' + shape(second))
    assert os.system(' '.join(shlex.quote(arg) for arg in [binary, path, 'fp16=0', *args])) == 0, name
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
        for xs in [inputs] + ([second] if second is not None else []) + list(extra) + [inputs]:
            samples = tuple(x.clone() for x in xs)
            expected = model(*samples)
            assert all(torch.equal(x, y) for x, y in zip(samples, xs)), (name, 'eager input mutation')
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            for label, candidate in [('saved', saved), ('pnnx', converted)]:
                samples = tuple(x.clone() for x in xs)
                actual = candidate(*samples)
                assert all(torch.equal(x, y) for x, y in zip(samples, xs)), (name, label, 'input mutation')
                if isinstance(actual, torch.Tensor):
                    actual = (actual,)
                assert len(actual) == len(expected), (name, label)
                for i, (a, y) in enumerate(zip(actual, expected)):
                    assert a.shape == y.shape and a.dtype == y.dtype, (name, label, i, a.shape, y.shape, a.dtype, y.dtype)
                    assert torch.allclose(a, y, 1e-5, 1e-5), (name, label, i)
            with net.create_extractor() as ex:
                for i, x in enumerate(xs):
                    assert ex.input('in%d' % i, ncnn.Mat(x.numpy(), batch_index=233).clone()) == 0
                for i, y in enumerate(expected):
                    status, output = ex.extract('out%d' % i)
                    assert status == 0
                    actual = torch.from_numpy(output.numpy(batch_index=233)).clone()
                    assert actual.shape == y.shape and actual.dtype == y.dtype, (name, 'ncnn', i, actual.shape, y.shape, actual.dtype, y.dtype)
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

                x = torch.arange(-54, 54).float().reshape(2, 3, 3, 6)
                for dropout in (torch.nn.Dropout(), torch.nn.Dropout2d(), torch.nn.AlphaDropout()):
                    run(binary, frontend, EvalDropoutModule(dropout), (x,), 0, 2, name='module_' + type(dropout).__name__)

                for kind in ('dropout2d', 'dropout3d', 'alpha_dropout', 'feature_alpha_dropout', 'type_as', 'to', 'to_copy'):
                    for sliced in (False, True):
                        if 'dropout' in kind:
                            # Exercise dropout3d's unbatched form within ncnn's rank limit.
                            x = torch.arange(-54, 54).float().reshape(2, 3, 3, 6)
                        else:
                            x = torch.arange(-6, 6).float().reshape(2, 6)
                        copies = int(sliced and kind != 'to_copy')
                        crops = (2 if sliced else 1) * (1 if kind == 'to_copy' else 2)
                        second = None
                        extra = ()
                        if sliced and kind in ('type_as', 'to_copy'):
                            second = (torch.arange(-6, 15).float().reshape(3, 7),)
                            extra = ((torch.arange(-6, 34).float().reshape(5, 8),),)
                        run(binary, frontend, ConditionalAlias(kind, sliced), (x,), copies, crops, second, extra,
                            name=kind + ('_slice' if sliced else '_root'))

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

                if frontend == 'torchscript':
                    x = torch.arange(-24, 24).float().reshape(2, 4, 6)
                    run(binary, frontend, HalfAllocationBoundary(), (x,), 0, 0)
        finally:
            os.chdir(cwd)
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
