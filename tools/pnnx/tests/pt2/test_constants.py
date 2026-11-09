# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
import subprocess

import torch

from common import rewrite_archive


class Factories(torch.nn.Module):
    def forward(self, x):
        i = torch.arange(4)
        mask = (i[:, None] <= i[None, :]).to(torch.float32)
        return (x + torch.ones((4, 4)), x + torch.full((4, 4), 0.1),
                x + torch.arange(4, dtype=torch.float32), x + torch.hann_window(4),
                x + mask, x + torch.triu(torch.ones((4, 4))),
                x + x.new_full((4, 4), 1e-6))


class ShapeFactory(torch.nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x + torch.full((n, 3), 0.1), x.reshape(n * 3)


class Uninitialized(torch.nn.Module):
    def forward(self, x):
        return torch.empty((2, 3)) + x


class Mutation(torch.nn.Module):
    def forward(self, x):
        a = torch.ones((2, 3))
        b = a.view(3, 2)
        b.add_(x)
        return a, b


class IntegerConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.tensor([2**40 + 3, 5], dtype=torch.int64))

    def forward(self, x):
        return x + (self.a * 2 + 1)


class IntegerFactories(torch.nn.Module):
    def forward(self, x):
        return (x + torch.full((2,), 16777217., dtype=torch.int64),
                x + torch.arange(16777217., 16777219., dtype=torch.int64),
                x + torch.full_like(x, 16777217.),
                x + torch.scalar_tensor(16777217., dtype=torch.int64))


class DynamicIntegerFactory(torch.nn.Module):
    def forward(self, x):
        return x + torch.full((x.shape[0], 3), 16777217.9, dtype=torch.int64)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False
        if expected.is_floating_point() or expected.is_complex():
            return torch.allclose(actual, expected, 2e-5, 2e-5)
        return torch.equal(actual, expected)
    return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))


def convert(name, model, x, options=(), dynamic=False):
    prefix = 'test_pt2_constants_' + name
    shapes = ({0: torch.export.Dim('batch', min=2, max=8)},) if dynamic else None
    program = torch.export.export(model, (x,), dynamic_shapes=shapes)
    torch.export.save(program, prefix + '.pt2')
    result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0', *options], capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(result.stderr)
    module = load(Path(prefix + '_pnnx.py'))
    if not equal(module.Model().eval()(x), model(x)):
        raise RuntimeError(name + ': converted values differ')
    return prefix, module


def test():
    torch.manual_seed(0)
    x = torch.rand(4, 4)
    net = Factories().eval()
    prefix, _ = convert('factories', net, x)
    param = Path(prefix + '.pnnx.param').read_text()
    if any(op in param for op in ('torch.ones ', 'torch.full ', 'torch.arange ', 'torch.hann_window ', 'torch.triu ', 'Tensor.to ')):
        return False
    if not equal(load(Path(prefix + '_ncnn.py')).test_inference(), net(x)):
        return False

    x = torch.rand(4, 3)
    net = ShapeFactory().eval()
    prefix, _ = convert('static_shape', net, x, ['inputshape=[4,3]'], True)
    param = Path(prefix + '.pnnx.param').read_text()
    if 'torch.full ' in param or 'Tensor.size ' in param or '(?,3)' in param:
        return False

    prefix, module = convert('dynamic_shape', net, x, ['inputshape=[4,3]', 'inputshape2=[6,3]'], True)
    param = Path(prefix + '.pnnx.param').read_text()
    if 'torch.full ' not in param:
        return False
    x2 = torch.rand(6, 3)
    if not equal(module.Model().eval()(x2), net(x2)):
        return False

    convert('mutation', Mutation().eval(), torch.rand(3, 2))
    convert('integer', IntegerConstants().eval(), torch.ones(2, dtype=torch.int64))
    net = IntegerFactories().eval()
    x = torch.ones(2, dtype=torch.int64)
    prefix, _ = convert('integer_factories', net, x)

    def integer_spelling(program, entries):
        for node in program['graph_module']['graph']['nodes']:
            for argument in node['inputs']:
                value = argument['arg'].get('as_float')
                if isinstance(value, float) and value.is_integer():
                    argument['arg']['as_float'] = int(value)

    rewritten = 'test_pt2_constants_integer_spelling'
    rewrite_archive(prefix + '.pt2', rewritten + '.pt2', integer_spelling)
    result = subprocess.run(['../../src/pnnx', rewritten + '.pt2', 'fp16=0'], capture_output=True, text=True)
    if result.returncode not in (0, 2) or not equal(load(Path(rewritten + '_pnnx.py')).Model().eval()(x), net(x)):
        return False
    net = DynamicIntegerFactory().eval()
    _, module = convert('dynamic_integer_factory', net, torch.ones(4, 3, dtype=torch.int64),
                        ['inputshape=[4,3]i64', 'inputshape2=[6,3]i64'], True)
    x = torch.ones(6, 3, dtype=torch.int64)
    if not equal(module.Model().eval()(x), net(x)):
        return False

    # Uninitialized allocations must remain runtime operations.
    net = Uninitialized().eval()
    torch.export.save(torch.export.export(net, (torch.rand(2, 3),)), 'test_pt2_constants_empty.pt2')
    result = subprocess.run(['../../src/pnnx', 'test_pt2_constants_empty.pt2', 'fp16=0'], capture_output=True, text=True)
    return result.returncode in (0, 2) and 'torch.empty ' in Path('test_pt2_constants_empty.pnnx.param').read_text()


if __name__ == '__main__':
    exit(0 if test() else 1)
