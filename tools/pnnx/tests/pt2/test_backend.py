# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import pathlib
import subprocess

import torch
from packaging import version


class HalfWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.25, -2.5, 0.125, 4.0], dtype=torch.float16))

    def forward(self, x):
        return x * self.weight


class ConstantIndices(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.register_buffer('indices', torch.tensor([[0, 3, 9]], dtype=dtype))

    def forward(self, x):
        return self.embedding(self.indices) + x


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, model, x, options=()):
    prefix = 'test_pt2_backend_' + name
    model.eval()
    torch.export.save(torch.export.export(model, (x,)), prefix + '.pt2')

    # Start with fresh backend paths so artifact checks refer to this run.
    ncnn_paths = [pathlib.Path(prefix + suffix) for suffix in ('.ncnn.param', '.ncnn.bin', '_ncnn.py')]
    for path in ncnn_paths:
        path.unlink(missing_ok=True)
    result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0', *options],
                            capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        print(name, result.stderr)
        return False
    for suffix in ('.pnnx.param', '.pnnx.bin', '_pnnx.py'):
        if not pathlib.Path(prefix + suffix).is_file():
            return False
    actual = load_module(pathlib.Path(prefix + '_pnnx.py')).Model().eval()(x)
    expected = model(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
        print(name, 'pnnx output mismatch')
        return False

    if not all(path.is_file() for path in ncnn_paths):
        return False
    if x.dtype in (torch.int32, torch.int64):
        if 'Embed' not in ncnn_paths[0].read_text() or 'dtype=torch.int)' not in ncnn_paths[2].read_text():
            return False
        if x.dtype == torch.int64 and 'replace ncnn input torch.long type with torch.int' not in result.stderr:
            return False
    if isinstance(model, ConstantIndices):
        # Confirm that this actually tests serialized integer indices rather
        # than an embedding result folded into a floating-point attribute.
        if 'Embed' not in ncnn_paths[0].read_text() or 'MemoryData' not in ncnn_paths[0].read_text():
            return False

    # The generated ncnn example creates the same seed-0 inputs as test(). For
    # int64 embedding it explicitly creates i32 inputs, not reinterpreted bytes.
    actual = load_module(ncnn_paths[2]).test_inference()
    if actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
        print(name, 'ncnn output mismatch', actual, expected)
        return False
    return True


def test():
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.rand(2, 4)
    if not check('half_weight', HalfWeight(), x):
        return False
    for dtype in (torch.int32, torch.int64):
        for shape in ((3,), (1, 3), (2, 3)):
            model = torch.nn.Embedding(10, 4).eval()
            torch.manual_seed(0)
            indices = torch.randint(10, shape, dtype=dtype)
            name = 'embedding_' + str(dtype).split('.')[-1] + '_' + '_'.join(map(str, shape))
            if not check(name, model, indices):
                return False
        # Keep the functional embedding form as well as the optlevel=2 module.
        model = torch.nn.Embedding(10, 4).eval()
        torch.manual_seed(0)
        indices = torch.randint(10, (1, 3), dtype=dtype)
        if not check('functional_embedding_' + str(dtype).split('.')[-1], model, indices, options=('optlevel=0',)):
            return False
        model = ConstantIndices(dtype).eval()
        torch.manual_seed(0)
        if not check('constant_indices_' + str(dtype).split('.')[-1], model, torch.rand(1, 3, 4)):
            return False

    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
