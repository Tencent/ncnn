# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import pathlib
import subprocess

import torch


class HalfWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.25, -2.5, 0.125, 4.0], dtype=torch.float16))

    def forward(self, x):
        return x * self.weight


class IntegerCompute(torch.nn.Module):
    def forward(self, x):
        return x + 3


class MixedIndexUse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)

    def forward(self, x):
        return self.embedding(x + 1)


class IntegerAttribute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([2**40, 3], dtype=torch.int64))

    def forward(self, x):
        return x * self.weight


class ConstantIndices(torch.nn.Module):
    def __init__(self, dtype, last=9):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.register_buffer('indices', torch.tensor([[0, 3, last]], dtype=dtype))

    def forward(self, x):
        return self.embedding(self.indices) + x


class StridedView(torch.nn.Module):
    def forward(self, x):
        return torch.as_strided(x.clone(memory_format=torch.contiguous_format), (2, 2), (3, 1), 0)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, model, x, supported, diagnostic='', options=()):
    prefix = 'test_pt2_backend_' + name
    model.eval()
    torch.export.save(torch.export.export(model, (x,)), prefix + '.pt2')

    # Start with fresh backend paths so artifact checks refer to this run.
    ncnn_paths = [pathlib.Path(prefix + suffix) for suffix in ('.ncnn.param', '.ncnn.bin', '_ncnn.py')]
    for path in ncnn_paths:
        path.unlink(missing_ok=True)
    onnx_path = pathlib.Path(prefix + '.pnnx.onnx')
    onnx_path.unlink(missing_ok=True)
    result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0', *options],
                            capture_output=True, text=True, timeout=90)
    expected_status = 0 if supported else 2
    if result.returncode != expected_status:
        print(name, 'expected status', expected_status, 'got', result.returncode, result.stderr)
        return False
    if diagnostic and diagnostic not in result.stderr:
        print(name, 'missing diagnostic', diagnostic, result.stderr)
        return False
    for suffix in ('.pnnx.param', '.pnnx.bin', '_pnnx.py'):
        if not pathlib.Path(prefix + suffix).is_file():
            return False
    actual = load_module(pathlib.Path(prefix + '_pnnx.py')).Model().eval()(x)
    expected = model(x)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.allclose(actual, expected, 2e-5, 2e-5):
        print(name, 'pnnx output mismatch')
        return False

    onnx_enabled = 'build without onnx-zero support' not in result.stderr
    if onnx_enabled:
        if x.dtype == torch.bfloat16:
            # ONNX rejection must still attempt and diagnose ncnn independently.
            if 'pt2 onnx: unsupported' not in result.stderr or onnx_path.exists():
                return False
        elif not onnx_path.is_file():
            return False

    if not supported:
        return ('pt2 ncnn: unsupported' in result.stderr
                and 'pnnx artifacts were saved' in result.stderr
                and all(not path.exists() for path in ncnn_paths))

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


def check_invalid_indices():
    # The ncnn serializer saturates out-of-range i64 values. Reject them at the
    # backend boundary rather than silently selecting a different embedding.
    for index in (-1, 10, 2**40):
        prefix = 'test_pt2_backend_invalid_index_' + str(index).replace('-', 'minus')
        model = ConstantIndices(torch.int64, index).eval()
        x = torch.rand(1, 3, 4)
        torch.export.save(torch.export.export(model, (x,)), prefix + '.pt2')
        paths = [pathlib.Path(prefix + suffix) for suffix in ('.ncnn.param', '.ncnn.bin', '_ncnn.py')]
        for path in paths:
            path.unlink(missing_ok=True)
        result = subprocess.run(['../../src/pnnx', prefix + '.pt2', 'fp16=0'],
                                capture_output=True, text=True, timeout=90)
        if result.returncode != 2 or 'pt2 ncnn: unsupported tensor attribute dtype' not in result.stderr:
            print(result.stderr)
            return False
        if any(path.exists() for path in paths):
            return False
        converted = load_module(pathlib.Path(prefix + '_pnnx.py')).Model().eval()
        try:
            converted(x)
        except IndexError:
            pass
        else:
            return False
    return True


def test():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.rand(2, 4)
    if not check('half_weight', HalfWeight(), x, True):
        return False
    for dtype in (torch.int32, torch.int64):
        for shape in ((3,), (1, 3), (2, 3)):
            model = torch.nn.Embedding(10, 4).eval()
            torch.manual_seed(0)
            indices = torch.randint(10, shape, dtype=dtype)
            name = 'embedding_' + str(dtype).split('.')[-1] + '_' + '_'.join(map(str, shape))
            if not check(name, model, indices, True):
                return False
        # Keep the functional embedding form as well as the optlevel=2 module.
        model = torch.nn.Embedding(10, 4).eval()
        torch.manual_seed(0)
        indices = torch.randint(10, (1, 3), dtype=dtype)
        if not check('functional_embedding_' + str(dtype).split('.')[-1], model, indices, True, options=('optlevel=0',)):
            return False
        model = ConstantIndices(dtype).eval()
        torch.manual_seed(0)
        if not check('constant_indices_' + str(dtype).split('.')[-1], model, torch.rand(1, 3, 4), True):
            return False

    if not check_invalid_indices():
        return False

    cases = (
        ('integer_compute', IntegerCompute(), torch.tensor([[2**40, 1]], dtype=torch.int64), 'non-FP32 runtime input'),
        ('integer_compute_i32', IntegerCompute(), torch.ones(1, 3, dtype=torch.int32), 'non-FP32 runtime input'),
        ('mixed_index_use', MixedIndexUse(), torch.tensor([[0, 1, 2]], dtype=torch.int64), 'non-FP32 runtime input'),
        ('integer_attribute', IntegerAttribute(), torch.rand(1, 2), 'tensor attribute dtype'),
        ('half_input', torch.nn.ReLU(), x.half(), 'non-FP32 runtime input'),
        ('double_input', torch.nn.ReLU(), x.double(), 'non-FP32 runtime input'),
        ('bfloat_input', torch.nn.ReLU(), x.bfloat16(), 'non-FP32 runtime input'),
        ('scalar_input', IntegerCompute(), torch.tensor(2.), 'scalar or empty runtime input'),
        ('empty_input', IntegerCompute(), torch.empty(0, 3), 'scalar or empty runtime input'),
        ('unconverted', StridedView(), torch.rand(2, 3), 'unsupported unconverted operator at torch.as_strided'),
        ('embedding_rank3', torch.nn.Embedding(10, 4), torch.ones(1, 2, 3, dtype=torch.int64), 'non-FP32 runtime input'),
    )
    for name, model, value, diagnostic in cases:
        if not check(name, model, value, False, diagnostic):
            return False
    return True


if __name__ == '__main__':
    exit(0 if test() else 1)
