# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""PNNX tensor storage/dtype tests, not a claim about native ncnn input dtypes.

The C++ companion supplies an IR-only Python fixture, independently of PT2
serializer/operator support. Real torch.export archives exercise the full CLI
at every optimization level; native ncnn inference is deliberately not run.
"""

import argparse
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import has_exported_program, run_pnnx


def tensor_bytes(tensor):
    # Flatten before reinterpreting: dtype view of a rank-zero tensor is not
    # supported when the item sizes differ. uint8 has a built-in NumPy dtype.
    if tensor.numel() == 0:
        return b""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def check_tensor(actual, expected):
    assert isinstance(actual, torch.Tensor), type(actual)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert tensor_bytes(actual) == tensor_bytes(expected), (actual, expected)


def check_outputs(actual, expected):
    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    assert len(actual) == len(expected), (len(actual), len(expected))
    for a, b in zip(actual, expected):
        check_tensor(a, b)


def import_generated(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_buffers(model, expected):
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    state = model.state_dict()
    assert all(p.is_floating_point() or p.is_complex() for p in parameters.values())
    for value in expected:
        matches = [name for name, buf in buffers.items()
                   if buf.dtype == value.dtype and buf.shape == value.shape
                   and tensor_bytes(buf) == tensor_bytes(value)]
        assert matches, (value, list(buffers))
        for name in matches:
            assert name not in parameters
            check_tensor(state[name], value)


def find_ir_test_executable(path):
    if path:
        result = Path(path).resolve()
        assert result.is_file(), result
        return str(result)
    for directory in (Path.cwd(), Path.cwd() / "Release", Path.cwd() / "Debug"):
        for name in ("test_pnnx_ir_tensor_contract", "test_pnnx_ir_tensor_contract.exe"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError("build test_pnnx_ir_tensor_contract or pass --ir-test-executable")


def test_ir_fixture(directory, executable):
    prefix = directory / "ir_dtype_fixture"
    subprocess.run([executable, "--python-fixture", prefix.as_posix()], check=True)
    module = import_generated(prefix.with_suffix(".py"))
    model = module.Model().eval()
    expected = (
        torch.tensor(1.25, dtype=torch.float32),
        torch.tensor(-1.25, dtype=torch.float16),
        torch.tensor(1.5, dtype=torch.bfloat16),
        torch.tensor(True),
        torch.tensor(-7, dtype=torch.int32),
        torch.tensor(1099511627776, dtype=torch.int64),
        torch.tensor(1.5 - 2j, dtype=torch.complex32),
        torch.empty((2, 0, 3), dtype=torch.bfloat16),
        torch.empty((0,), dtype=torch.complex32),
        torch.empty((0, 2), dtype=torch.bool),
        torch.tensor([[1., -2., .5], [4., -0., 6.]], dtype=torch.bfloat16),
        torch.tensor([1 + 2j, -3 + .5j], dtype=torch.complex32),
        torch.tensor(1.25, dtype=torch.float32),
    )
    with torch.no_grad():
        check_outputs(model(), expected)
    check_buffers(model, (expected[3], expected[4], expected[5], expected[9]))
    assert "identity.counter" in dict(model.named_buffers())
    assert "identity.flag" in dict(model.named_buffers())
    assert "identity.running_mean" in dict(model.named_buffers())
    check_tensor(model.identity.running_mean, torch.ones(1, dtype=torch.bfloat16))
    # Scalars, empty tensors and complex values must remain in state_dict too.
    check_tensor(model.state_dict()["scalar_bf16_data"], expected[2])
    check_tensor(model.state_dict()["scalar_c32_data"], expected[6])
    check_tensor(model.state_dict()["empty_bf16_data"], expected[7])
    test_generated_loader(model)


def test_generated_loader(model):
    cases = (
        ("float32", torch.tensor(-0., dtype=torch.float32)),
        ("float16", torch.tensor(-1.25, dtype=torch.float16)),
        ("bfloat16", torch.tensor(1.5, dtype=torch.bfloat16)),
        ("chalf", torch.tensor(1.5 - 2j, dtype=torch.complex32)),
        ("bool", torch.tensor(True)),
        ("int32", torch.tensor(-7, dtype=torch.int32)),
        ("bfloat16", torch.empty((0,), dtype=torch.bfloat16)),
        ("bfloat16", torch.empty((2, 0, 3), dtype=torch.bfloat16)),
        ("chalf", torch.empty((3, 0), dtype=torch.complex32)),
    )
    for dtype, expected in cases:
        raw = tensor_bytes(expected)
        with io.BytesIO() as stream:
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr("value", raw)
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                actual = model.load_pnnx_bin_as_tensor(archive, "value", tuple(expected.shape), dtype)
        # The returned tensor owns its storage after the archive is closed.
        check_tensor(actual, expected)
        for wrong in (raw + b"\x00", raw[:-1] if raw else b"\x00\x00"):
            with io.BytesIO() as stream:
                with zipfile.ZipFile(stream, "w") as archive:
                    archive.writestr("value", wrong)
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    try:
                        model.load_pnnx_bin_as_tensor(archive, "value", tuple(expected.shape), dtype)
                    except ValueError as error:
                        assert "invalid attribute payload size" in str(error), error
                    else:
                        raise AssertionError("malformed payload was accepted: " + dtype)


class ScalarModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(-1.25, dtype=dtype))
        self.register_buffer("flag", torch.tensor(True))
        self.register_buffer("index", torch.tensor(-7, dtype=torch.int32))
        self.register_buffer("large_index", torch.tensor(1099511627776, dtype=torch.int64))
        self.register_buffer("empty", torch.empty((0, 2), dtype=dtype))

    def forward(self, x):
        # Return the weight as well, so a scalar expression rewrite cannot hide
        # a lost attribute dtype or rank by returning the right product alone.
        return x * self.weight, self.weight, self.flag, self.index, self.large_index, self.empty


class StridedBFloat16Model(nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(-16, 24, dtype=torch.float32).to(torch.bfloat16)
        self.weight = nn.Parameter(storage.as_strided((4, 3), (7, 2), 3))
        self.register_buffer("scale", torch.tensor(1.5, dtype=torch.bfloat16))
        self.register_buffer("empty", torch.empty((2, 0, 3), dtype=torch.bfloat16))

    def forward(self, x):
        return F.linear(x, self.weight), self.weight, self.scale, self.empty


class ComplexHalfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(1.5 - 2j, dtype=torch.complex32))
        self.register_buffer("vector", torch.tensor([1 + 2j, -3 + .5j], dtype=torch.complex32))
        self.register_buffer("empty", torch.empty((2, 0), dtype=torch.complex32))

    def forward(self, x):
        # Avoid depending on experimental complex-half arithmetic kernels.
        return x, self.scalar, self.vector, self.empty


def test_exported_case(directory, name, model, inputs, buffers=(), pnnx_executable=None):
    model.eval()
    with torch.no_grad():
        expected = model(*inputs)
    exported = torch.export.export(model, inputs)
    archive_path = directory / (name + ".pt2")
    torch.export.save(exported, str(archive_path))
    assert zipfile.is_zipfile(archive_path), archive_path
    reloaded = torch.export.load(str(archive_path))
    with torch.no_grad():
        check_outputs(reloaded.module()(*inputs), expected)
    if isinstance(model, StridedBFloat16Model):
        assert not model.weight.is_contiguous()
        assert model.weight.storage_offset() == 3
        # Verify this really exercises serialized strided storage, not a
        # synthetic metadata dictionary or a silently contiguous test weight.
        saved_weight = reloaded.state_dict["weight"]
        assert saved_weight.stride() == model.weight.stride()
        assert saved_weight.storage_offset() == model.weight.storage_offset()

    for optlevel in (0, 1, 2):
        prefix = directory / (name + "_opt" + str(optlevel))
        arguments = ("optlevel=" + str(optlevel), "fp16=0")
        if pnnx_executable is None:
            result = run_pnnx(archive_path.as_posix(), prefix.as_posix(), arguments, capture_output=True)
        else:
            # CTest supplies the exact configuration-specific binary on MSVC.
            result = subprocess.run(
                [pnnx_executable, archive_path.as_posix(),
                 "pnnxparam=" + prefix.as_posix() + ".pnnx.param",
                 "pnnxbin=" + prefix.as_posix() + ".pnnx.bin",
                 "pnnxpy=" + prefix.as_posix() + "_pnnx.py",
                 "ncnnparam=" + prefix.as_posix() + ".ncnn.param",
                 "ncnnbin=" + prefix.as_posix() + ".ncnn.bin",
                 "ncnnpy=" + prefix.as_posix() + "_ncnn.py", *arguments],
                check=False, capture_output=True, text=True,
            )
        assert result.returncode == 0, result.stdout + result.stderr
        module = import_generated(Path(prefix.as_posix() + "_pnnx.py"))
        generated = module.Model().eval()
        with torch.no_grad():
            check_outputs(generated(*inputs), expected)
        check_buffers(generated, buffers)
        if isinstance(model, StridedBFloat16Model):
            weights = [value for value in generated.state_dict().values()
                       if value.shape == model.weight.shape and value.dtype == torch.bfloat16]
            assert weights, "bf16 weight was promoted before Python serialization"
            for weight in weights:
                check_tensor(weight, model.weight)
                assert weight.is_contiguous(), "PNNX must materialize logical tensor order"
        # Check the generated convenience inference path without executing any
        # generated ncnn code. Its inputs are deterministic after seed(0).
        if isinstance(model, ScalarModel):
            torch.manual_seed(0)
            random_input = torch.rand((), dtype=inputs[0].dtype)
            with torch.no_grad():
                check_outputs(module.test_inference(), model(random_input))


def test(ir_test_executable=None, pnnx_executable=None):
    executable = find_ir_test_executable(ir_test_executable)
    if pnnx_executable is not None:
        pnnx_executable = str(Path(pnnx_executable).resolve())
    with tempfile.TemporaryDirectory(prefix="pnnx_dtype_") as temporary:
        directory = Path(temporary)
        test_ir_fixture(directory, executable)
        if not has_exported_program():
            print("SKIP real PT2 archives: torch.export.save is unavailable in " + torch.__version__)
            return True
        for dtype, name in ((torch.float32, "scalar_f32"), (torch.float16, "scalar_f16"),
                            (torch.bfloat16, "scalar_bf16")):
            model = ScalarModel(dtype)
            test_exported_case(directory, name, model, (torch.tensor(2., dtype=dtype),),
                               (model.flag, model.index, model.large_index), pnnx_executable)
        test_exported_case(directory, "strided_bf16", StridedBFloat16Model(),
                           (torch.tensor([[1., 0., -1.], [2., -1., 1.]], dtype=torch.bfloat16),),
                           pnnx_executable=pnnx_executable)
        test_exported_case(directory, "complex32", ComplexHalfModel(), (torch.ones(2),),
                           pnnx_executable=pnnx_executable)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-test-executable")
    parser.add_argument("--pnnx-executable")
    arguments = parser.parse_args()
    raise SystemExit(0 if test(arguments.ir_test_executable, arguments.pnnx_executable) else 1)