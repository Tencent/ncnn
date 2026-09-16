# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PT2 CLI contract tests; no CTest registration is added here.

Run from a build test directory or pass --pnnx-executable explicitly. Every
conversion uses run_pnnx's argument, timeout and crash checks. Only its output
cleanup is disabled: these fresh temporary workspaces must expose premature
artifacts and must not hide deletion of existing files by the CLI itself.

NumPy inputs are validated examples, not value specialization or execution.
inputshape2/input2 validate a second sample, not a runtime dynamic-shape guard.
Export/import failures are errors, not skips; ONNX output alone is conditional
on the binary's explicit build-without-onnx diagnostic. Native ncnn inference
is not exercised by this frontend contract suite.
"""

import argparse
from pathlib import Path
import tempfile
from unittest import mock
import zipfile

import numpy as np
import torch
from torch import nn

import pnnx_test_utils


OUTPUT_SUFFIXES = (
    ".pnnx.param", ".pnnx.bin", "_pnnx.py",
    ".ncnn.param", ".ncnn.bin", "_ncnn.py",
)
NO_ONNX = "pnnx build without onnx-zero support, skip saving onnx"


class Model(nn.Module):
    def forward(self, x, y):
        # Independent operations preserve both user inputs, including dtypes.
        return torch.relu(x), torch.relu(y)


def output_path(prefix, suffix):
    return Path(prefix.as_posix() + suffix)


def new_prefix(directory, name):
    case = directory / name
    case.mkdir()
    return case / "model"


def conversion(path, prefix, arguments, timeout):
    # Do not override the six output keys forbidden by run_pnnx. The optional
    # ONNX key is allowed and must also stay inside the temporary workspace.
    with mock.patch.object(pnnx_test_utils, "_cleanup_outputs", return_value=None):
        return pnnx_test_utils.run_pnnx(
            path.as_posix(), prefix.as_posix(),
            arguments=("fp16=0", "pnnxonnx=" + output_path(prefix, ".pnnx.onnx").as_posix(),
                       *arguments),
            capture_output=True, timeout=timeout,
        )


def diagnostic(result):
    return ("command: " + repr(result.args) + "\nreturn code: " + str(result.returncode)
            + "\n" + result.stdout + "\n" + result.stderr)


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes() if p.is_file() else None
            for p in directory.rglob("*")}


def check_success(path, prefix, arguments, model, inputs, timeout):
    result = conversion(path, prefix, arguments, timeout)
    details = diagnostic(result)
    assert result.returncode == 0, details
    for suffix in OUTPUT_SUFFIXES:
        assert output_path(prefix, suffix).is_file(), (suffix, details)
    assert output_path(prefix, ".pnnx.onnx").is_file() == (NO_ONNX not in result.stderr), details
    generated = pnnx_test_utils.import_model(output_path(prefix, "_pnnx.py").as_posix())
    with torch.no_grad():
        expected = model(*inputs)
        actual = generated(*inputs)
    assert type(actual) is tuple and len(actual) == len(expected), details
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype and a.shape == b.shape, (a, b, details)
        assert torch.equal(a, b), (a, b, details)
    return result


def check_failure(path, prefix, arguments, needle, timeout):
    before = snapshot(prefix.parent)
    result = conversion(path, prefix, arguments, timeout)
    details = diagnostic(result)
    # Crashes, timeouts and launch errors already raise in run_pnnx.
    assert result.returncode != 0, details
    assert needle in result.stderr, (needle, details)
    assert "############# pass_level2" not in result.stderr, details
    assert snapshot(prefix.parent) == before, "CLI modified artifacts before validation\n" + details
    return result


def save_export(directory, name, model, inputs, dynamic_shapes=None):
    path = directory / (name + ".pt2")
    torch.export.save(torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes), path.as_posix())
    return path


def save_numpy(directory, name, tensor):
    path = directory / (name + ".npy")
    np.save(path, tensor.detach().cpu().numpy(), allow_pickle=False)
    return path.as_posix()


def test_inputs(directory, model, inputs, path, timeout):
    x, y = inputs
    x_path = save_numpy(directory, "x", x)
    y_path = save_numpy(directory, "y", y)
    numpy_pair = x_path + "," + y_path
    shapes = "[2,3]f32,[2,3]f32"
    for name, arguments in (
        ("graph_metadata", ()),
        ("tagged_f32", ("inputshape=" + shapes, "device=cpu", "customop=", "moduleop=")),
        ("numpy", ("input=" + numpy_pair,)),
        ("second_shape", ("inputshape=" + shapes, "inputshape2=" + shapes)),
        ("second_numpy", ("input=" + numpy_pair, "input2=" + numpy_pair)),
        ("shape_and_numpy", ("inputshape=" + shapes, "input2=" + numpy_pair)),
    ):
        check_success(path, new_prefix(directory, name), arguments, model, inputs, timeout)

    for option in ("inputshape", "inputshape2"):
        for name, value, needle in (
            ("dtype", "[2,3]f32,[2,3]i64", "dtype mismatch: expected f32 but got i64"),
            ("unknown_dtype", "[2,3]f99,[2,3]f32", "dtype mismatch: expected f32 but got f99"),
            ("rank", "[6]f32,[2,3]f32", "rank mismatch"),
            ("dimension", "[2,4]f32,[2,3]f32", "dimension 1 is 4, expected 3"),
            ("count_few", "[2,3]f32", "inputshape count mismatch"),
            ("count_many", shapes + ",[2,3]f32", "inputshape count mismatch"),
        ):
            # Validate inputshape2 independently, not only when inputshape exists.
            check_failure(path, new_prefix(directory, option + "_" + name),
                          (option + "=" + value,), needle, timeout)

    wrong_dtype = save_numpy(directory, "wrong_dtype", y.to(torch.float64))
    wrong_rank = save_numpy(directory, "wrong_rank", x.reshape(6))
    wrong_dimension = save_numpy(directory, "wrong_dimension", torch.ones(2, 4, device="cpu"))
    truncated = directory / "truncated.npy"
    truncated.write_bytes(Path(x_path).read_bytes()[:-1])
    for option in ("input", "input2"):
        for name, value, needle in (
            ("dtype", x_path + "," + wrong_dtype, "dtype mismatch: expected f32 but got f64"),
            ("rank", wrong_rank + "," + y_path, "rank mismatch"),
            ("dimension", wrong_dimension + "," + y_path, "dimension 1 is 4, expected 3"),
            ("count_few", x_path, "inputshape count mismatch"),
            ("count_many", numpy_pair + "," + x_path, "inputshape count mismatch"),
            ("truncated", truncated.as_posix() + "," + y_path, "exceeds remaining file size"),
            ("missing", (directory / "missing.npy").as_posix(), "npy load failed"),
        ):
            check_failure(path, new_prefix(directory, option + "_" + name),
                          (option + "=" + value,), needle, timeout)

    for option, shape_option in (("input", "inputshape"), ("input2", "inputshape2")):
        arguments = (option + "=" + numpy_pair, shape_option + "=" + shapes)
        for index, args in enumerate((arguments, arguments[::-1])):
            check_failure(path, new_prefix(directory, option + "_conflict_" + str(index)),
                          args, "parameter conflict", timeout)

    # Omitted suffixes are not an implicit request to cast a PT2 input to f32.
    mixed_inputs = (x, y.to(torch.float16))
    mixed_path = save_export(directory, "mixed_dtype", model, mixed_inputs)
    mixed_numpy = x_path + "," + save_numpy(directory, "y_half", mixed_inputs[1])
    for index, arguments in enumerate((
        (),
        ("inputshape=[2,3],[2,3]",),
        ("inputshape=[2,3]f32,[2,3]",),
        ("inputshape=[2,3],[2,3]f16",),
        ("inputshape=[2,3]f32,[2,3]f16",),
        ("inputshape2=[2,3],[2,3]",),
        ("input=" + mixed_numpy,),
        ("inputshape=[2,3]f64,[2,3]f32", "inputshape=[2,3],[2,3]"),
    )):
        check_success(mixed_path, new_prefix(directory, "mixed_" + str(index)),
                      arguments, model, mixed_inputs, timeout)
    for option in ("inputshape", "inputshape2"):
        check_failure(mixed_path, new_prefix(directory, "mixed_bad_" + option),
                      (option + "=" + shapes,), "dtype mismatch: expected f16 but got f32", timeout)


def test_unsupported_options(directory, path, timeout):
    for name, argument, needle in (
        ("customop", "customop=not_loaded.dll", "pt2 customop is not supported"),
        ("moduleop", "moduleop=models.Custom", "pt2 moduleop is not supported"),
        ("gpu", "device=gpu", "only device=cpu is supported"),
        ("unknown_device", "device=unrecognized", "only device=cpu is supported"),
    ):
        for existing in (False, True):
            prefix = new_prefix(directory, name + "_" + str(existing))
            if existing:
                for suffix in (*OUTPUT_SUFFIXES, ".pnnx.onnx"):
                    output_path(prefix, suffix).write_bytes(b"existing user output\n")
            check_failure(path, prefix, (argument,), needle, timeout)


def test_dynamic_samples(directory, model, timeout):
    inputs = (torch.ones(3, 3, device="cpu"), torch.zeros(3, 3, device="cpu"))
    batch = torch.export.Dim("batch", min=2, max=8)
    path = save_export(directory, "dynamic", model, inputs,
                       {"x": {0: batch}, "y": {0: batch}})
    primary = (torch.ones(4, 3, device="cpu"), torch.zeros(4, 3, device="cpu"))
    second = (torch.ones(5, 3, device="cpu"), torch.zeros(5, 3, device="cpu"))
    second_numpy = ",".join(save_numpy(directory, "dynamic_" + str(i), t)
                            for i, t in enumerate(second))
    for name, argument in (("shape", "inputshape2=[5,3]f32,[5,3]f32"),
                           ("numpy", "input2=" + second_numpy)):
        prefix = new_prefix(directory, "dynamic_second_" + name)
        check_success(path, prefix, ("inputshape=[4,3]f32,[4,3]f32", argument),
                      model, primary, timeout)
        input_lines = [line for line in output_path(prefix, ".pnnx.param").read_text().splitlines()
                       if line.startswith("pnnx.Input")]
        assert len(input_lines) == 2 and all("(4,3)f32" in line for line in input_lines), input_lines
    for name, arguments, needle in (
        ("range", ("inputshape=[9,3],[9,3]",), "allowed range is [2, 8]"),
        ("shared", ("inputshape=[4,3],[5,3]",), "shared symbol requires 4"),
        ("second_range", ("inputshape=[4,3],[4,3]", "inputshape2=[9,3],[9,3]"),
         "allowed range is [2, 8]"),
    ):
        check_failure(path, new_prefix(directory, "dynamic_bad_" + name), arguments, needle, timeout)


def test_formats(directory, path, timeout):
    for name, records, needle in (
        ("unknown", {"unrelated": b"data"}, "unsupported model zip archive"),
        # Even TorchScript-looking records cannot override a corrupt PT2 marker.
        ("incomplete", {"archive_format": b"pt2", "archive_version": b"0",
                        "data.pkl": b"", "constants.pkl": b""}, "incomplete pt2 model archive"),
    ):
        target = directory / (name + ".pt2")
        with zipfile.ZipFile(target, "w") as archive:
            for key, value in records.items():
                archive.writestr(key, value)
        check_failure(target, new_prefix(directory, "format_" + name), (), needle, timeout)

    with zipfile.ZipFile(path) as archive:
        records = {name: archive.read(name) for name in archive.namelist()}
    model_names = [name for name in records if name.endswith("serialized_exported_program.json")
                   or ((name.startswith("models/") or "/models/" in name) and name.endswith(".json"))]
    assert len(model_names) == 1, model_names
    records[model_names[0]] = b"{"
    corrupt = directory / "corrupt.pt2"
    with zipfile.ZipFile(corrupt, "w") as archive:
        for key, value in records.items():
            archive.writestr(key, value)
    check_failure(corrupt, new_prefix(directory, "format_corrupt"), (),
                  "load exported program failed:", timeout)
    check_failure(corrupt, new_prefix(directory, "reject_before_import"), ("customop=unused.dll",),
                  "pt2 customop is not supported", timeout)

    truncated = directory / "truncated.pt2"
    truncated.write_bytes(path.read_bytes()[:32])
    check_failure(truncated, new_prefix(directory, "format_truncated"), (),
                  "unsupported model format: failed to read model zip archive", timeout)


def test_output_failures(directory, path, timeout):
    prefix = new_prefix(directory, "missing_output_directory") / "missing" / "model"
    result = conversion(path, prefix, (), timeout)
    assert result.returncode != 0 and "save pnnx graph failed" in result.stderr, diagnostic(result)
    assert not prefix.parent.exists(), diagnostic(result)

    # Directories are portable fopen failures (including on Windows/admin).
    # This reaches every output stage without overriding run_pnnx's output keys.
    for suffix, needle in (
        (".pnnx.param", "save pnnx graph failed"),
        (".pnnx.bin", "save pnnx graph failed"),
        ("_pnnx.py", "save pnnx python failed"),
        (".pnnx.onnx", "save pnnx onnx failed"),
        (".ncnn.param", "save ncnn failed"),
        (".ncnn.bin", "save ncnn failed"),
        ("_ncnn.py", "save ncnn failed"),
    ):
        prefix = new_prefix(directory, "blocked" + suffix)
        blocked = output_path(prefix, suffix)
        blocked.mkdir()
        keep = blocked / "user_file"
        keep.write_bytes(b"keep this file\n")
        later_output = output_path(prefix, "_ncnn.py")
        if suffix != "_ncnn.py":
            later_output.write_bytes(b"existing later output\n")
        result = conversion(path, prefix, (), timeout)
        details = diagnostic(result)
        if suffix == ".pnnx.onnx" and NO_ONNX in result.stderr:
            # Verify the explicitly disabled build feature, not a broad skip.
            assert result.returncode == 0, details
        else:
            assert result.returncode != 0 and needle in result.stderr, (needle, details)
            if suffix != "_ncnn.py":
                assert later_output.read_bytes() == b"existing later output\n", details
            if suffix != ".pnnx.param":
                # No transactional rollback of already written outputs.
                assert output_path(prefix, ".pnnx.param").is_file(), details
        assert keep.read_bytes() == b"keep this file\n", details


def test(pnnx_executable=None, timeout=None):
    if not pnnx_test_utils.has_exported_program():
        raise RuntimeError("PT2 CLI tests require torch.export.save in torch " + torch.__version__)
    executable = Path(pnnx_executable or pnnx_test_utils.find_pnnx()).resolve()
    if not executable.is_file():
        raise RuntimeError("pnnx executable was not found: " + str(executable))
    model = Model().cpu().eval()
    x = torch.tensor([[-3., 2., -1.], [0., 4., 5.]], dtype=torch.float32, device="cpu")
    inputs = (x, -x)
    with tempfile.TemporaryDirectory(prefix="pnnx_cli_") as temporary:
        directory = Path(temporary)
        # Retain real timeout/crash handling while selecting the exact binary.
        with mock.patch.object(pnnx_test_utils, "find_pnnx", return_value=str(executable)):
            path = save_export(directory, "float_model", model, inputs)
            test_inputs(directory, model, inputs, path, timeout)
            test_unsupported_options(directory, path, timeout)
            test_dynamic_samples(directory, model, timeout)
            test_formats(directory, path, timeout)
            test_output_failures(directory, path, timeout)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnnx-executable")
    parser.add_argument("--timeout", type=float, default=None)
    arguments = parser.parse_args()
    raise SystemExit(0 if test(arguments.pnnx_executable, arguments.timeout) else 1)