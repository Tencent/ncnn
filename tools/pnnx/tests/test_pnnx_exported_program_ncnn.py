# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""PT2 -> PNNX -> native ncnn regression with one shared float32 input.

The parent CMake registration must gate on torch.export.save and usable ncnn
bindings, and pass --pnnx-executable explicitly. Missing dependencies and
conversion/inference failures are errors, never successful skips.
"""

import argparse
from pathlib import Path
import tempfile
from unittest import mock

import ncnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pnnx_test_utils


ARTIFACT_SUFFIXES = (
    ".pnnx.param", ".pnnx.bin", "_pnnx.py",
    ".ncnn.param", ".ncnn.bin", "_ncnn.py",
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.linear = nn.Linear(4 * 2 * 3, 5)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32)
                # Nonzero, signed, varied weights/biases expose lost storage
                # and ordering errors without depending on random initializers.
                magnitude = ((values * 7 + index * 3).remainder(23) + 1) / 64
                sign = torch.where((values + index).remainder(2) == 0, 1., -1.)
                parameter.copy_((magnitude * sign).reshape(parameter.shape))

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (2, 3))
        x = torch.flatten(x, 1)
        return self.linear(x)


def check_output(label, actual, expected):
    assert isinstance(actual, torch.Tensor), (label, type(actual))
    assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
    # Check metadata before allclose so broadcasting cannot hide a layout bug.
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert torch.isfinite(actual).all().item(), label + ": non-finite output"
    difference = (actual - expected).abs()
    max_abs = difference.max().item()
    max_rel = (difference / expected.abs().clamp_min(1e-8)).max().item()
    diagnostic = ("{}: max_abs={:.8g}, max_rel={:.8g} (denominator floor=1e-8)"
                  .format(label, max_abs, max_rel))
    print(diagnostic)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), diagnostic


def convert(archive_path, prefix, executable, x):
    input_shape = "inputshape=[" + ",".join(str(size) for size in x.shape) + "]"
    # Patch discovery only: the real helper still runs the requested binary
    # with its timeout/crash checks and exact artifact/cache cleanup.
    with mock.patch.object(pnnx_test_utils, "find_pnnx", return_value=executable):
        result = pnnx_test_utils.run_pnnx(
            archive_path.as_posix(), prefix.as_posix(),
            arguments=(input_shape, "fp16=0", "optlevel=2"), capture_output=True,
        )
    diagnostic = ("pnnx return code: {}\nstdout:\n{}\nstderr:\n{}"
                  .format(result.returncode, result.stdout, result.stderr))
    assert result.returncode == 0, diagnostic
    artifacts = {}
    for suffix in ARTIFACT_SUFFIXES:
        path = Path(prefix.as_posix() + suffix)
        assert path.is_file(), "missing artifact: " + str(path) + "\n" + diagnostic
        assert path.stat().st_size > 0, "empty artifact: " + str(path) + "\n" + diagnostic
        artifacts[suffix] = path
    return artifacts


def native_inference(artifacts, x):
    assert x.dtype == torch.float32 and x.ndim == 4 and x.shape[0] == 1
    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = 1
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        net.opt.use_bf16_storage = False
        status = net.load_param(artifacts[".ncnn.param"].as_posix())
        assert status == 0, ("ncnn load_param failed", status, artifacts[".ncnn.param"])
        status = net.load_model(artifacts[".ncnn.bin"].as_posix())
        assert status == 0, ("ncnn load_model failed", status, artifacts[".ncnn.bin"])
        with net.create_extractor() as extractor:
            # convert_input/eliminate_output canonicalize these names, also
            # used by save_ncnn.cpp. Like the native batch-one tests, supply
            # CHW, then restore only the known singleton output batch axis.
            input_mat = ncnn.Mat(x.squeeze(0).contiguous().numpy()).clone()
            status = extractor.input("in0", input_mat)
            assert status == 0, ("ncnn input in0 failed", status)
            status, output = extractor.extract("out0")
            assert status == 0, ("ncnn extract out0 failed", status)
            # Copy before the extractor/Net is destroyed; do not cast dtype
            # or reshape to the expected tensor, which would hide regressions.
            actual = torch.from_numpy(np.array(output, copy=True)).unsqueeze(0)
    return actual


def test(pnnx_executable):
    executable = Path(pnnx_executable).resolve()
    assert executable.is_file(), "pnnx executable not found: " + str(executable)
    assert pnnx_test_utils.has_exported_program(), "torch.export.save is required"

    torch.manual_seed(0)
    model = Model().eval()
    # Batch one follows the supported CHW native layout. The asymmetric
    # spatial pattern and non-global pooling retain ordering-sensitive data.
    values = torch.arange(3 * 8 * 10, dtype=torch.float32)
    x = ((values * 13).remainder(113) - 56).reshape(1, 3, 8, 10) / 32
    with torch.no_grad():
        expected = model(x)
    assert expected.dtype == torch.float32 and expected.shape == (1, 5)
    assert torch.isfinite(expected).all().item()
    assert torch.count_nonzero(expected).item() == expected.numel()

    with tempfile.TemporaryDirectory(prefix="pnnx_pt2_ncnn_") as temporary:
        directory = Path(temporary).resolve()
        archive_path = directory / "model.pt2"
        exported = torch.export.export(model, (x,))
        # POSIX paths keep generated Python string literals valid on Windows.
        torch.export.save(exported, archive_path.as_posix())
        assert archive_path.is_file() and archive_path.stat().st_size > 0

        first_bytes = None
        for run in ("first", "repeat"):
            # Independent prefixes in a fresh directory cannot reuse the
            # first conversion's files if the second invocation omits one.
            artifacts = convert(archive_path, directory / run, executable.as_posix(), x)
            generated = pnnx_test_utils.import_model(artifacts["_pnnx.py"].as_posix())
            with torch.no_grad():
                check_output(run + " PNNX", generated(x), expected)
            # Do not call generated test_inference(): it makes its own random
            # input. Exercise the actual native Net with the exact eager x.
            check_output(run + " ncnn", native_inference(artifacts, x), expected)

            model_bytes = {
                suffix: artifacts[suffix].read_bytes()
                for suffix in (".pnnx.param", ".pnnx.bin", ".ncnn.param", ".ncnn.bin")
            }
            if first_bytes is None:
                first_bytes = model_bytes
            else:
                for suffix, payload in model_bytes.items():
                    assert payload == first_bytes[suffix], "nondeterministic artifact: " + suffix
        print("Repeat conversion: PNNX/ncnn param and bin files are byte-identical")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnnx-executable", required=True, help="path to the built pnnx executable")
    arguments = parser.parse_args()
    raise SystemExit(0 if test(arguments.pnnx_executable) else 1)