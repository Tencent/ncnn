#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
import pathlib
import struct
import subprocess
import tempfile

import torch


class StateAndConstants(torch.nn.Module):
    def __init__(self):
        super(StateAndConstants, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 16)
        self.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) / 8)
        self.register_buffer("persistent_buffer", torch.tensor([0.25, -0.5, 0.75, -1], dtype=torch.float32))
        self.register_buffer("non_persistent_buffer", torch.tensor([1, 1.5, 2, 2.5], dtype=torch.float32), persistent=False)
        self.tensor_constant = torch.tensor([0.125, 0.25, 0.375, 0.5], dtype=torch.float32)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias) + self.persistent_buffer + self.non_persistent_buffer + self.tensor_constant


class StridedTensors(torch.nn.Module):
    def __init__(self):
        super(StridedTensors, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(30, dtype=torch.float32).reshape(5, 6).transpose(0, 1))
        shared = torch.arange(20, dtype=torch.float32)
        self.register_buffer("offset_view", shared[3:8])
        self.register_buffer("strided_view", shared[3:13:2])

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.offset_view + self.strided_view


class StructuredIo(torch.nn.Module):
    def forward(self, x, y, scale=3):
        value = (x + y) * scale
        return {"value": value, "summary": (value.mean(), value.sum(dim=1))}


class BFloat16Weights(torch.nn.Module):
    def __init__(self):
        super(BFloat16Weights, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(2, 1, 2, 2) / 16)
        self.bias = torch.nn.Parameter(torch.arange(2, dtype=torch.bfloat16) / 8)

    def forward(self, x):
        return torch.nn.functional.conv2d(x.to(torch.bfloat16), self.weight, self.bias).float()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, cwd=None):
    process = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise AssertionError("command failed: %s\n%s" % (" ".join(str(x) for x in command), process.stdout))
    return process.stdout


def check_output(expected, actual, rtol=0, atol=0, reshape=False):
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    if len(expected) != len(actual):
        raise AssertionError("output count differs")
    for a, b in zip(expected, actual):
        if reshape and a.numel() == b.numel():
            b = b.reshape(a.shape)
        torch.testing.assert_close(b, a, rtol=rtol, atol=atol)


def check_case(args, root, name, model, export_args, export_kwargs, inference_args):
    workdir = root / name
    workdir.mkdir()
    model.eval()
    torch.export.save(torch.export.export(model, export_args, kwargs=export_kwargs), workdir / "model.pt2")

    version = torch.__version__.split("+")[0]
    legacy = tuple(int(x) for x in version.split(".")[:2]) < (2, 8)
    archive_format = "pt2-legacy-exported-program" if legacy else "pt2-archive"
    container = "legacy-exported-program" if legacy else "archive"
    run([args.archive_tester, workdir / "model.pt2", archive_format])
    run([args.archive_tester, "--program-archive", workdir / "model.pt2"])
    run([args.archive_tester, "--weights-archive", workdir / "model.pt2", name])
    run([args.graph_tester, workdir / "model.pt2", name])

    output = run([args.pnnx, "model.pt2"], workdir)
    if "pt2 container=%s" % container not in output or "schema=8." not in output or "opset=aten:10" not in output or "producer=torch-%s" % version not in output:
        raise AssertionError("unexpected pnnx result\n" + output)
    for filename in ("model.pnnx.param", "model.pnnx.bin", "model_pnnx.py", "model.ncnn.param", "model.ncnn.bin", "model_ncnn.py"):
        if not (workdir / filename).is_file():
            raise AssertionError("missing PT2 conversion output " + filename)

    oldcwd = pathlib.Path.cwd()
    try:
        os.chdir(workdir)
        with torch.no_grad():
            expected = model(*inference_args)
        if isinstance(expected, dict):
            expected = expected["value"], expected["summary"][0], expected["summary"][1]
        pnnx_output = load_module(workdir / "model_pnnx.py", "pt2_%s_pnnx" % name).test_inference()
        ncnn_output = load_module(workdir / "model_ncnn.py", "pt2_%s_ncnn" % name).test_inference()
    finally:
        os.chdir(oldcwd)

    check_output(expected, pnnx_output)
    check_output(expected, ncnn_output, 1e-3, 1e-3, True)


def check_bfloat16(args, root):
    workdir = root / "bfloat16_weights"
    workdir.mkdir()
    model = BFloat16Weights().eval()
    torch.export.save(torch.export.export(model, (torch.rand(1, 1, 3, 3),)), workdir / "model.pt2")
    run([args.archive_tester, "--weights-archive", workdir / "model.pt2", "bfloat16_weights"])
    run([args.graph_tester, workdir / "model.pt2", "bfloat16_weights"])
    run([args.pnnx, "model.pt2", "fp16=0"], workdir)
    weights = struct.pack("=8f", *(x / 16 for x in range(8)))
    if weights not in (workdir / "model.ncnn.bin").read_bytes():
        raise AssertionError("bfloat16 weights were not converted to float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=pathlib.Path, required=True)
    parser.add_argument("--archive-tester", type=pathlib.Path, required=True)
    parser.add_argument("--graph-tester", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.pnnx = args.pnnx.resolve()
    args.archive_tester = args.archive_tester.resolve()
    args.graph_tester = args.graph_tester.resolve()

    import ncnn
    if not hasattr(ncnn, "Net") or not hasattr(ncnn, "Mat"):
        raise RuntimeError("complete ncnn Python binding is required")

    torch.manual_seed(0)
    cases = (
        ("state_and_constants", StateAndConstants(), (torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10,), {}),
        ("strided_tensors", StridedTensors(), (torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10,), {}),
        ("structured_io", StructuredIo(), (torch.arange(8, dtype=torch.float32).reshape(2, 4), torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)), {"scale": 3}),
    )

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for name, model, export_args, export_kwargs in cases:
            torch.manual_seed(0)
            inference_args = tuple(torch.rand(x.shape, dtype=x.dtype) for x in export_args)
            check_case(args, root, name, model, export_args, export_kwargs, inference_args)
        check_bfloat16(args, root)


if __name__ == "__main__":
    main()
