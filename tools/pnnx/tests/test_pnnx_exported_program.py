# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess
import sys

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        y = self.linear(x)
        return torch.relu(y) * self.scale, y


def find_pnnx():
    candidates = [
        os.path.join("..", "src", "pnnx"),
        os.path.join("..", "src", "pnnx.exe"),
        os.path.join("..", "src", "Release", "pnnx.exe"),
        os.path.join("src", "pnnx"),
        os.path.join("src", "pnnx.exe"),
        os.path.join("src", "Release", "pnnx.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError("pnnx executable was not found")


def import_model(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def convert(pnnx, model_path, output_prefix):
    result = subprocess.run(
        [
            pnnx,
            model_path,
            "pnnxparam=" + output_prefix + ".pnnx.param",
            "pnnxbin=" + output_prefix + ".pnnx.bin",
            "pnnxpy=" + output_prefix + "_pnnx.py",
            "ncnnparam=" + output_prefix + ".ncnn.param",
            "ncnnbin=" + output_prefix + ".ncnn.bin",
            "ncnnpy=" + output_prefix + "_ncnn.py",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("pnnx conversion failed for " + model_path)


def test():
    if not hasattr(torch, "export") or not hasattr(torch.export, "save"):
        print("SKIP: torch.export.save is unavailable in torch " + torch.__version__)
        return True

    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 3)
    expected = model(x)

    torchscript_path = "test_pnnx_exported_program_torchscript.pt"
    pt2_path = "test_pnnx_exported_program_pt2.pt2"
    torch.jit.trace(model, (x,)).save(torchscript_path)
    torch.export.save(torch.export.export(model, (x,)), pt2_path)

    pnnx = find_pnnx()
    convert(pnnx, torchscript_path, "test_pnnx_exported_program_torchscript")
    convert(pnnx, pt2_path, "test_pnnx_exported_program_pt2")

    torchscript_model = import_model(
        "test_pnnx_exported_program_torchscript_pnnx.py",
        "test_pnnx_exported_program_torchscript_pnnx",
    )
    pt2_model = import_model(
        "test_pnnx_exported_program_pt2_pnnx.py",
        "test_pnnx_exported_program_pt2_pnnx",
    )

    torchscript_output = torchscript_model(x)
    pt2_output = pt2_model(x)
    return all(
        torch.allclose(eager, torchscript)
        and torch.allclose(eager, exported)
        and torch.allclose(torchscript, exported)
        for eager, torchscript, exported in zip(expected, torchscript_output, pt2_output)
    )


if __name__ == "__main__":
    sys.exit(0 if test() else 1)