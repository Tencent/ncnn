# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
import pathlib
import subprocess

import torch

from pnnx_test_utils import _check_output, _run_ncnn


class Model(torch.nn.Module):
    def forward(self, x):
        return x[:x.shape[0] - 1]


def load_model(path):
    spec = importlib.util.spec_from_file_location("pnnx_dynamic_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    model = Model().eval()
    exported = torch.export.export(
        model,
        (torch.randn(4, 3),),
        dynamic_shapes=({0: torch.export.Dim("batch", min=3, max=8)},),
    )
    torch.export.save(exported, args.workdir / "model.pt2")

    command = [
        str(pathlib.Path(args.pnnx).resolve()),
        "model.pt2",
        "inputshape=[6,3]",
        "inputshape2=[4,3]",
    ]
    process = subprocess.run(command, cwd=args.workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        print(process.stdout)
        return 1

    previous_workdir = pathlib.Path.cwd()
    try:
        os.chdir(args.workdir)
        converted = load_model(args.workdir / "model_pnnx.py").eval()
        for shape in ((6, 3), (4, 3)):
            x = torch.randn(shape)
            expected = model(x)
            _check_output(expected, converted(x))
            actual = _run_ncnn(args.workdir, (x,), 1)
            if actual is not None:
                _check_output(expected, actual, 1e-3, 1e-3, True)
    finally:
        os.chdir(previous_workdir)

    invalid = subprocess.run(
        [str(pathlib.Path(args.pnnx).resolve()), "model.pt2", "inputshape=[9,3]"],
        cwd=args.workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if invalid.returncode == 0 or "outside the exported range" not in invalid.stdout:
        print(invalid.stdout)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
