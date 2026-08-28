# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import os
import pathlib
import re
import subprocess

import torch


class Model(torch.nn.Module):
    def forward(self, x):
        a = x.sum().item()
        b = x.mean().item()
        c = (a + b) * 2.0 / 3.0
        return x + torch.sym_max(torch.sym_sqrt(c ** 2.0 % 5.0), b)


class ScalarModel(torch.nn.Module):
    def forward(self, x):
        return x.sum().item() / 2.0


def load_model(path):
    spec = importlib.util.spec_from_file_location("pnnx_symfloat_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def run_ncnn(workdir, x):
    import ncnn

    source = (workdir / "model_ncnn.py").read_text()
    input_axis = int(re.search(r'ex\.input\("in0", .*batch_index=(-?\d+)', source).group(1))
    output_axis = int(re.search(r'out0\.numpy\(batch_index=(-?\d+)\)', source).group(1))
    with ncnn.Net() as net:
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        if net.load_param(str(workdir / "model.ncnn.param")) != 0 or net.load_model(str(workdir / "model.ncnn.bin")) != 0:
            raise RuntimeError("failed to load ncnn model")
        with net.create_extractor() as ex:
            if ex.input("in0", ncnn.Mat(x.numpy(), batch_index=input_axis).clone()) != 0:
                raise RuntimeError("failed to set ncnn input")
            ret, output = ex.extract("out0")
            if ret != 0:
                raise RuntimeError("failed to extract ncnn output")
            return torch.from_numpy(output.numpy(batch_index=output_axis))


def convert(pnnx, workdir, model):
    torch.export.save(torch.export.export(model, (torch.rand(2, 3),), strict=False), workdir / "model.pt2")
    result = subprocess.run([pnnx, "model.pt2"], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        return 77

    args.workdir.mkdir(parents=True, exist_ok=True)
    convert(str(pathlib.Path(args.pnnx).resolve()), args.workdir, Model().eval())
    if "pnnx.Assert" not in (args.workdir / "model.pnnx.param").read_text():
        return 1

    previous_workdir = pathlib.Path.cwd()
    try:
        os.chdir(args.workdir)
        x = torch.rand(2, 3)
        expected = Model()(x)
        torch.testing.assert_close(load_model(args.workdir / "model_pnnx.py")(x), expected)
        torch.testing.assert_close(run_ncnn(args.workdir, x), expected, rtol=1e-3, atol=1e-3)
    finally:
        os.chdir(previous_workdir)

    scalar = args.workdir / "scalar"
    scalar.mkdir(exist_ok=True)
    convert(str(pathlib.Path(args.pnnx).resolve()), scalar, ScalarModel().eval())
    previous_workdir = pathlib.Path.cwd()
    try:
        os.chdir(scalar)
        x = torch.rand(2, 3)
        expected = ScalarModel()(x)
        torch.testing.assert_close(load_model(scalar / "model_pnnx.py")(x), expected)
        torch.testing.assert_close(run_ncnn(scalar, x).reshape(()), torch.tensor(expected), rtol=1e-3, atol=1e-3)
    finally:
        os.chdir(previous_workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
