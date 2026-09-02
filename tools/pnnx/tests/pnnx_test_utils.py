# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess

import torch


def has_exported_program():
    return hasattr(torch, "export") and hasattr(torch.export, "save")


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


def export_model(model, inputs, name, model_format):
    if model_format == "torchscript":
        path = name + "_torchscript.pt"
        torch.jit.trace(model, inputs).save(path)
        return path
    if model_format == "pt2":
        if not has_exported_program():
            raise RuntimeError("torch.export.save is unavailable in torch " + torch.__version__)
        path = name + "_pt2.pt2"
        torch.export.save(torch.export.export(model, inputs), path)
        return path
    raise ValueError("unknown model format " + model_format)


def convert_model(model_path, output_prefix, arguments=()):
    command = [
        find_pnnx(),
        model_path,
        "pnnxparam=" + output_prefix + ".pnnx.param",
        "pnnxbin=" + output_prefix + ".pnnx.bin",
        "pnnxpy=" + output_prefix + "_pnnx.py",
        "ncnnparam=" + output_prefix + ".ncnn.param",
        "ncnnbin=" + output_prefix + ".ncnn.bin",
        "ncnnpy=" + output_prefix + "_ncnn.py",
        *arguments,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError("pnnx conversion failed for " + model_path)
    return output_prefix + "_pnnx.py"


def import_model(path, module_name=None):
    name = module_name or os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def export_convert_import(model, inputs, name, model_format, arguments=()):
    model_path = export_model(model, inputs, name, model_format)
    output_prefix = name + "_" + model_format
    generated_path = convert_model(model_path, output_prefix, arguments)
    return import_model(generated_path, output_prefix + "_pnnx")