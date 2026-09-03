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


def export_model(model, inputs, name, model_format, dynamic_shapes=None, check_trace=True):
    if model_format == "torchscript":
        path = name + "_torchscript.pt"
        torch.jit.trace(model, inputs, check_trace=check_trace).save(path)
        return path
    if model_format == "pt2":
        if not has_exported_program():
            raise RuntimeError("torch.export.save is unavailable in torch " + torch.__version__)
        path = name + "_pt2.pt2"
        torch.export.save(
            torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes),
            path,
        )
        return path
    raise ValueError("unknown model format " + model_format)


def run_pnnx(model_path, output_prefix, arguments=(), capture_output=False):
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
    return subprocess.run(
        command,
        check=False,
        capture_output=capture_output,
        text=capture_output,
    )


def convert_model(model_path, output_prefix, arguments=()):
    result = run_pnnx(model_path, output_prefix, arguments)
    if result.returncode != 0:
        raise RuntimeError("pnnx conversion failed for " + model_path)
    return output_prefix + "_pnnx.py"


def import_model(path, module_name=None):
    name = module_name or os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def export_convert_import(model, inputs, name, model_format, arguments=(), check_trace=True):
    model_path = export_model(model, inputs, name, model_format, check_trace=check_trace)
    output_prefix = name + "_" + model_format
    generated_path = convert_model(model_path, output_prefix, arguments)
    return import_model(generated_path, output_prefix + "_pnnx")


def test_model_formats(model, inputs, expected, name, compare=torch.equal, check_trace=True,
                       unsupported_by_torch_export=None, unsupported_by_pnnx_pt2=None,
                       converted_inputs=None, torchscript_inputs=None, pt2_inputs=None):
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    converted_inputs = inputs if converted_inputs is None else converted_inputs
    torchscript_inputs = converted_inputs if torchscript_inputs is None else torchscript_inputs
    pt2_inputs = converted_inputs if pt2_inputs is None else pt2_inputs
    torchscript_model = export_convert_import(model, inputs, name, "torchscript", check_trace=check_trace)
    torchscript_result = torchscript_model(*torchscript_inputs)
    torchscript_outputs = torchscript_result if isinstance(torchscript_result, tuple) else (torchscript_result,)
    if len(expected_outputs) != len(torchscript_outputs) or not all(
        compare(a, b) for a, b in zip(expected_outputs, torchscript_outputs)
    ):
        return False

    if not has_exported_program():
        print("SKIP PT2: torch.export.save is unavailable in torch " + torch.__version__)
        return True

    if unsupported_by_torch_export:
        try:
            export_model(model, inputs, name, "pt2")
        except Exception as exception:
            if unsupported_by_torch_export in str(exception):
                print("UNSUPPORTED_BY_TORCH_EXPORT: " + unsupported_by_torch_export)
                return True
            raise
        raise RuntimeError("torch.export unexpectedly supports " + name)

    if unsupported_by_pnnx_pt2:
        model_path = export_model(model, inputs, name, "pt2")
        result = run_pnnx(model_path, name + "_pt2", capture_output=True)
        output = result.stdout + result.stderr
        if result.returncode != 0 and unsupported_by_pnnx_pt2 in output:
            print("UNSUPPORTED_BY_PNNX_PT2: " + unsupported_by_pnnx_pt2)
            return True
        if result.returncode == 0:
            raise RuntimeError("pnnx unexpectedly supports PT2 conversion for " + name)
        raise RuntimeError("unexpected pnnx PT2 failure for " + name + "\n" + output)

    pt2_model = export_convert_import(model, inputs, name, "pt2")
    pt2_result = pt2_model(*pt2_inputs)
    pt2_outputs = pt2_result if isinstance(pt2_result, tuple) else (pt2_result,)
    return len(expected_outputs) == len(pt2_outputs) and all(
        compare(a, b) for a, b in zip(expected_outputs, pt2_outputs)
    )