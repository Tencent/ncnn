# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import re
import subprocess

import torch


def has_torch_export():
    numbers = re.match(r"(\d+)\.(\d+)", torch.__version__)
    torch_version = tuple(map(int, numbers.groups())) if numbers else (0, 0)
    return torch_version >= (2, 2) and hasattr(torch, "export") and hasattr(torch.export, "save")


def load_pnnx_model(basename):
    module_name = basename + "_pnnx"
    spec = importlib.util.spec_from_file_location(module_name, module_name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def torchscript_to_pnnx(basename, inputshape):
    pnnx = os.path.join("..", "src", "pnnx")
    subprocess.check_call([pnnx, basename + ".pt", "inputshape=" + inputshape])
    return load_pnnx_model(basename)


def exported_program_to_pnnx(model, inputs, basename, dynamic_shapes=None):
    if not has_torch_export():
        return None

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    pt2path = basename + ".pt2"
    exported_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    torch.export.save(exported_program, pt2path)

    pnnx = os.path.join("..", "src", "pnnx")
    subprocess.check_call([pnnx, pt2path])
    return load_pnnx_model(basename)
