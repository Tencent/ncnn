# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import re
import subprocess
import time

import torch


WINDOWS_DLL_NOT_FOUND = (-1073741515, 3221225781)


def run_pnnx(arguments):
    pnnx = os.path.join("..", "src", "pnnx")
    command = [pnnx] + arguments
    environment = os.environ.copy()
    torch_library_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    environment["PATH"] = torch_library_path + os.pathsep + environment.get("PATH", "")
    attempts = 3 if os.name == "nt" else 1
    for attempt in range(attempts):
        result = subprocess.run(command, check=False, env=environment)
        if result.returncode == 0:
            return
        if result.returncode not in WINDOWS_DLL_NOT_FOUND or attempt + 1 == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(2)


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
    run_pnnx([basename + ".pt", "inputshape=" + inputshape])
    return load_pnnx_model(basename)


def exported_program_to_pnnx(model, inputs, basename, dynamic_shapes=None):
    if not has_torch_export():
        return None

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    pt2path = basename + ".pt2"
    exported_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    torch.export.save(exported_program, pt2path)

    run_pnnx([pt2path])
    return load_pnnx_model(basename)
