#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch


PNNX = Path(sys.argv[1]).resolve()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


def run_pnnx(work_dir, model_path, *arguments):
    return subprocess.run(
        [str(PNNX), model_path.name, *arguments],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def save_exported_program(model, archive_path, example_inputs=None, **export_kwargs):
    if example_inputs is None:
        example_inputs = (torch.ones(2, 4),)
    program = torch.export.export(model.eval(), example_inputs, **export_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(program, archive_path)


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


@contextmanager
def temporary_work_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def load_generated_module(work_dir, basename, suffix="_pnnx"):
    module_path = work_dir / f"{basename}{suffix}.py"
    module_name = f"test_exported_program_{basename}{suffix}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated module {module_path}")

    module = importlib.util.module_from_spec(spec)
    with working_directory(work_dir):
        spec.loader.exec_module(module)
    return module
