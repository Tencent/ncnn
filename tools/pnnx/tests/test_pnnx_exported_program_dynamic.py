# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys

import torch
import torch.nn as nn

from pnnx_test_utils import export_model, has_exported_program, run_pnnx


class Model(nn.Module):
    def forward(self, x, y):
        return x + y


def run(*arguments):
    return run_pnnx(
        "test_pnnx_exported_program_dynamic_pt2.pt2",
        "test_pnnx_exported_program_dynamic_pt2",
        arguments,
        capture_output=True,
    )


def test():
    if not has_exported_program():
        print("SKIP: torch.export.save is unavailable in torch " + torch.__version__)
        return True

    batch = torch.export.Dim("batch", min=2, max=8)
    height = torch.export.Dim("height", min=4, max=16)
    width = torch.export.Dim("width", min=4, max=16)
    x = torch.randn(3, 4, 5)
    y = torch.randn(3, 4, 5)
    dynamic_shapes = {
        "x": {0: batch, 1: height, 2: width},
        "y": {0: batch, 1: height, 2: width},
    }
    export_model(
        Model(),
        (x, y),
        "test_pnnx_exported_program_dynamic",
        "pt2",
        dynamic_shapes=dynamic_shapes,
    )

    valid = run("inputshape=[5,8,9],[5,8,9]")
    if valid.returncode != 0:
        print(valid.stdout, valid.stderr)
        return False

    range_error = run("inputshape=[9,8,9],[9,8,9]")
    shared_error = run("inputshape=[5,8,9],[6,8,9]")
    second_error = run(
        "inputshape=[5,8,9],[5,8,9]",
        "inputshape2=[5,17,9],[5,17,9]",
    )

    return (
        range_error.returncode != 0
        and "allowed range is [2, 8]" in range_error.stderr
        and shared_error.returncode != 0
        and "shared symbol requires 5" in shared_error.stderr
        and second_error.returncode != 0
        and "allowed range is [4, 16]" in second_error.stderr
    )


if __name__ == "__main__":
    sys.exit(0 if test() else 1)