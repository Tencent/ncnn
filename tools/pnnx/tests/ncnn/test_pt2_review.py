# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# PR review 回归：non-persistent buffer 的 constants 路径，以及 mutation
# signature 不得被误暴露成 public output。

import os
import re
import subprocess
import tempfile

import torch
import torch.nn as nn


def _pnnx_binary():
    candidate = os.path.join("..", "..", "src", "pnnx")
    if os.path.exists(candidate):
        return candidate
    return os.environ.get("PNNX_BIN", "pnnx")


def _export_and_convert(model, inputs, inputshape):
    with tempfile.TemporaryDirectory(prefix="pt2_review_") as directory:
        pt2_path = os.path.join(directory, "model.pt2")
        exported = torch.export.export(model.eval(), inputs)
        torch.export.save(exported, pt2_path)

        result = subprocess.run(
            [_pnnx_binary(), pt2_path, f"inputshape={inputshape}"],
            check=False,
            capture_output=True,
            text=True,
        )
        param_path = os.path.join(directory, "model.pnnx.param")
        param = ""
        if os.path.exists(param_path):
            with open(param_path, "r", encoding="utf-8") as file:
                param = file.read()
        return result, param


class NonPersistentBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]), persistent=False)

    def forward(self, x):
        return x * self.scale


class BufferMutation(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x):
        self.state.add_(1.0)
        return x


def main():
    failures = []

    def check(condition, message, output=""):
        print(("ok   " if condition else "FAIL ") + message)
        if not condition:
            if output:
                print(output[-2000:])
            failures.append(message)

    result, _ = _export_and_convert(
        NonPersistentBuffer(), (torch.ones(1, 4),), "[1,4]"
    )
    check(
        result.returncode == 0,
        "review: non-persistent buffer resolves from constants",
        result.stderr,
    )

    result, param = _export_and_convert(
        BufferMutation(), (torch.ones(1, 4),), "[1,4]"
    )
    output_count = len(re.findall(r"^pnnx.Output\s", param, re.MULTILINE))
    check(
        result.returncode == 0 and output_count == 1,
        "review: mutation specs do not add public outputs",
        result.stderr + "\n" + param,
    )

    if failures:
        print("RESULT: %d failed" % len(failures))
        return 1
    print("RESULT: all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
