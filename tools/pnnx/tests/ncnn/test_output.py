#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnnx_test_utils import convert_and_import
from pnnx_test_utils import _import_generated_module


class Model(nn.Module):
    def __init__(self, output_type):
        super().__init__()
        self.output_type = output_type

    def forward(self, x):
        a = torch.relu(x)
        b = torch.sigmoid(x)
        if self.output_type == "single":
            return [a]
        if self.output_type == "list":
            return [a, b]
        return (a, [b, (torch.tanh(x),)])


def test():
    torch.manual_seed(0)
    x = torch.rand(3, 4)
    for output_type in ("single", "list", "nested"):
        net = Model(output_type).eval()
        expected = net(x)
        flat_expected, _ = tree_flatten(expected)
        for optlevel in (0, 1, 2):
            print("output type=%s optlevel=%d" % (output_type, optlevel), flush=True)
            module = convert_and_import(net, (x,), "test_output",
                                        pnnx_args=("optlevel=%d" % optlevel, "fp16=0"))
            torch.testing.assert_close(module.test_inference(), expected)

            ncnn_path = Path(module.__file__.replace("_pnnx.py", "_ncnn.py"))
            native = _import_generated_module(ncnn_path, "test_output_ncnn")
            actual, _ = tree_flatten(native.test_inference())
            if len(actual) != len(flat_expected):
                return False
            for a, b in zip(flat_expected, actual):
                torch.testing.assert_close(b, a)

    return True


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
