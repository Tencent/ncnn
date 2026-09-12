# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util

import torch
import torch.nn as nn

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export


class DoubleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3).double()

    def forward(self, x):
        return self.linear(x)


class BoolIdentity(nn.Module):
    def forward(self, x):
        return x


class ScalarFloatInput(nn.Module):
    def forward(self, x):
        return x + 1.0


def load_generated_module(basename):
    module_name = basename + "_pnnx"
    spec = importlib.util.spec_from_file_location(module_name, module_name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roundtrip(model, inputs, basename):
    expected = model(*inputs)
    converted = exported_program_to_pnnx(model, inputs, basename)
    if not torch.equal(expected, converted(*inputs)):
        return False

    generated = load_generated_module(basename)
    exported_program = generated.export_exported_program()
    actual = exported_program.module()(*inputs)
    return torch.equal(expected, actual)


def test():
    if not has_torch_export():
        return True

    torch.manual_seed(0)
    if not test_roundtrip(DoubleLinear().eval(), (torch.rand(2, 4, dtype=torch.double),), "test_exported_program_roundtrip_double"):
        return False

    bool_input = torch.tensor([[True, False, True], [False, True, False]])
    if not test_roundtrip(BoolIdentity().eval(), (bool_input,), "test_exported_program_roundtrip_bool"):
        return False

    return test_roundtrip(ScalarFloatInput().eval(), (torch.tensor(2.5),), "test_exported_program_roundtrip_scalar_float_input")


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
