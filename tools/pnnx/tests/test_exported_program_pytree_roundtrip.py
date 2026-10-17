# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util

import torch
import torch.nn as nn

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, load_pnnx_model, run_pnnx


class Model(nn.Module):
    def forward(self, pair, tail):
        x, nested = pair
        y, z = nested
        first = x + y
        second = z * tail
        return [first, (second, first - second)]


def close(a, b):
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def load_generated_module(basename):
    module_name = basename + "_pnnx"
    spec = importlib.util.spec_from_file_location(module_name, module_name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test():
    if not has_torch_export():
        return True

    torch.manual_seed(0)
    inputs = ((torch.rand(2, 3), [torch.rand(2, 3), torch.rand(2, 3)]), torch.rand(2, 3))
    net = Model().eval()
    expected = net(*inputs)

    basename = "test_exported_program_pytree_roundtrip"
    converted = exported_program_to_pnnx(net, inputs, basename)
    if not close(expected, converted(*inputs)):
        return False
    try:
        converted(inputs[0][0], inputs[0][1], inputs[1])
        return False
    except ValueError as error:
        if "input tuple/list structure" not in str(error):
            return False

    generated = load_generated_module(basename)
    if not close(expected, generated.test_inference()):
        return False
    default_exported_program = generated.export_exported_program()
    if not close(expected, default_exported_program.module()(*inputs)):
        return False
    exported_program = generated.export_exported_program(inputs)
    if not close(expected, exported_program.module()(*inputs)):
        return False

    roundtrip_basename = basename + "_pnnx"
    run_pnnx([roundtrip_basename + ".pt2"])
    roundtrip = load_pnnx_model(roundtrip_basename)
    return close(expected, roundtrip(*inputs))


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
