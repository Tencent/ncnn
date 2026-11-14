# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import subprocess

import onnx
from onnx import helper, TensorProto
import torch
import torch.nn.functional as F


def test():
    x = torch.linspace(-4, 4, 24).reshape(2, 3, 4)
    case = 0
    for op in ("Softmax", "LogSoftmax"):
        for opset in (11, 13):
            for axis in (None, 1, -2):
                name = "test_softmax_legacy_%d" % case
                attrs = {} if axis is None else {"axis": axis}
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node(op, ["x"], ["y"], **attrs)], name,
                        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
                        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])]),
                    opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
                onnx.checker.check_model(model)
                onnx.save(model, name + ".onnx")
                subprocess.run(["../../src/pnnx", name + ".onnx", "inputshape=[2,3,4]", "fp16=0"], check=True)

                spec = importlib.util.spec_from_file_location(name, name + "_pnnx.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                actual = module.Model().eval()(x)

                dim = (1 if opset < 13 else -1) if axis is None else axis
                dim %= x.ndim
                normalized = x
                if opset < 13:
                    normalized = x.reshape(x.shape[:dim].numel(), x.shape[dim:].numel())
                    dim = 1
                expected = F.softmax(normalized, dim) if op == "Softmax" else F.log_softmax(normalized, dim)
                if not torch.allclose(actual, expected.reshape_as(x), rtol=1e-5, atol=1e-6):
                    return False
                case += 1
    return True


if __name__ == "__main__":
    exit(0 if test() else 1)
