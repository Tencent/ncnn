# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper


def make_tensor_value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def make_integer_tensor(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def test_model(nodes, outputs, initializers, opset, case_name):
    graph = helper.make_graph(nodes, "sequence", [make_tensor_value_info("x", [2, 3]), make_tensor_value_info("y", [2, 3])], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    name = "test_onnx_eliminate_sequences_%d_%s" % (opset, case_name.replace("-", "negative"))
    onnx.save(model, name + ".onnx")

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = x + 20
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = session.run(None, {"x": x.numpy(), "y": y.numpy()})

    if subprocess.call(["../../src/pnnx", name + ".onnx", "inputshape=[2,3],[2,3]"]) != 0:
        return False
    with open(name + ".pnnx.param") as param_file:
        if "Sequence" in param_file.read():
            return False
    spec = importlib.util.spec_from_file_location(name + "_pnnx", name + "_pnnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.Model().eval()
    with torch.no_grad():
        actual = net(x, y)
    if not isinstance(actual, (tuple, list)):
        actual = [actual]
    if len(expected) != len(actual):
        return False
    for expected_output, actual_output in zip(expected, actual):
        if expected_output.dtype != actual_output.numpy().dtype or not np.allclose(expected_output, actual_output.numpy(), rtol=1e-5, atol=1e-5):
            return False
    return True


def test():
    for opset in (11, 13, 18):
        for use_raw_data in (False, True):
            index = make_integer_tensor("index", -1)
            if not use_raw_data:
                index.ClearField("raw_data")
                index.int64_data.append(-1)
            nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"]),
                     helper.make_node("SequenceAt", ["seq", "index"], ["at"]),
                     helper.make_node("ConcatFromSequence", ["seq"], ["cat"], axis=1)]
            if not test_model(nodes, [make_tensor_value_info("at", [2, 3]), make_tensor_value_info("cat", [2, 6])], [index], opset, "cat_at_%d" % use_raw_data):
                return False
        for axis, shape in ((0, [2, 2, 3]), (1, [2, 2, 3]), (-1, [2, 3, 2])):
            nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"]),
                     helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=axis, new_axis=1)]
            if not test_model(nodes, [make_tensor_value_info("out", shape)], [], opset, "stack_%d" % axis):
                return False
        for case_name, sizes, keepdims, shape in (("unbind", None, 0, [2]), ("unit", None, 1, [2, 1]),
                                                 ("chunk", 2, 1, [2, 1]), ("sizes", [1, 2], 1, [2, 2])):
            inputs = ["x"]
            constants = [make_integer_tensor("index", -1)]
            if sizes is not None:
                inputs.append("sizes")
                constants.append(make_integer_tensor("sizes", sizes))
            nodes = [helper.make_node("SplitToSequence", inputs, ["seq"], axis=1, keepdims=keepdims),
                     helper.make_node("SequenceAt", ["seq", "index"], ["out"])]
            if not test_model(nodes, [make_tensor_value_info("out", shape)], constants, opset, case_name):
                return False
        nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"], name="pnnx_seq_0"),
                 helper.make_node("Constant", [], ["index"], value=make_integer_tensor("", 0)),
                 helper.make_node("SequenceAt", ["seq", "index"], ["first"]),
                 helper.make_node("SequenceAt", ["seq", "index"], ["second"])]
        if not test_model(nodes, [make_tensor_value_info("first", [2, 3]), make_tensor_value_info("second", [2, 3])], [], opset, "aliases"):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
