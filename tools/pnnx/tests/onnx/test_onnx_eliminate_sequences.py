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


def test_model(nodes, outputs, initializers, opset, case_name, functions=None, two_shapes=False):
    input_shape = [None, 3] if two_shapes else [2, 3]
    graph = helper.make_graph(nodes, "sequence", [make_tensor_value_info("x", input_shape), make_tensor_value_info("y", input_shape)], outputs, initializers)
    if two_shapes:
        for output in graph.output:
            if output.type.HasField("tensor_type") and output.type.tensor_type.shape.dim:
                output.type.tensor_type.shape.dim[0].ClearField("dim_value")
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)] + ([helper.make_opsetid("local", 1)] if functions else []), functions=functions or [])
    model.ir_version = 8
    onnx.checker.check_model(model)
    name = "test_onnx_eliminate_sequences_%d_%s" % (opset, case_name.replace("-", "negative"))
    onnx.save(model, name + ".onnx")

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    command = ["../../src/pnnx", name + ".onnx", "inputshape=[2,3],[2,3]"]
    if two_shapes:
        command.append("inputshape2=[4,3],[4,3]")
    if subprocess.call(command) != 0:
        return False
    with open(name + ".pnnx.param") as param_file:
        if "Sequence" in param_file.read():
            return False
    spec = importlib.util.spec_from_file_location(name + "_pnnx", name + "_pnnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.Model().eval()
    for rows in ((2, 4) if two_shapes else (2,)):
        x = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        y = x + 20
        expected = session.run(None, {"x": x.numpy(), "y": y.numpy()})
        with torch.no_grad():
            actual = net(x, y)
        if len(outputs) == 1 and isinstance(expected[0], list):
            actual = [actual]
        elif not isinstance(actual, (tuple, list)):
            actual = [actual]
        if len(expected) != len(actual):
            return False
        for expected_output, actual_output in zip(expected, actual):
            expected_output = np.asarray(expected_output)
            actual_output = actual_output.numpy() if isinstance(actual_output, torch.Tensor) else np.asarray(actual_output)
            if expected_output.dtype != actual_output.dtype or expected_output.shape != actual_output.shape or not np.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5):
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
    for outputs_order in (("length",), ("length", "intermediate"), ("intermediate", "length")):
        nodes = [helper.make_node("Add", ["x", "y"], ["intermediate"]),
                 helper.make_node("SplitToSequence", ["intermediate"], ["seq"], axis=1),
                 helper.make_node("SequenceLength", ["seq"], ["length"])]
        outputs = [helper.make_tensor_value_info(name, TensorProto.INT64, []) if name == "length"
                   else make_tensor_value_info(name, [2, 3]) for name in outputs_order]
        if not test_model(nodes, outputs, [], 13, "length_" + "_".join(outputs_order)):
            return False
    for output_kind in ("constant", "initializer", "input", "identity"):
        nodes = [helper.make_node("Add", ["x", "y"], ["sum"])]
        initializers = []
        output_name = "extra"
        if output_kind == "constant":
            nodes.append(helper.make_node("Constant", [], ["extra"], value=make_integer_tensor("", 7)))
        elif output_kind == "initializer":
            initializers.append(make_integer_tensor("extra", 7))
        elif output_kind == "input":
            output_name = "x"
        else:
            nodes.append(helper.make_node("Identity", ["x"], ["extra"]))
        extra = helper.make_tensor_value_info(output_name, TensorProto.INT64, []) if output_kind in ("constant", "initializer") else make_tensor_value_info(output_name, [2, 3])
        for reverse in (False, True):
            outputs = [extra, make_tensor_value_info("sum", [2, 3])]
            if reverse:
                outputs.reverse()
            if not test_model(nodes, outputs, initializers, 13, "output_%s_%d" % (output_kind, reverse), two_shapes=True):
                return False
    nodes = [helper.make_node("Add", ["x", "y"], ["sum"])]
    if not test_model(nodes, [make_tensor_value_info("sum", [2, 3]), make_tensor_value_info("sum", [2, 3])], [], 13, "duplicate_outputs", two_shapes=True):
        return False
    nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"])]
    output = helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, [2, 3])
    if not test_model(nodes, [output], [], 13, "sequence_output", two_shapes=True):
        return False
    for keepdims in (0, 1):
        nodes = [helper.make_node("Add", ["x", "y"], ["intermediate"]),
                 helper.make_node("SplitToSequence", ["intermediate"], ["seq"], axis=1, keepdims=keepdims),
                 helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=1, new_axis=1 - keepdims)]
        if not test_model(nodes, [make_tensor_value_info("out", [2, 3])], [], 13, "inferred_%d" % keepdims):
            return False

    function = helper.make_function("local", "stack_pair", ["a", "b"], ["out_a", "out_b"],
                                    [helper.make_node("ConcatFromSequence", ["a"], ["out_a"], axis=0, new_axis=1),
                                     helper.make_node("ConcatFromSequence", ["b"], ["out_b"], axis=0, new_axis=1)],
                                    [helper.make_opsetid("", 13)])
    for reverse, repeated in ((False, False), (True, False), (False, True)):
        producers = [helper.make_node("SequenceConstruct", ["x", "y"], ["a"]),
                     helper.make_node("SequenceConstruct", ["y", "x"], ["b"])]
        nodes = (producers[::-1] if reverse else producers) + [helper.make_node("stack_pair", ["a", "a" if repeated else "b"], ["out_a", "out_b"], domain="local")]
        if not test_model(nodes, [make_tensor_value_info("out_a", [2, 2, 3]), make_tensor_value_info("out_b", [2, 2, 3])], [], 13, "function_%d_%d" % (reverse, repeated), [function]):
            return False

    branch = helper.make_graph([helper.make_node("Add", ["x", "x"], ["pnnx_seq_0"])], "branch", [], [make_tensor_value_info("pnnx_seq_0", [2, 3])])
    nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"]),
             helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=0, new_axis=1),
             helper.make_node("If", ["condition"], ["branch_out"], then_branch=branch, else_branch=branch)]
    if not test_model(nodes, [make_tensor_value_info("branch_out", [2, 3]), make_tensor_value_info("out", [2, 2, 3])], [numpy_helper.from_array(np.array(True), "condition")], 13, "nested_names"):
        return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
