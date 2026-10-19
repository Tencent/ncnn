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


def test_model(nodes, outputs, initializers, opset, case_name, two_shapes=False, value_info=(), preserve_sequence=False):
    input_shape = [None, 3] if two_shapes else [2, 3]
    graph = helper.make_graph(nodes, "sequence", [make_tensor_value_info("x", input_shape), make_tensor_value_info("y", input_shape)], outputs, initializers, value_info=value_info)
    if two_shapes:
        for output in graph.output:
            if output.type.HasField("tensor_type") and output.type.tensor_type.shape.dim:
                output.type.tensor_type.shape.dim[0].ClearField("dim_value")
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    name = "test_onnx_eliminate_sequences_%d_%s" % (opset, case_name.replace("-", "negative"))
    onnx.save(model, name + ".onnx")

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    command = ["../../src/pnnx", name + ".onnx", "inputshape=[2,3],[2,3]"]
    if two_shapes:
        command.append("inputshape2=[4,3],[4,3]")
    if preserve_sequence:
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or "ort CreateSession failed" in result.stderr or "ort Run failed" in result.stderr or "missing runtime output" in result.stderr:
            print(result.stderr)
            return False
        transformed = onnx.load(name + ".pnnxsim.onnx")
        sequence_op = "SplitToSequence" if preserve_sequence is True else preserve_sequence
        return any(node.op_type == sequence_op for node in transformed.graph.node) and not any(node.op_type in ("Split", "Concat") for node in transformed.graph.node)
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
        if not isinstance(actual, (tuple, list)):
            actual = [actual]
        if len(expected) != len(actual):
            return False
        for expected_output, actual_output in zip(expected, actual):
            expected_output = np.asarray(expected_output)
            actual_output = actual_output.numpy() if isinstance(actual_output, torch.Tensor) else np.asarray(actual_output)
            if expected_output.dtype != actual_output.dtype or expected_output.shape != actual_output.shape:
                return False
            if np.issubdtype(expected_output.dtype, np.floating) and expected_output.dtype != np.float64:
                equal = np.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5)
            else:
                equal = np.array_equal(expected_output, actual_output)
            if not equal:
                return False
    return True



def test():
    # Opset 13 changes Split and squeeze/unsqueeze axes from attributes to inputs.
    for opset in (11, 13):
        nodes = []
        outputs = []
        constants = []
        for use_raw_data in (False, True):
            suffix = "raw" if use_raw_data else "typed"
            index_name, seq, at, cat = [name + "_" + suffix for name in ("index", "seq", "at", "cat")]
            index = make_integer_tensor(index_name, -1)
            if not use_raw_data:
                index.ClearField("raw_data")
                index.int64_data.append(-1)
            constants.append(index)
            nodes.extend([helper.make_node("SequenceConstruct", ["x", "y"], [seq]),
                          helper.make_node("SequenceAt", [seq, index_name], [at]),
                          helper.make_node("ConcatFromSequence", [seq], [cat], axis=1)])
            outputs.extend([make_tensor_value_info(at, [2, 3]), make_tensor_value_info(cat, [2, 6])])
        nodes.append(helper.make_node("SequenceConstruct", ["x", "y"], ["stack_seq"]))
        for axis, shape in ((0, [2, 2, 3]), (1, [2, 2, 3]), (-1, [2, 3, 2])):
            name = "stack_" + str(axis).replace("-", "negative")
            nodes.append(helper.make_node("ConcatFromSequence", ["stack_seq"], [name], axis=axis, new_axis=1))
            outputs.append(make_tensor_value_info(name, shape))
        nodes.extend([helper.make_node("SequenceConstruct", ["x", "y"], ["alias_seq"], name="pnnx_seq_0"),
                      helper.make_node("Constant", [], ["alias_index"], value=make_integer_tensor("", 0)),
                      helper.make_node("SequenceAt", ["alias_seq", "alias_index"], ["first"]),
                      helper.make_node("SequenceAt", ["alias_seq", "alias_index"], ["second"])])
        outputs.extend([make_tensor_value_info("first", [2, 3]), make_tensor_value_info("second", [2, 3])])
        constants.append(make_integer_tensor("index", -1))
        for case_name, sizes, keepdims, shape in (("unbind", None, 0, [2]), ("unit", None, 1, [2, 1]),
                                                 ("chunk", 2, 1, [2, 1]), ("sizes", [1, 2], 1, [2, 2])):
            inputs = ["x"]
            if sizes is not None:
                inputs.append("sizes_" + case_name)
                constants.append(make_integer_tensor(inputs[-1], sizes))
            seq, out = "seq_" + case_name, "out_" + case_name
            nodes.extend([helper.make_node("SplitToSequence", inputs, [seq], axis=1, keepdims=keepdims),
                          helper.make_node("SequenceAt", [seq, "index"], [out])])
            outputs.append(make_tensor_value_info(out, shape))
        if not test_model(nodes, outputs, constants, opset, "sequence_variants"):
            return False
    nodes = [helper.make_node("SplitToSequence", ["x", "sizes"], ["seq"], axis=1),
             helper.make_node("SequenceAt", ["seq", "index"], ["out"])]
    if not test_model(nodes, [make_tensor_value_info("out", [2, 3])],
                      [make_integer_tensor("sizes", 3), make_integer_tensor("index", 0)], 13, "single_chunk"):
        return False
    for keepdims in (0, 1):
        axis = 0
        nodes = [helper.make_node("Add", ["x", "y"], ["intermediate"]),
                 helper.make_node("SplitToSequence", ["intermediate"], ["seq"], axis=axis, keepdims=keepdims),
                 helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=axis, new_axis=1 - keepdims)]
        # A stale concrete intermediate shape must not specialize the dynamic split.
        if not test_model(nodes, [make_tensor_value_info("out", [2, 3])], [], 13, "inferred_%d" % keepdims, two_shapes=bool(keepdims),
                          value_info=[make_tensor_value_info("intermediate", [2, 3])] if keepdims else (), preserve_sequence=bool(keepdims)):
            return False

    nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"]),
             helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=1),
             helper.make_node("SequenceLength", ["seq"], ["length"])]
    if not test_model(nodes, [make_tensor_value_info("out", [2, 6]), helper.make_tensor_value_info("length", TensorProto.INT64, [])],
                      [], 13, "unsupported_consumer", preserve_sequence="SequenceConstruct"):
        return False

    constants = [numpy_helper.from_array(np.array([1], dtype=np.float32), "a"),
                 numpy_helper.from_array(np.array([2], dtype=np.float32), "b"), make_integer_tensor("index", 0)]
    nodes = [helper.make_node("Add", ["a", "b"], ["c0"]),
             helper.make_node("Add", ["c0", "a"], ["c"]),
             helper.make_node("Add", ["x", "y"], ["sum"]),
             helper.make_node("Add", ["sum", "c"], ["out"]),
             helper.make_node("SequenceConstruct", ["c", "x"], ["seq"]),
             helper.make_node("SequenceAt", ["seq", "index"], ["selected"])]
    if not test_model(nodes, [make_tensor_value_info("out", [2, 3]), make_tensor_value_info("selected", [1])],
                      constants, 13, "constant_expression", preserve_sequence="SequenceConstruct"):
        return False

    branch = helper.make_graph([helper.make_node("Add", ["x", "x"], ["pnnx_seq_0"])], "branch", [], [make_tensor_value_info("pnnx_seq_0", [2, 3])])
    nodes = [helper.make_node("SequenceConstruct", ["x", "y"], ["seq"]),
             helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=0, new_axis=1),
             helper.make_node("If", ["condition"], ["branch_out"], then_branch=branch, else_branch=branch)]
    if not test_model(nodes, [make_tensor_value_info("branch_out", [2, 3]), make_tensor_value_info("out", [2, 2, 3])], [numpy_helper.from_array(np.array(True), "condition")], 13, "nested_names"):
        return False
    return True


if __name__ == "__main__":
    if not test():
        exit(1)
