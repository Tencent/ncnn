# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
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



def make_shared_sequence_cases():
    # aten_ calls survive inline_containers, exercising this pass in the loader.
    def stack_function(name):
        return helper.make_function("local", name, ["s"], ["out"],
                                    [helper.make_node("ConcatFromSequence", ["s"], ["out"], axis=0, new_axis=1)],
                                    [helper.make_opsetid("", 13)])

    stack = stack_function("aten_sequence_stack")
    other = stack_function("aten_sequence_other")
    pair = helper.make_function("local", "aten_sequence_pair", ["s", "t"], ["a", "b"],
                                [helper.make_node("ConcatFromSequence", ["s"], ["a"], axis=0, new_axis=1),
                                 helper.make_node("ConcatFromSequence", ["t"], ["b"], axis=0, new_axis=1)],
                                [helper.make_opsetid("", 13)])
    alias = helper.make_function("local", "aten_sequence_alias", ["s", "t"], ["a", "b"],
                                 [helper.make_node("ConcatFromSequence", ["s"], ["a"], axis=0, new_axis=1),
                                  helper.make_node("Identity", ["a"], ["b"])],
                                 [helper.make_opsetid("", 13)])
    bad = helper.make_function("local", "aten_sequence_bad", ["s"], ["out"],
                               [helper.make_node("ConcatFromSequence", ["s"], ["cat"], axis=0, new_axis=1),
                                helper.make_node("Neg", ["cat"], ["out"])],
                               [helper.make_opsetid("", 13)])
    functions = [stack, other, pair, alias, bad]

    def call(function, inputs, outputs):
        return helper.make_node(function, inputs, outputs, domain="local")

    producer = helper.make_node("SequenceConstruct", ["x", "y"], ["s"], name="pnnx_seq_0")
    second = helper.make_node("SequenceConstruct", ["y", "x"], ["t"])
    for same in (False, True):
        for reverse in (False, True):
            calls = [call(stack.name, ["s"], ["a"]), call(stack.name if same else other.name, ["s"], ["b"])]
            yield "shared_%d_%d" % (same, reverse), [producer] + (calls[::-1] if reverse else calls), ["a", "b"], functions, True
    calls = [call(stack.name, ["s"], ["a"]), call(pair.name, ["s", "t"], ["b", "c"]), call(other.name, ["t"], ["d"])]
    for reverse in (False, True):
        yield "transitive_%d" % reverse, [producer, second] + (calls[::-1] if reverse else calls), ["a", "b", "c", "d"], functions, True
    yield "native_local", [producer, call(stack.name, ["s"], ["a"]), helper.make_node("ConcatFromSequence", ["s"], ["b"], axis=0, new_axis=1)], ["a", "b"], functions, True
    yield "repeated_alias", [producer, call(alias.name, ["s", "s"], ["a", "b"]), call(stack.name, ["s"], ["c"])], ["a", "b", "c"], functions, True
    blocked = [producer, call(stack.name, ["s"], ["a"]), call(bad.name, ["s"], ["b"])]
    yield "unsupported_sibling", blocked, ["a", "b"], functions, False
    yield "unsupported_native", [producer, call(stack.name, ["s"], ["a"]), helper.make_node("SequenceInsert", ["s", "x"], ["extended_s"])], ["a"], functions, False
    yield "dynamic_sibling", [producer, call(stack.name, ["s"], ["a"]), helper.make_node("SequenceAt", ["s", "dynamic_index"], ["b"])], ["a", "b"], functions, False
    yield "transitive_blocked", [producer, second, call(stack.name, ["s"], ["a"]), call(pair.name, ["s", "t"], ["b", "c"]), call(bad.name, ["t"], ["d"])], ["a", "b", "c", "d"], functions, False
    yield "independent_island", blocked + [second, call(stack.name, ["t"], ["c"]), call(other.name, ["t"], ["d"])], ["a", "b", "c", "d"], functions, False
    for guard in ("domain", "name", "overload", "opset", "call_attribute", "function_attribute", "nested_body"):
        guarded_function = stack_function("aten_sequence_guarded")
        guarded_call = call(guarded_function.name, ["s"], ["b"])
        if guard == "domain":
            guarded_function.domain = "other"
        elif guard == "name":
            guarded_function.name = "aten_sequence_missing"
        elif guard == "overload":
            guarded_function.overload = "other"
        elif guard == "opset":
            guarded_function.opset_import[0].version = 11
        elif guard == "call_attribute":
            guarded_call.attribute.append(helper.make_attribute("unused", 1))
        elif guard == "function_attribute":
            guarded_function.attribute.append("unused")
            guarded_call.attribute.append(helper.make_attribute("unused", 1))
        else:
            guarded_function.node[0].domain = "other"
            guarded_function.opset_import.append(helper.make_opsetid("other", 1))
        yield "guard_" + guard, [producer, call(stack.name, ["s"], ["a"]), guarded_call], ["a", "b"], [stack, guarded_function], False


def test_shared_sequence_regions():
    # Optional standalone pass runner validates atomic fallback without asking
    # the downstream converter to support deliberately unsupported operators.
    runner = os.environ.get("PNNX_ONNX_SEQUENCE_PASS_RUNNER")
    for case_name, nodes, output_names, functions, lowerable in make_shared_sequence_cases():
        outputs = [make_tensor_value_info(name, [2, 3] if case_name == "dynamic_sibling" and name == "b" else [2, 2, 3]) for name in output_names]
        if lowerable and not test_model(nodes, outputs, [], 13, case_name, functions):
            return False
        if not runner:
            continue
        inputs = [make_tensor_value_info("x", [2, 3]), make_tensor_value_info("y", [2, 3])]
        if case_name == "dynamic_sibling":
            inputs.append(helper.make_tensor_value_info("dynamic_index", TensorProto.INT64, []))
        model = helper.make_model(helper.make_graph(nodes, case_name, inputs, outputs), functions=functions,
                                  opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("local", 1)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        name = "test_onnx_eliminate_sequences_pass_" + case_name
        onnx.save(model, name + ".onnx")
        subprocess.check_call([runner, name + ".onnx", name + ".out.onnx"])
        transformed = onnx.load(name + ".out.onnx")
        onnx.checker.check_model(transformed)
        if lowerable:
            if any(node.domain == "local" or "Sequence" in node.op_type for node in transformed.graph.node):
                return False
        elif case_name == "independent_island":
            if list(transformed.graph.node[:3]) != list(model.graph.node[:3]):
                return False
            if any("t" in node.input or "t" in node.output for node in transformed.graph.node):
                return False
        elif transformed.SerializeToString() != model.SerializeToString():
            return False
        subprocess.check_call([runner, name + ".out.onnx", name + ".twice.onnx"])
        if onnx.load(name + ".twice.onnx").SerializeToString() != transformed.SerializeToString():
            return False
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        feeds = {"x": x, "y": x + 20}
        if case_name == "dynamic_sibling":
            feeds["dynamic_index"] = np.array(-1, dtype=np.int64)
        if case_name.startswith("guard_"):
            continue
        original_outputs = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)
        transformed_outputs = ort.InferenceSession(transformed.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)
        for expected, actual in zip(original_outputs, transformed_outputs):
            np.testing.assert_array_equal(expected, actual)
    return True


def test():
    if not test_shared_sequence_regions():
        return False
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
