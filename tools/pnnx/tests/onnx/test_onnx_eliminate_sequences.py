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


def test_model(nodes, outputs, initializers, opset, case_name, functions=None, two_shapes=False, external=False, nonzero=False):
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
    if external:
        if os.path.exists(name + ".onnx.data"):
            os.remove(name + ".onnx.data")
        onnx.save_model(model, name + ".onnx", save_as_external_data=True, all_tensors_to_one_file=True, location=name + ".onnx.data", size_threshold=0)
    else:
        onnx.save(model, name + ".onnx")

    reference_model = onnx.load(name + ".onnx")
    if external:
        onnx.external_data_helper.convert_model_from_external_data(reference_model)
    session = ort.InferenceSession(reference_model.SerializeToString(), providers=["CPUExecutionProvider"])
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
        x = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + int(nonzero)
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
            if expected_output.dtype != actual_output.dtype or expected_output.shape != actual_output.shape:
                return False
            if np.issubdtype(expected_output.dtype, np.floating) and expected_output.dtype != np.float64:
                equal = np.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5)
            else:
                equal = np.array_equal(expected_output, actual_output)
            if not equal:
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
    for case, casts, denominator in (("positive", (), 2), ("negative", (), -2),
                                      ("dynamic", (TensorProto.INT32,), None),
                                      ("roundtrip", (TensorProto.FLOAT, TensorProto.INT64), 2),
                                      ("shape", (TensorProto.FLOAT,), 2)):
        nodes = [helper.make_node("SequenceConstruct", ["x", "y", "x"], ["seq"]),
                 helper.make_node("ConcatFromSequence", ["seq"], ["cat"], axis=1),
                 helper.make_node("SequenceLength", ["seq"], ["length"])]
        value = "length"
        for i, dtype in enumerate(casts):
            cast_output = "cast_%d" % i
            nodes.append(helper.make_node("Cast", [value], [cast_output], to=dtype))
            value = cast_output
        inputs = [make_tensor_value_info("x", [2, 3]), make_tensor_value_info("y", [2, 3])]
        dtype = casts[-1] if casts else TensorProto.INT64
        shape = []
        if case == "shape":
            nodes.append(helper.make_node("Shape", [value], ["shape"]))
            value, dtype, shape = "shape", TensorProto.INT64, [0]
        nodes.append(helper.make_node("Identity", [value], ["identity"]))
        if denominator is None:
            inputs.append(helper.make_tensor_value_info("denominator", dtype, []))
        else:
            nodes.append(helper.make_node("Constant", [], ["denominator"],
                                          value=helper.make_tensor("divisor", dtype, [], [denominator])))
        nodes.append(helper.make_node("Div", ["identity", "denominator"], ["out"]))
        model = helper.make_model(helper.make_graph(nodes, "integer_div_guard", inputs,
                                  [make_tensor_value_info("cat", [2, 9]), helper.make_tensor_value_info("out", dtype, shape)]),
                                  opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        if runner:
            name = "test_onnx_eliminate_sequences_integer_div_" + case
            onnx.save(model, name + ".onnx")
            subprocess.check_call([runner, name + ".onnx", name + ".out.onnx"])
            transformed = onnx.load(name + ".out.onnx")
            if transformed.SerializeToString() != model.SerializeToString():
                return False
    return True


def test():
    if not test_shared_sequence_regions():
        return False
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
    nodes = [helper.make_node("SequenceConstruct", ["x", "y", "x"], ["seq"]),
             helper.make_node("SequenceLength", ["seq"], ["length"]),
             helper.make_node("Cast", ["length"], ["double_length"], to=TensorProto.DOUBLE),
             helper.make_node("Cast", ["x"], ["double_x"], to=TensorProto.DOUBLE)]
    outputs = []
    for op_type in ("Add", "Sub", "Mul", "Div", "Equal", "Greater", "Less"):
        output = "double_" + op_type
        nodes.append(helper.make_node(op_type, ["double_x", "double_length"], [output]))
        outputs.append(helper.make_tensor_value_info(output, TensorProto.BOOL if op_type in ("Equal", "Greater", "Less") else TensorProto.DOUBLE, [2, 3]))
    for op_type in ("Add", "Greater"):
        output = "large_" + op_type
        nodes.append(helper.make_node(op_type, ["length", "large"], [output]))
        outputs.append(helper.make_tensor_value_info(output, TensorProto.BOOL if op_type == "Greater" else TensorProto.INT64, []))
    nodes.append(helper.make_node("Cast", ["length"], ["float_length"], to=TensorProto.FLOAT))
    constants = [make_integer_tensor("large", 9007199254740993),
                 make_integer_tensor("negative_two", -2),
                 numpy_helper.from_array(np.array(2, dtype=np.float32), "float_two"),
                 make_integer_tensor("scalar_index", 0)]
    for op_type in ("Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"):
        for reverse in (False, True):
            output = "%s_%d" % (op_type, reverse)
            inputs = ["negative_two", "length"] if reverse else ["length", "negative_two"]
            nodes.append(helper.make_node(op_type, inputs, [output]))
            outputs.append(helper.make_tensor_value_info(output, TensorProto.BOOL, []))
    for op_type in ("Relu", "Sqrt", "Gather", "ReduceSum"):
        value = "float_length"
        if op_type == "Gather":
            nodes.append(helper.make_node("Unsqueeze", [value, "axes"], ["vector_length"]))
            constants.append(make_integer_tensor("axes", [0]))
            inputs = ["vector_length", "scalar_index"]
        else:
            inputs = [value]
        value, output = op_type + "_value", op_type + "_div"
        nodes.extend([helper.make_node(op_type, inputs, [value], **({"axis": 0} if op_type == "Gather" else {})),
                      helper.make_node("Div", [value, "float_two"], [output])])
        outputs.append(make_tensor_value_info(output, []))
    if not test_model(nodes, outputs, constants, 13, "precise_length_consumers"):
        return False

    for outputs_order in (("length",), ("length", "intermediate"), ("intermediate", "length")):
        nodes = [helper.make_node("Add", ["x", "y"], ["intermediate"]),
                 helper.make_node("SplitToSequence", ["intermediate"], ["seq"], axis=1),
                 helper.make_node("SequenceLength", ["seq"], ["length"])]
        outputs = [helper.make_tensor_value_info(name, TensorProto.INT64, []) if name == "length"
                   else make_tensor_value_info(name, [2, 3]) for name in outputs_order]
        if not test_model(nodes, outputs, [], 13, "length_" + "_".join(outputs_order)):
            return False
    nodes = [helper.make_node("Add", ["x", "y"], ["sum"]),
             helper.make_node("Constant", [], ["constant"], value=make_integer_tensor("", 7)),
             helper.make_node("Identity", ["x"], ["identity"])]
    initializers = [make_integer_tensor("initializer", 7)]
    outputs = [helper.make_tensor_value_info(name, TensorProto.INT64, []) for name in ("constant", "initializer")]
    outputs.extend(make_tensor_value_info(name, [2, 3]) for name in ("x", "identity", "sum", "sum"))
    for reverse in (False, True):
        if not test_model(nodes, outputs[::-1] if reverse else outputs, initializers, 13, "output_order_%d" % reverse, two_shapes=True):
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


def test_review_regressions():
    for kind in ("initializer", "constant"):
        value = numpy_helper.from_array(np.array([7, 8], dtype=np.int64), "value")
        nodes = [helper.make_node("Add", ["x", "y"], ["sum"])]
        initializers = [value] if kind == "initializer" else []
        if kind == "constant":
            nodes.append(helper.make_node("Constant", [], ["value"], value=value))
        nodes.extend([helper.make_node("SequenceConstruct", ["value", "value"], ["seq"]),
                      helper.make_node("ConcatFromSequence", ["seq"], ["cat"], axis=0, new_axis=0),
                      helper.make_node("ConcatFromSequence", ["seq"], ["stack"], axis=0, new_axis=1)])
        outputs = [make_tensor_value_info("sum", [2, 3]),
                   helper.make_tensor_value_info("cat", TensorProto.INT64, [4]),
                   helper.make_tensor_value_info("stack", TensorProto.INT64, [2, 2])]
        if not test_model(nodes, outputs, initializers, 13, kind + "_integer_cat_stack"):
            return False

    for shape in ([3, 2], [6]):
        value = make_integer_tensor("value", shape)
        nodes = [helper.make_node("Add", ["x", "y"], ["sum"]),
                 helper.make_node("Reshape", ["sum", "value"], ["reshaped"]),
                 helper.make_node("SequenceConstruct", ["value", "value"], ["seq"]),
                 helper.make_node("ConcatFromSequence", ["seq"], ["out"], axis=0, new_axis=0)]
        outputs = [make_tensor_value_info("reshaped", shape),
                   helper.make_tensor_value_info("out", TensorProto.INT64, [2 * len(shape)])]
        if not test_model(nodes, outputs, [value], 13, "shared_initializer_shape%d" % len(shape)):
            return False

    for kind in ("initializer", "constant"):
        nodes = [helper.make_node("Add", ["x", "y"], ["sum"]),
                 helper.make_node("SequenceConstruct", ["x", "y"], ["length_seq"]),
                 helper.make_node("SequenceLength", ["length_seq"], ["length"])]
        initializers = [make_integer_tensor("index", 0)]
        outputs = [make_tensor_value_info("sum", [2, 3])]
        for shape in ([], [1], [1, 1]):
            suffix = str(len(shape))
            name = "value" + suffix
            value = numpy_helper.from_array(np.full(shape, 7, dtype=np.int64), name)
            if kind == "initializer":
                initializers.append(value)
            else:
                nodes.append(helper.make_node("Constant", [], [name], value=value))
            nodes.extend([helper.make_node("SequenceConstruct", [name], ["seq" + suffix]),
                          helper.make_node("SequenceAt", ["seq" + suffix, "index"], ["out" + suffix])])
            outputs.append(helper.make_tensor_value_info("out" + suffix, TensorProto.INT64, shape))
            for op_type in ("Add", "Less", "Greater"):
                output = op_type.lower() + suffix
                nodes.append(helper.make_node(op_type, ["length", name], [output]))
                outputs.append(helper.make_tensor_value_info(output, TensorProto.INT64 if op_type == "Add" else TensorProto.BOOL, shape))
        if not test_model(nodes, outputs, initializers, 13, kind + "_ranks"):
            return False

        value = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "value")
        nodes = [helper.make_node("Add", ["x", "y"], ["sum"])]
        initializers = [make_integer_tensor("index", 1)]
        if kind == "initializer":
            initializers.append(value)
        else:
            nodes.append(helper.make_node("Constant", [], ["value"], value=value))
        nodes.extend([helper.make_node("SplitToSequence", ["value"], ["seq"], axis=0),
                      helper.make_node("SequenceAt", ["seq", "index"], ["out"])])
        if not test_model(nodes, [make_tensor_value_info("sum", [2, 3]), make_tensor_value_info("out", [1, 3])], initializers, 13, kind + "_split_no_value_info"):
            return False

    nodes = [helper.make_node("Add", ["x", "y"], ["sum"]),
             helper.make_node("SplitToSequence", ["value", "sizes"], ["seq"], axis=0),
             helper.make_node("SequenceAt", ["seq", "index"], ["out"])]
    initializers = [numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "value"),
                    make_integer_tensor("sizes", [2]), make_integer_tensor("index", 0)]
    if not test_model(nodes, [make_tensor_value_info("sum", [2, 3]), make_tensor_value_info("out", [2, 3])], initializers, 13, "initializer_single_split"):
        return False

    nodes = [helper.make_node("Split", ["x", "single_sizes"], ["single"], axis=1),
             helper.make_node("Split", ["y", "pair_sizes"], ["first", "last"], axis=1)]
    initializers = [make_integer_tensor("single_sizes", [3]), make_integer_tensor("pair_sizes", [1, 2])]
    outputs = [make_tensor_value_info("single", [2, 3]), make_tensor_value_info("first", [2, 1]), make_tensor_value_info("last", [2, 2])]
    if not test_model(nodes, outputs, initializers, 13, "external_native_split", external=True):
        return False

    for split in (None, 1, [1, 2]):
        nodes = [helper.make_node("SplitToSequence", ["x"] + (["sizes"] if split is not None else []), ["seq"], axis=1),
                 helper.make_node("SequenceAt", ["seq", "index"], ["out"]),
                 helper.make_node("Add", ["x", "y"], ["sum"])]
        initializers = [make_integer_tensor("index", 1)]
        if split is not None:
            initializers.append(make_integer_tensor("sizes", split))
        width = 2 if isinstance(split, list) else 1
        if not test_model(nodes, [make_tensor_value_info("out", [2, width]), make_tensor_value_info("sum", [2, 3])], initializers, 13, "external_%s" % str(split), external=True):
            return False
    return True


if __name__ == "__main__":
    if test() and test_review_regressions():
        exit(0)
    else:
        exit(1)
