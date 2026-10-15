# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import onnx
import numpy as np
from onnx import TensorProto, helper

def test():
    x = torch.tensor([[-4, -2, -1, 0], [0.25, 1, 2, 4]], dtype=torch.float16)
    y = x.float()
    z = x.int()
    w = x.long()
    d = x.double()

    inputs = []
    outputs = []
    nodes = []
    initializers = []
    a = []

    for name, input, data_type, dtype in (
        ("x", x, TensorProto.FLOAT16, np.float16),
        ("y", y, TensorProto.FLOAT, np.float32),
        ("z", z, TensorProto.INT32, np.int32),
        ("w", w, TensorProto.INT64, np.int64),
        ("d", d, TensorProto.DOUBLE, np.float64)
    ):
        inputs.append(helper.make_tensor_value_info(name, data_type, [2, 4]))
        values = [0, 1, -2, 3]
        if data_type in (TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE):
            values += [0.5, -1.5, 1.4142135623730951]
        if data_type == TensorProto.INT64:
            values += [9007199254740993, -9007199254740993]

        for raw in (True, False):
            for dims in ([], [1]):
                for value in values:
                    array = np.array(value, dtype=dtype).reshape(dims)
                    tensor = TensorProto()
                    tensor.name = "constant_%d" % len(initializers)
                    tensor.data_type = data_type
                    tensor.dims.extend(dims)
                    if raw:
                        tensor.raw_data = array.astype(array.dtype.newbyteorder("<")).tobytes()
                    elif data_type == TensorProto.FLOAT16:
                        # float16 is stored as a bit pattern in int32_data
                        tensor.int32_data.append(int(array.view(np.uint16).item()))
                    elif data_type == TensorProto.FLOAT:
                        tensor.float_data.append(float(array.item()))
                    elif data_type == TensorProto.DOUBLE:
                        tensor.double_data.append(float(array.item()))
                    elif data_type == TensorProto.INT32:
                        tensor.int32_data.append(int(array.item()))
                    else:
                        tensor.int64_data.append(int(array.item()))
                    initializers.append(tensor)

                    # keep a runtime input so the scalar is not folded away
                    output = "output_%d" % len(outputs)
                    nodes.append(helper.make_node("Add", [name, tensor.name], [output]))
                    outputs.append(helper.make_tensor_value_info(output, data_type, [2, 4]))
                    a.append(input + array.item())

    for name, input, data_type in (("y", y, TensorProto.FLOAT), ("z", z, TensorProto.INT32), ("w", w, TensorProto.INT64)):
        for dims in ([], [1]):
            constant = "operand_%s_%d" % (name, len(dims))
            initializers.append(helper.make_tensor(constant, data_type, dims, [2]))
            for reverse in (False, True):
                operands = [constant, name] if reverse else [name, constant]
                lhs, rhs = (torch.tensor(2, dtype=input.dtype), input) if reverse else (input, 2)
                for op_type, operation in (("Equal", torch.eq), ("Greater", torch.gt), ("Less", torch.lt), ("GreaterOrEqual", torch.ge), ("LessOrEqual", torch.le), ("Div", torch.div)):
                    if op_type == "Div" and reverse:
                        # Avoid undefined integer division by zero without changing the other probes.
                        denominator = constant + "_nonzero"
                        nodes.append(helper.make_node("Mul", [name, name], [denominator + "_square"]))
                        nodes.append(helper.make_node("Add", [denominator + "_square", constant], [denominator]))
                        operands = [constant, denominator]
                        lhs, rhs = torch.tensor(2, dtype=input.dtype), input * input + 2
                    output = "output_%d" % len(outputs)
                    nodes.append(helper.make_node(op_type, operands, [output]))
                    outputs.append(helper.make_tensor_value_info(output, data_type if op_type == "Div" else TensorProto.BOOL, [2, 4]))
                    if op_type == "Div" and data_type in (TensorProto.INT32, TensorProto.INT64):
                        a.append(torch.div(lhs, rhs, rounding_mode="trunc"))
                    else:
                        a.append(operation(lhs, rhs))

    indices = torch.tensor([0], dtype=torch.int64)
    inputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, [1]))
    external_data = bytearray()
    # Exercise external scalar imports and the existing Gather Attribute path.
    for dims, op_type in (([], "Add"), ([1], "Add"), ([1, 1], "Gather")):
        for value in (2, 9007199254740993):
            name = "external_%d" % len(initializers)
            tensor = onnx.numpy_helper.from_array(np.array(value, dtype=np.int64).reshape(dims), name)
            onnx.external_data_helper.set_external_data(tensor, "test_onnx_scalar_constants.onnx.data", offset=len(external_data), length=8)
            external_data.extend(tensor.raw_data)
            tensor.ClearField("raw_data")
            initializers.append(tensor)
            output = "output_%d" % len(outputs)
            if op_type == "Gather":
                nodes.append(helper.make_node(op_type, [name, "indices"], [output]))
                outputs.append(helper.make_tensor_value_info(output, TensorProto.INT64, [1, 1]))
                a.append(torch.tensor([[value]], dtype=torch.int64))
            else:
                nodes.append(helper.make_node(op_type, ["w", name], [output]))
                outputs.append(helper.make_tensor_value_info(output, TensorProto.INT64, [2, 4]))
                a.append(w + value)

    # save onnx
    graph = helper.make_graph(nodes, "scalar_constants", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    with open("test_onnx_scalar_constants.onnx.data", "wb") as f:
        f.write(external_data)
    onnx.save(model, "test_onnx_scalar_constants.onnx")
    onnx.checker.check_model("test_onnx_scalar_constants.onnx")

    # onnx to pnnx
    import os
    ret = os.system("../../src/pnnx test_onnx_scalar_constants.onnx inputshape=[2,4]f16,[2,4]f32,[2,4]i32,[2,4]i64,[2,4]f64,[1]i64")
    if ret != 0:
        return False

    # pnnx inference
    import test_onnx_scalar_constants_pnnx
    net = test_onnx_scalar_constants_pnnx.Model()
    net.eval()
    b = net(x, y, z, w, d, indices)

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if a0.dtype != b0.dtype or a0.shape != b0.shape or not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
