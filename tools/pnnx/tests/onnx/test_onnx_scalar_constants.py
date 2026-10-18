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

    inputs = []
    outputs = []
    nodes = []
    initializers = []
    a = []

    for name, input, data_type, dtype in (
        ("x", x, TensorProto.FLOAT16, np.float16),
        ("y", y, TensorProto.FLOAT, np.float32),
        ("z", z, TensorProto.INT32, np.int32),
        ("w", w, TensorProto.INT64, np.int64)
    ):
        inputs.append(helper.make_tensor_value_info(name, data_type, [2, 4]))
        values = [0, 1, -2, 3]
        if data_type in (TensorProto.FLOAT16, TensorProto.FLOAT):
            values += [0.5, -1.5, 1.4142135623730951]

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

    # save onnx
    graph = helper.make_graph(nodes, "scalar_constants", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, "test_onnx_scalar_constants.onnx")

    # onnx to pnnx
    import os
    ret = os.system("../../src/pnnx test_onnx_scalar_constants.onnx inputshape=[2,4]f16,[2,4]f32,[2,4]i32,[2,4]i64")
    if ret != 0:
        return False

    # pnnx inference
    import test_onnx_scalar_constants_pnnx
    net = test_onnx_scalar_constants_pnnx.Model()
    net.eval()
    b = net(x, y, z, w)

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if a0.dtype != b0.dtype or not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
