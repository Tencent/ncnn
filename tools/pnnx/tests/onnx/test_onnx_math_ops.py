# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset11 as op

@script()
def Model(x: FLOAT[2,3,4], y: FLOAT[2,3,4]):
    return (op.Abs(x),
        op.Acos(x),
        op.Asin(x),
        op.Atan(x),
        op.Cos(x),
        op.Exp(x),
        op.Floor(x),
        op.Log(x),
        op.Neg(x),
        op.Reciprocal(x),
        op.Relu(x),
        op.Sigmoid(x),
        op.Sin(x),
        op.Sqrt(x),
        op.Tan(x),
        op.Tanh(x),
        op.Add(x, y),
        op.Sub(x, y),
        op.Mul(x, y),
        op.Div(x, y),
        op.Min(x, y),
        op.Max(x, y),
        op.Pow(x, op.Abs(y)),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_math_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4)
    y = torch.rand(2, 3, 4)

    # ort inference
    sess = ort.InferenceSession("test_onnx_math_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy()}))

    # onnx to ncnn
    import os
    os.system("../../src/pnnx test_onnx_math_ops.onnx")

    # ncnn inference
    import test_onnx_math_ops_ncnn
    b = test_onnx_math_ops_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
