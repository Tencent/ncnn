# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset11 as op

@script()
def Model(x: FLOAT["C","H","W"], y: FLOAT["C","H","W"]):
    return (op.Abs(x),
        op.Acos(x),
        # op.Acosh(x),
        op.Asin(x),
        # op.Asinh(x),
        op.Atan(x),
        # op.Atanh(x),
        op.Ceil(x),
        op.Cos(x),
        # op.Cosh(x),
        op.Exp(x),
        # op.Erf(x),
        op.Floor(x),
        op.Log(x),
        op.Neg(x),
        op.Reciprocal(x),
        op.Relu(x),
        op.Sin(x),
        # op.Sinh(x),
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

        op.Sum(x, op.Relu(y)),
        op.Sum(x, op.Floor(y), y, op.Sin(y)),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_math_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5)
    y = torch.rand(3, 4, 5)

    # ort inference
    sess = ort.InferenceSession("test_onnx_math_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_math_ops.onnx inputshape=[3,4,5],[3,4,5] inputshape2=[13,14,15],[13,1,15]")

    # pnnx inference
    import test_onnx_math_ops_pnnx
    b = test_onnx_math_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_math_ops_ncnn
    c = test_onnx_math_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
