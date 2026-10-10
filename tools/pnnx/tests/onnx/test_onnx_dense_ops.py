# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset19 as op

@script()
def Model(x: FLOAT["M","K"], y: FLOAT["K","N"], z: FLOAT["M","N"]):

    W = op.RandomNormal(seed=0.0, shape=[32,64])
    B = op.RandomNormal(seed=1.0, shape=[32])

    return (
        op.Gemm(x, y, None),
        op.Gemm(y, x, None, transA=1, transB=1),
        op.Gemm(x, y, z, alpha=0.8, beta=0.5),
        op.Gemm(x, W, B, transB=1),
        op.Gemm(x, op.Transpose(W, perm=[1,0]), B, transB=0),
        op.MatMul(x, y),
        op.MatMul(op.Transpose(y, perm=[1,0]), op.Transpose(x, perm=[1,0])),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_dense_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(48, 64)
    y = torch.rand(64, 32)
    z = torch.rand(48, 32)

    # ort inference
    sess = ort.InferenceSession("test_onnx_dense_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy(), "z": z.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_dense_ops.onnx inputshape=[48,64],[64,32],[48,32] fp16=0")

    # pnnx inference
    import test_onnx_dense_ops_pnnx
    b = test_onnx_dense_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_dense_ops_ncnn
    c = test_onnx_dense_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
