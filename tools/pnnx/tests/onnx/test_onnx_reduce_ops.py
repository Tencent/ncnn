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
def Model(x: FLOAT["C","H","W"]):
    return (
        op.ReduceL1(x, axes=[1], keepdims=1),
        op.ReduceL1(x, axes=[2], keepdims=0),
        op.ReduceL1(x),
        op.ReduceL2(x, axes=[1], keepdims=1),
        op.ReduceL2(x, axes=[2], keepdims=0),
        op.ReduceL2(x),
        op.ReduceMax(x, axes=[1], keepdims=1),
        op.ReduceMax(x, axes=[2], keepdims=0),
        op.ReduceMax(x),
        op.ReduceMin(x, axes=[1], keepdims=1),
        op.ReduceMin(x, axes=[2], keepdims=0),
        op.ReduceMin(x),
        op.ReduceMean(x, axes=[1], keepdims=1),
        op.ReduceMean(x, axes=[2], keepdims=0),
        op.ReduceMean(x),
        op.ReduceSum(x, axes=[1], keepdims=1),
        op.ReduceSum(x, axes=[2], keepdims=0),
        op.ReduceSum(x),
        op.ReduceProd(x, axes=[1], keepdims=1),
        op.ReduceProd(x, axes=[2], keepdims=0),
        op.ReduceProd(x),
        op.ReduceSumSquare(x, axes=[1], keepdims=1),
        op.ReduceSumSquare(x, axes=[2], keepdims=0),
        op.ReduceSumSquare(x),
        op.ReduceLogSum(x, axes=[1], keepdims=1),
        op.ReduceLogSum(x, axes=[2], keepdims=0),
        op.ReduceLogSum(x),
        op.ReduceLogSumExp(x, axes=[1], keepdims=1),
        op.ReduceLogSumExp(x, axes=[2], keepdims=0),
        op.ReduceLogSumExp(x),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_reduce_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5)

    # ort inference
    sess = ort.InferenceSession("test_onnx_reduce_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_reduce_ops.onnx inputshape=[3,4,5] inputshape2=[13,14,15]")

    # pnnx inference
    import test_onnx_reduce_ops_pnnx
    b = test_onnx_reduce_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_reduce_ops_ncnn
    c = test_onnx_reduce_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
