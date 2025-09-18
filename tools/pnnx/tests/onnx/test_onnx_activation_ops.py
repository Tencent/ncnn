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
def Model(x: FLOAT[2,3,4]):
    return (
        op.Celu(x),
        op.Clip(x),
        op.Elu(x),
        # op.Gelu(x),
        op.HardSigmoid(x),
        op.HardSwish(x),
        op.LeakyRelu(x),
        op.LogSoftmax(x),
        op.Mish(x),
        # op.PRelu(x),
        op.Relu(x),
        op.Selu(x),
        op.Sigmoid(x),
        op.Softmax(x),
        op.Softplus(x),
        # op.Swish(x),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_activation_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(2, 3, 4)

    # ort inference
    sess = ort.InferenceSession("test_onnx_activation_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    # onnx to ncnn
    import os
    os.system("../../src/pnnx test_onnx_activation_ops.onnx")

    # ncnn inference
    import test_onnx_activation_ops_ncnn
    b = test_onnx_activation_ops_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
