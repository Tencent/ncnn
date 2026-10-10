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
def Model(x: FLOAT["N","C","W"], y: FLOAT["N","C","H","W"], z: FLOAT["N","C","D","H","W"]):

    prelu_slope = op.RandomNormal(seed=0.0, shape=[12])

    return (
        op.Celu(x),
        op.Clip(x, min=0, max=2.3),
        op.Elu(x),
        op.HardSigmoid(x),
        op.HardSwish(x),
        op.LeakyRelu(x),
        op.LogSoftmax(x),
        op.Mish(x),
        op.PRelu(x, op.Unsqueeze(prelu_slope, axes=[1])),
        op.Relu(x),
        op.Selu(x),
        op.Shrink(x),
        op.Sigmoid(x),
        op.Softmax(x),
        op.Softplus(x),

        op.Celu(y),
        op.Clip(y, min=0, max=2.3),
        op.Elu(y),
        op.HardSigmoid(y),
        op.HardSwish(y),
        op.LeakyRelu(y),
        op.LogSoftmax(y),
        op.Mish(y),
        op.PRelu(y, op.Unsqueeze(prelu_slope, axes=[1,2])),
        op.Relu(y),
        op.Selu(y),
        op.Shrink(y),
        op.Sigmoid(y),
        op.Softmax(y),
        op.Softplus(y),

        # op.Celu(z),
        op.Clip(z, min=0, max=2.3),
        # op.Elu(z),
        op.HardSigmoid(z),
        op.HardSwish(z),
        op.LeakyRelu(z),
        # op.LogSoftmax(z),
        op.Mish(z),
        op.PRelu(z, op.Unsqueeze(prelu_slope, axes=[1,2,3])),
        op.Relu(z),
        # op.Selu(z),
        op.Shrink(z),
        op.Sigmoid(z),
        # op.Softmax(z),
        op.Softplus(z),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_activation_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64)
    y = torch.rand(1, 12, 48, 64)
    z = torch.rand(1, 12, 21, 28, 44)

    # ort inference
    sess = ort.InferenceSession("test_onnx_activation_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy(), "z": z.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_activation_ops.onnx inputshape=[1,12,64],[1,12,48,64],[1,12,21,28,44] inputshape2=[7,12,22],[8,12,33,11],[9,12,9,12,13] fp16=0")

    # pnnx inference
    import test_onnx_activation_ops_pnnx
    b = test_onnx_activation_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_activation_ops_ncnn
    c = test_onnx_activation_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
