# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import torch
import onnx
import onnxruntime as ort
from onnxscript import FLOAT, script
from onnxscript import opset21 as op

@script()
def Model(x: FLOAT["N","C","W"], y: FLOAT["N","C","H","W"], z: FLOAT["N","C","D","H","W"]):

    gn_scale = op.RandomNormal(seed=4.0, shape=[12])
    gn_bias = op.RandomNormal(seed=5.0, shape=[12])

    return (
        op.Gelu(x),
        # op.Swish(x),

        op.GroupNormalization(x, gn_scale, gn_bias, num_groups=3),

        op.Gelu(y),
        # op.Swish(y),

        op.GroupNormalization(y, gn_scale, gn_bias, num_groups=3),

        op.Gelu(z),
        # op.Swish(z),

        op.GroupNormalization(z, gn_scale, gn_bias, num_groups=3),

        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_opset21_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64)
    y = torch.rand(1, 12, 48, 64)
    z = torch.rand(1, 12, 21, 28, 44)

    # ort inference
    sess = ort.InferenceSession("test_onnx_opset21_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy(), "z": z.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_opset21_ops.onnx inputshape=[1,12,64],[1,12,48,64],[1,12,21,28,44] inputshape2=[7,12,22],[8,12,33,11],[9,12,9,12,13] fp16=0")

    # pnnx inference
    import test_onnx_opset21_ops_pnnx
    b = test_onnx_opset21_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_opset21_ops_ncnn
    c = test_onnx_opset21_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
