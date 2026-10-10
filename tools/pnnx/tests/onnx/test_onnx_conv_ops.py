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

    conv1d_W = op.RandomNormal(seed=0.0, shape=[14,12,3])
    conv1d_B = op.RandomNormal(seed=1.0, shape=[14])

    deconv1d_W = op.RandomNormal(seed=2.0, shape=[12,14,3])
    deconv1d_B = op.RandomNormal(seed=3.0, shape=[14])

    conv2d_W = op.RandomNormal(seed=0.0, shape=[14,12,3,3])
    conv2d_B = op.RandomNormal(seed=1.0, shape=[14])

    deconv2d_W = op.RandomNormal(seed=2.0, shape=[12,14,3,3])
    deconv2d_B = op.RandomNormal(seed=3.0, shape=[14])

    conv3d_W = op.RandomNormal(seed=0.0, shape=[14,12,2,3,3])
    conv3d_B = op.RandomNormal(seed=1.0, shape=[14])

    deconv3d_W = op.RandomNormal(seed=2.0, shape=[12,14,2,3,3])
    deconv3d_B = op.RandomNormal(seed=3.0, shape=[14])

    return (
        op.Conv(x, conv1d_W, conv1d_B),
        op.Conv(x, conv1d_W, None, dilations=[2], strides=[2], pads=[1,1]),
        op.Conv(x, conv1d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(x, deconv1d_W, deconv1d_B),
        op.ConvTranspose(x, deconv1d_W, None, dilations=[2], strides=[2], pads=[1,1]),
        # op.ConvTranspose(x, deconv1d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(x, deconv1d_W, None, strides=[2], output_padding=[1]),

        op.Conv(y, conv2d_W, conv2d_B),
        op.Conv(y, conv2d_W, None, dilations=[1,2], strides=[2,2], pads=[1,1,1,1]),
        op.Conv(y, conv2d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(y, deconv2d_W, deconv2d_B),
        op.ConvTranspose(y, deconv2d_W, None, dilations=[1,2], strides=[2,2], pads=[1,1,1,1]),
        # op.ConvTranspose(y, deconv2d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(y, deconv2d_W, None, strides=[2,2], output_padding=[1,1]),

        op.Conv(z, conv3d_W, conv3d_B),
        op.Conv(z, conv3d_W, None, dilations=[1,2,2], strides=[2,2,2], pads=[1,1,1,1,1,1]),
        op.Conv(z, conv3d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(z, deconv3d_W, deconv3d_B),
        op.ConvTranspose(z, deconv3d_W, None, dilations=[1,2,2], strides=[2,2,2], pads=[1,1,1,1,1,1]),
        # op.ConvTranspose(z, deconv3d_W, None, auto_pad='SAME_UPPER'),
        op.ConvTranspose(z, deconv3d_W, None, strides=[2,2,2], output_padding=[1,1,1]),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_conv_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64)
    y = torch.rand(1, 12, 48, 64)
    z = torch.rand(1, 12, 21, 28, 44)

    # ort inference
    sess = ort.InferenceSession("test_onnx_conv_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy(), "z": z.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_conv_ops.onnx inputshape=[1,12,64],[1,12,48,64],[1,12,21,28,44] fp16=0")

    # pnnx inference
    import test_onnx_conv_ops_pnnx
    b = test_onnx_conv_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_conv_ops_ncnn
    c = test_onnx_conv_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
