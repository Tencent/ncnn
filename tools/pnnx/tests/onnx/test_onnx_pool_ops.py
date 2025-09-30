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
def Model(x: FLOAT["N","C","D","H","W"], y: FLOAT["N","C","H","W"], z: FLOAT["N","H","W"]):

    return (
        op.AveragePool(x, kernel_shape=[2,2,2], strides=[1,2,2], pads=[1,1,1,1,1,1]),
        op.AveragePool(x, kernel_shape=[5,5,5], auto_pad='SAME_LOWER', ceil_mode=True, count_include_pad=True),
        op.AveragePool(x, kernel_shape=[3,3,3]),
        op.AveragePool(y, kernel_shape=[3,3], strides=[2,2], pads=[1,1,1,1]),
        op.AveragePool(y, kernel_shape=[5,5], auto_pad='SAME_LOWER', ceil_mode=True, count_include_pad=True),
        op.AveragePool(y, kernel_shape=[3,3]),
        op.AveragePool(z, kernel_shape=[3], strides=[2], pads=[1,1]),
        op.AveragePool(z, kernel_shape=[5], auto_pad='SAME_LOWER', ceil_mode=True, count_include_pad=True),
        op.AveragePool(z, kernel_shape=[3]),

        op.MaxPool(x, kernel_shape=[2,2,2], strides=[1,2,2], pads=[1,1,1,1,1,1]),
        op.MaxPool(x, kernel_shape=[5,5,5], auto_pad='SAME_LOWER', ceil_mode=True),
        op.MaxPool(x, kernel_shape=[3,3,3]),
        op.MaxPool(y, kernel_shape=[3,3], strides=[2,2], pads=[1,1,1,1]),
        op.MaxPool(y, kernel_shape=[5,5], auto_pad='SAME_LOWER', ceil_mode=True),
        op.MaxPool(y, kernel_shape=[3,3]),
        op.MaxPool(z, kernel_shape=[3], strides=[2], pads=[1,1]),
        op.MaxPool(z, kernel_shape=[5], auto_pad='SAME_LOWER', ceil_mode=True),
        op.MaxPool(z, kernel_shape=[3]),

        op.GlobalAveragePool(x),
        op.GlobalAveragePool(y),
        op.GlobalAveragePool(z),

        op.GlobalMaxPool(x),
        op.GlobalMaxPool(y),
        op.GlobalMaxPool(z),
        )

def test():
    # save onnx
    onnx.save(Model.to_model_proto(), "test_onnx_pool_ops.onnx")

    torch.manual_seed(0)
    x = torch.rand(1, 12, 15, 24, 25)
    y = torch.rand(1, 12, 24, 25)
    z = torch.rand(1, 24, 25)

    # ort inference
    sess = ort.InferenceSession("test_onnx_pool_ops.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy(), "z": z.numpy()}))

    # onnx to pnnx and ncnn
    import os
    os.system("../../src/pnnx test_onnx_pool_ops.onnx inputshape=[1,12,15,24,25],[1,12,24,25],[1,24,25] inputshape2=[10,14,17,26,29],[10,14,26,29],[10,26,29]")

    # pnnx inference
    import test_onnx_pool_ops_pnnx
    b = test_onnx_pool_ops_pnnx.test_inference()

    # ncnn inference
    import test_onnx_pool_ops_ncnn
    c = test_onnx_pool_ops_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-3, 1e-3) or not torch.allclose(a0, c0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
