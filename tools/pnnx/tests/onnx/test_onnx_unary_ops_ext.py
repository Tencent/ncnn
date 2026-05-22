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
def Model(x: FLOAT[3, 4, 5]):
    return (
        op.Sign(x),
        op.Sinh(x),
        op.Cosh(x),
        op.Asinh(x),
        op.Atanh(x),
    )


def test():
    onnx.save(Model.to_model_proto(), "test_onnx_unary_ops_ext.onnx")

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5)

    sess = ort.InferenceSession("test_onnx_unary_ops_ext.onnx")
    a = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy()}))

    import os
    os.system("../../src/pnnx test_onnx_unary_ops_ext.onnx inputshape=[3,4,5]")

    import test_onnx_unary_ops_ext_pnnx
    b = test_onnx_unary_ops_ext_pnnx.test_inference()

    import test_onnx_unary_ops_ext_ncnn
    c = test_onnx_unary_ops_ext_ncnn.test_inference()

    for a0, b0, c0 in zip(a, b, c):
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
