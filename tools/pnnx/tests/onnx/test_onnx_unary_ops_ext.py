# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys
if sys.version_info < (3, 9):
    sys.exit(0)

import os

import onnxruntime as ort
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return (
            torch.sign(x),
            torch.sinh(x),
            torch.asinh(x),
            torch.cosh(x),
            torch.acosh(y),
            torch.atanh(x),
        )


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 4, 5) - 0.5
    y = torch.rand(3, 4, 5) + 1.0

    a = net(x, y)

    # export onnx
    torch.onnx.export(net, (x, y), "test_onnx_unary_ops_ext.onnx",
                      input_names=["x", "y"],
                      dynamic_axes={"x": {0: "d0", 1: "d1", 2: "d2"},
                                    "y": {0: "d0", 1: "d1", 2: "d2"}},
                      opset_version=19)

    sess = ort.InferenceSession("test_onnx_unary_ops_ext.onnx")
    a_ort = tuple(torch.from_numpy(out) for out in sess.run(None, {"x": x.numpy(), "y": y.numpy()}))

    os.system("../../src/pnnx test_onnx_unary_ops_ext.onnx inputshape=[3,4,5],[3,4,5] inputshape2=[2,3,4],[2,3,4] fp16=0")

    import test_onnx_unary_ops_ext_pnnx
    b = test_onnx_unary_ops_ext_pnnx.test_inference()

    import test_onnx_unary_ops_ext_ncnn
    c = test_onnx_unary_ops_ext_ncnn.test_inference()

    for a0, a1, b0, c0 in zip(a, a_ort, b, c):
        if not torch.allclose(a0, a1, 1e-4, 1e-4):
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4) or not torch.allclose(a0, c0, 1e-4, 1e-4):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
