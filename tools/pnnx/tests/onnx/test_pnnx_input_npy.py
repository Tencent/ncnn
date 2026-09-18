# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return x + y, x * 0.25 - y


def _allclose(a, b):
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


def test():
    net = Model()
    net.eval()

    x_np = (np.arange(12, dtype=np.float32).reshape(1, 3, 4) - 5.0) / 7.0
    y_np = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)

    x_path = "test_pnnx_input_npy_onnx_x.npy"
    y_path = "test_pnnx_input_npy_onnx_y.npy"
    np.save(x_path, np.array(x_np, dtype=np.float32, order="F"))
    np.save(y_path, y_np)

    x = torch.from_numpy(np.ascontiguousarray(np.load(x_path)))
    y = torch.from_numpy(np.ascontiguousarray(np.load(y_path)))

    a = net(x, y)

    torch.onnx.export(net, (x, y), "test_pnnx_input_npy.onnx")

    ret = os.system("../../src/pnnx test_pnnx_input_npy.onnx input=test_pnnx_input_npy_onnx_x.npy,test_pnnx_input_npy_onnx_y.npy")
    if ret != 0:
        return False

    import test_pnnx_input_npy_pnnx
    pnnx_net = test_pnnx_input_npy_pnnx.Model()
    pnnx_net.eval()
    b = pnnx_net(x, y)

    return _allclose(a, b)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
