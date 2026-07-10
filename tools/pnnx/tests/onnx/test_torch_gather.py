# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        # 1D gather along axis 0
        idx_1d = torch.tensor([2, 0, 1], dtype=torch.int64)
        a = torch.gather(x, 0, idx_1d)

        # 2D gather along axis 0
        idx_2d_axis0 = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int64)
        b = torch.gather(y, 0, idx_2d_axis0)

        # 2D gather along axis 1
        idx_2d_axis1 = torch.tensor([[1, 0, 2], [0, 2, 1]], dtype=torch.int64)
        c = torch.gather(y, 1, idx_2d_axis1)

        # 3D gather along axis 1
        idx_3d = torch.zeros(2, 2, 4, dtype=torch.int64)
        d = torch.gather(z, 1, idx_3d)

        # 3D gather along last axis (negative index)
        idx_3d_last = torch.zeros(2, 3, 2, dtype=torch.int64)
        e = torch.gather(z, -1, idx_3d_last)

        return a, b, c, d, e


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(5)
    y = torch.rand(3, 4)
    z = torch.rand(2, 3, 4)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_torch_gather.onnx",
                      opset_version=13)

    # onnx to pnnx
    import os
    os.system(
        "../../src/pnnx test_torch_gather.onnx "
        "inputshape=[5],[3,4],[2,3,4]"
    )

    # pnnx inference
    import test_torch_gather_pnnx
    b = test_torch_gather_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
