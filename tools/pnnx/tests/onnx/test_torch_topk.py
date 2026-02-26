# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x_values, x_indices = torch.topk(
            x, 2, dim=1, largest=True, sorted=True
        )
        y_values, y_indices = torch.topk(
            y, 4, dim=3, largest=False, sorted=True
        )
        z_values, z_indices = torch.topk(
            z, 3, dim=0, largest=True, sorted=True
        )
        return x_values, x_indices, y_values, y_indices, z_values, z_indices


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_torch_topk.onnx")

    # onnx to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_topk.onnx "
        "inputshape=[1,3,16],[1,5,9,11],[14,8,5,9,10]"
    )

    # pnnx inference
    import test_torch_topk_pnnx
    b = test_torch_topk_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
