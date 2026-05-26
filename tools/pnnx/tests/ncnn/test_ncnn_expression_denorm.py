# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = x * -4.928072e-40 + y
        out1 = out0 * -4.881353e-40 + y
        return out0, out1


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 4, 8, 8)
    y = torch.rand(1, 4, 8, 8)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_ncnn_expression_denorm.pt")

    # torchscript to pnnx
    if (
        os.system(
            "../../src/pnnx test_ncnn_expression_denorm.pt inputshape=[1,4,8,8],[1,4,8,8]"
        )
        != 0
    ):
        return False

    # ensure pass_ncnn produced output files
    if not os.path.exists("test_ncnn_expression_denorm.ncnn.param"):
        return False
    if not os.path.exists("test_ncnn_expression_denorm.ncnn.bin"):
        return False

    # ensure the test model really contains denormalized literal expression
    if not os.path.exists("test_ncnn_expression_denorm.pnnx.param"):
        return False
    with open("test_ncnn_expression_denorm.pnnx.param", "r", encoding="utf-8") as f:
        pnnx_param = f.read()

    if "pnnx.Expression" not in pnnx_param:
        return False
    if "e-40" not in pnnx_param:
        return False

    # ncnn inference
    import test_ncnn_expression_denorm_ncnn

    b = test_ncnn_expression_denorm_ncnn.test_inference()

    for aa, bb in zip(a, b):
        if not torch.allclose(aa, bb, 1e-6, 1e-6):
            return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
