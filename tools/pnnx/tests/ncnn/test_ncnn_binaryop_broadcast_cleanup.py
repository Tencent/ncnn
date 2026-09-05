# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn


class ModelChannelBroadcastBatch(nn.Module):
    def __init__(self):
        super(ModelChannelBroadcastBatch, self).__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, x, y):
        x = self.conv(x)
        y = y.reshape(2, 4, 1, 1)
        return x * y, y + x


class ModelChannelBroadcastNoBatch(nn.Module):
    def __init__(self):
        super(ModelChannelBroadcastNoBatch, self).__init__()

    def forward(self, x, y):
        y = y.reshape(4, 1, 1)
        return x * y, y + x


class ModelLastDimBroadcast(nn.Module):
    def __init__(self):
        super(ModelLastDimBroadcast, self).__init__()

    def forward(self, x, y):
        return x * y, y + x


def compare(a, b):
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True


def run_model(name, net, inputs, inputshape, expect_reshape):
    net.eval()

    a = net(*inputs)

    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    if os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape) != 0:
        return False

    with open(name + ".ncnn.param") as f:
        has_reshape = "Reshape" in f.read()

    if has_reshape != expect_reshape:
        return False

    ncnn = __import__(name + "_ncnn")
    b = ncnn.test_inference()

    return compare(a, b)


def test():
    torch.manual_seed(0)
    x0 = torch.rand(2, 4, 5, 7)
    y0 = torch.rand(2, 4)

    torch.manual_seed(0)
    x1 = torch.rand(4, 5, 7)
    y1 = torch.rand(4)

    torch.manual_seed(0)
    x2 = torch.rand(4)
    y2 = torch.rand(4, 5, 4)

    if not run_model("test_ncnn_binaryop_broadcast_cleanup_batch", ModelChannelBroadcastBatch(), (x0, y0), "[2,4,5,7],[2,4]", True):
        return False
    if not run_model("test_ncnn_binaryop_broadcast_cleanup_nobatch", ModelChannelBroadcastNoBatch(), (x1, y1), "[4,5,7],[4]", False):
        return False
    if not run_model("test_ncnn_binaryop_broadcast_cleanup_lastdim", ModelLastDimBroadcast(), (x2, y2), "[4],[4,5,4]", True):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
