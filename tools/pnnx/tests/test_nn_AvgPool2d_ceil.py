# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # ceil_mode pads the last window with zeros,
        # torch still divides by the number of real elements
        self.pool_0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=True)
        self.pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True, count_include_pad=True)
        # stride > kernel, the last window starts outside the 3x3 input and is dropped
        self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=3, padding=0, ceil_mode=True, count_include_pad=True)
        # stride > kernel but the last window still overlaps the 4x4 input, it is kept
        self.pool_3 = nn.AvgPool2d(kernel_size=2, stride=3, padding=0, ceil_mode=True, count_include_pad=True)

    def forward(self, x):
        a = self.pool_0(x)
        b = self.pool_1(x)
        c = self.pool_2(x[:, :, :3, :3])
        d = self.pool_3(x[:, :, :4, :4])
        return a, b, c, d

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, 7, 9)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AvgPool2d_ceil.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AvgPool2d_ceil.pt inputshape=[1,1,7,9]")

    # pnnx inference
    import test_nn_AvgPool2d_ceil_pnnx
    b = test_nn_AvgPool2d_ceil_pnnx.test_inference()

    if len(a) != len(b):
        return False

    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0):
            return False

    # the pnnx python module only replays torch ops, the ncnn conversion itself
    # must be checked on the generated param: ceil_mode needs full padding to
    # keep the zero padded last window, unless stride > kernel where that window
    # starts outside the input and torch drops it, and count_include_pad must be
    # off so the average divides by the real element count
    expect = {
        "avgpool2d_0": (2, 2, 0, 0),
        "avgpool2d_1": (3, 2, 0, 0),
        "avgpool2d_2": (2, 3, 1, 0),
        "avgpool2d_3": (2, 3, 0, 0),
    }
    found = {}
    with open("test_nn_AvgPool2d_ceil.ncnn.param") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or parts[0] != "Pooling":
                continue
            kv = {}
            for p in parts[4:]:
                if "=" in p:
                    k, v = p.split("=")
                    kv[int(k)] = int(float(v))
            found[parts[1]] = (kv.get(1), kv.get(2), kv.get(5), kv.get(6))

    if found != expect:
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
