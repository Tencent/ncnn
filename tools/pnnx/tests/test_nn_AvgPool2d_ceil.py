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
        # stride > kernel and the out of range window only happens on one axis,
        # 3x4 drops the extra row but keeps the extra column, 4x3 the other way round
        self.pool_4 = nn.AvgPool2d(kernel_size=2, stride=3, padding=0, ceil_mode=True, count_include_pad=True)

    def forward(self, x):
        a = self.pool_0(x)
        b = self.pool_1(x)
        c = self.pool_2(x[:, :, :3, :3])
        d = self.pool_3(x[:, :, :4, :4])
        e = self.pool_4(x[:, :, :3, :4])
        f = self.pool_4(x[:, :, :4, :3])
        return a, b, c, d, e, f

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
    # a mixed axis case needs one pooling with full padding plus one crop
    expect = [
        (2, 2, 0, 0),
        (3, 2, 0, 0),
        (2, 3, 1, 0),
        (2, 3, 0, 0),
        (2, 3, 0, 0),
        (2, 3, 0, 0),
    ]
    pooling = []
    pooling_outputs = set()
    crops = 0
    with open("test_nn_AvgPool2d_ceil.ncnn.param") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            if parts[0] == "Crop":
                # the tensor slicing in forward() also becomes Crop, only count
                # the ones that trim a pooling result
                if parts[4] in pooling_outputs:
                    crops += 1
                continue
            if parts[0] != "Pooling":
                continue
            kv = {}
            for p in parts[6:]:
                if "=" in p:
                    k, v = p.split("=")
                    kv[int(k)] = int(float(v))
            pooling.append((kv.get(1), kv.get(2), kv.get(5), kv.get(6)))
            pooling_outputs.add(parts[5])

    if sorted(pooling) != sorted(expect):
        return False

    # the two mixed axis pools must each be followed by a crop
    if crops != 2:
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
