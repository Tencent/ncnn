# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, w, x, y, z):
        # 1d: dim omitted (torch flattens first) and a plain 1d repeat
        w0 = torch.repeat_interleave(w, 2)
        w1 = torch.repeat_interleave(w, 3, dim=0)
        # 2d: there is no channel axis in ncnn here, so the slice+concat path is used
        x0 = torch.repeat_interleave(x, 2, dim=0)
        x1 = torch.repeat_interleave(x, 2, dim=1)
        x2 = torch.repeat_interleave(x, 2)
        # 3d: dim0 is the channel axis, dim1/dim2 need a permute first; dim0 also tests per-element repeats
        y0 = torch.repeat_interleave(y, 2, dim=0)
        y1 = torch.repeat_interleave(y, 3, dim=1)
        y2 = torch.repeat_interleave(y, 2, dim=-1)
        y3 = torch.repeat_interleave(y, torch.tensor([2, 1, 3]), dim=0)
        y4 = torch.repeat_interleave(y, 2)
        y5 = torch.repeat_interleave(y, 1, dim=1)
        # 4d: dim0 has length 1 (the base case), dim1/dim2/dim3 land in the d/h/w slots
        z0 = torch.repeat_interleave(z, 2, dim=0)
        z1 = torch.repeat_interleave(z, 2, dim=1)
        z2 = torch.repeat_interleave(z, 2, dim=2)
        z3 = torch.repeat_interleave(z, 3, dim=3)
        return w0, w1, x0, x1, x2, y0, y1, y2, y3, y4, y5, z0, z1, z2, z3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    w = torch.rand(4)
    x = torch.rand(3, 5)
    y = torch.rand(3, 7, 8)
    z = torch.rand(1, 3, 5, 6)

    a = net(w, x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (w, x, y, z))
    mod.save("test_torch_repeat_interleave.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_torch_repeat_interleave.pt inputshape=[4],[3,5],[3,7,8],[1,3,5,6]")

    # ncnn inference
    import test_torch_repeat_interleave_ncnn
    b = test_torch_repeat_interleave_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            print(a0)
            print(b0)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
