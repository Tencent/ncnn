# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q, r):
        z = F.max_pool2d(z, kernel_size=(2,2))
        w = F.max_pool2d(w, kernel_size=(2,2))
        q = F.max_pool2d(q, 1)
        r = F.max_pool2d(r, 1)
        out0 = torch.stack((x, y), dim=0)
        out1 = torch.stack((x, y), dim=2)
        out2 = torch.stack((z, w), dim=2)
        out3 = torch.stack((z, w), dim=-1)
        out4 = torch.stack((q, r), dim=1)
        out5 = torch.stack((q, r), dim=-1)
        out0.relu_()
        out1.relu_()
        out2.relu_()
        out3.relu_()
        out4.relu_()
        out5.relu_()
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(1, 5, 10, 4)
    w = torch.rand(1, 5, 10, 4)
    q = torch.rand(2, 3, 5, 7)
    r = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q, r))
    mod.save("test_torch_stack.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_stack.pt inputshape=[3,16],[3,16],[1,5,10,4],[1,5,10,4],[2,3,5,7],[2,3,5,7]")

    # ncnn inference
    import test_torch_stack_ncnn
    b = test_torch_stack_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
