# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.stack((x, y), dim=0)
        out1 = torch.stack((x, y), dim=2)
        out2 = torch.stack((z, w), dim=2)
        out3 = torch.stack((z, w), dim=-1)
        out0.relu_()
        out1.relu_()
        out2.relu_()
        out3.relu_()
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(5, 9, 3)
    w = torch.rand(5, 9, 3)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_stack.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_stack.pt inputshape=[3,16],[3,16],[5,9,3],[5,9,3]")

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
