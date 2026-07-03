# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.act_0 = nn.Softmax2d()

    def forward(self, x, y):
        x = x * 2 - 1
        y = y * 2 - 1
        x = self.act_0(x)
        y = self.act_0(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)
    y = torch.rand(2, 12, 24, 64)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_Softmax2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Softmax2d.pt inputshape=[1,12,24,64],[2,12,24,64]")

    # ncnn inference
    import test_nn_Softmax2d_ncnn
    b = test_nn_Softmax2d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
