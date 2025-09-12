# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.Dropout2d()
        self.dropout_1 = nn.Dropout2d(p=0.7)

    def forward(self, x, y):
        x = self.dropout_0(x)
        y = self.dropout_1(y)
        x = F.relu(x)
        y = F.relu(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 24, 64)
    y = torch.rand(3, 4, 5)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_Dropout2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Dropout2d.pt inputshape=[12,24,64],[3,4,5]")

    # ncnn inference
    import test_nn_Dropout2d_ncnn
    b = test_nn_Dropout2d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
