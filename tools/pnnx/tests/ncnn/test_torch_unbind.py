# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x0, x1, x2 = torch.unbind(x, dim=0)
        y0, y1, y2, y3, y4, y5, y6, y7, y8 = torch.unbind(y, dim=1)

        x0 = F.relu(x0)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        y0 = F.relu(y0)
        y1 = F.relu(y1)
        y2 = F.relu(y2)
        y3 = F.relu(y3)
        y4 = F.relu(y4)
        y5 = F.relu(y5)
        y6 = F.relu(y6)
        y7 = F.relu(y7)
        y8 = F.relu(y8)
        return x0, x1, x2, y0, y1, y2, y3, y4, y5, y6, y7, y8

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_unbind.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_unbind.pt inputshape=[3,16],[5,9,11]")

    # ncnn inference
    import test_torch_unbind_ncnn
    b = test_torch_unbind_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
