# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z):
        x = torch.t(x)
        y = torch.t(y)
        z = self.conv(z)
        z = torch.flatten(z, 1)
        z = torch.t(z)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3)
    y = torch.rand(5, 9)
    z = torch.rand(2, 3, 1, 1)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_t.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_t.pt inputshape=[3],[5,9],[2,3,1,1]")

    # ncnn inference
    import test_torch_t_ncnn
    b = test_torch_t_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
