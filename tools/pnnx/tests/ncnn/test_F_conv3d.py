# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w2 = nn.Parameter(torch.rand(12, 6, 4, 4, 4))
        self.b2 = nn.Parameter(torch.rand(12))
        self.w3 = nn.Parameter(torch.rand(6, 4, 3, 3, 3))

    def forward(self, y):
        y = F.conv3d(y, self.w2, self.b2, stride=(2,2,2), padding=(2,2,2))
        y = F.conv3d(y, self.w3, None, stride=(2,2,2), padding=(1,1,1), groups=3)
        return y

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    y = torch.rand(1, 6, 12, 11, 10)

    a = net(y)

    # export torchscript
    mod = torch.jit.trace(net, y)
    mod.save("test_F_conv3d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_conv3d.pt inputshape=[1,6,12,11,10]")

    # ncnn inference
    import test_F_conv3d_ncnn
    b = test_F_conv3d_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
