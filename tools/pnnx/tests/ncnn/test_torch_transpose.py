# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 3, 1)
        self.conv3 = nn.Conv2d(3, 4, 1)
        self.conv4 = nn.Conv2d(3, 4, 1)
        self.conv5 = nn.Conv2d(3, 4, 1)
        self.conv6 = nn.Conv2d(3, 4, 1)
        self.conv7 = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w, v):
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 1, 2)
        z = torch.transpose(z, 0, 2)
        w0 = self.conv(w)
        w0 = torch.transpose(w0, 0, 1).reshape(8, 5, 7)
        w1 = self.conv3(w)
        w1 = torch.transpose(w1, 0, 1)
        w1 = torch.transpose(w1, 2, 3)
        w1 = torch.flatten(w1, start_dim=0, end_dim=1)
        w2 = self.conv4(w)
        w2 = torch.transpose(w2, 0, 1)
        w2 = torch.clone(w2)
        w2 = w2.permute(0, 2, 1, 3)
        w2 = torch.clone(w2)
        w2 = torch.flatten(w2, start_dim=0, end_dim=2)
        w3 = self.conv5(w)
        w3 = torch.transpose(w3, 0, 1)
        w4 = self.conv6(w)
        w4 = torch.transpose(w4, 0, 2)
        w4 = F.relu(w4)
        w5 = self.conv7(w)
        w5 = torch.transpose(w5, 0, 3)
        w5 = F.relu(w5)
        v = v.reshape(4, 2, 5, 7)
        v = torch.transpose(v, 0, 1)
        v = torch.clone(v)
        v = v.permute(0, 1, 3, 2)
        v = self.conv2(v)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z, w0, w1, w2, w3, w4, w5, v

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)

    a = net(x, y, z, w, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, v))
    mod.save("test_torch_transpose.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_transpose.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7],[280]")

    # ncnn inference
    import test_torch_transpose_ncnn
    b = test_torch_transpose_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
