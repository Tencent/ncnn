# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dropout_0 = nn.Dropout()
        self.dropout_1 = nn.Dropout(p=0.7)

    def forward(self, x, y, z, w, q):
        x = self.dropout_0(x)
        y = self.dropout_0(y)
        z = self.dropout_1(z)
        w = self.dropout_1(w)
        q = F.max_pool2d(q, 1)
        q = self.dropout_0(q)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w = F.relu(w)
        q = F.relu(q)
        return x, y, z, w, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(12, 64)
    z = torch.rand(12, 24, 64)
    w = torch.rand(12, 24, 32, 64)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q))
    mod.save("test_nn_Dropout.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Dropout.pt inputshape=[12],[12,64],[12,24,64],[12,24,32,64],[2,3,5,7]")

    # ncnn inference
    import test_nn_Dropout_ncnn
    b = test_nn_Dropout_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
