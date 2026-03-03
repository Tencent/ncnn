# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = F.dropout(x, training=False)
        y = F.dropout(y, training=False)
        z = F.dropout(z, p=0.6, training=False)
        w = F.dropout(w, p=0.1, training=False)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        w = F.relu(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(2, 16)
    z = torch.rand(3, 12, 16)
    w = torch.rand(5, 7, 9, 11)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_F_dropout.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_dropout.pt inputshape=[16],[2,16],[3,12,16],[5,7,9,11]")

    # ncnn inference
    import test_F_dropout_ncnn
    b = test_F_dropout_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
