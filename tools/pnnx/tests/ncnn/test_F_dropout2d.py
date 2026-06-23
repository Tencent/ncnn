# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, q):
        x = F.dropout2d(x, training=False)
        z = F.dropout2d(y, p=0.6, training=False)
        q = F.max_pool2d(q, 1)
        q = F.dropout2d(q, training=False)
        x = F.relu(x)
        y = F.relu(y)
        q = F.relu(q)
        return x, y, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 2, 16)
    y = torch.rand(3, 12, 16)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, q))
    mod.save("test_F_dropout2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_dropout2d.pt inputshape=[12,2,16],[3,12,16],[2,3,5,7]")

    # ncnn inference
    import test_F_dropout2d_ncnn
    b = test_F_dropout2d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
