# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1
        x = F.softmax(x, 1)
        y = F.softmax(y, 0)
        z = F.softmax(z, 2)
        w = F.softmax(w, 3)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)

    a0, a1, a2, a3 = net(x, y, z, w)

    program = torch.export.export(net, (x, y, z, w))
    torch.export.save(program, "test_F_softmax.pt2")

    from pnnxutil import run_pnnx
    run_pnnx("test_F_softmax.pt2", "inputshape=[1,16],[12,2,16],[1,3,12,16],[1,5,7,9,11]")

    import test_F_softmax_pnnx
    b0, b1, b2, b3 = test_F_softmax_pnnx.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4) and torch.allclose(a2, b2, 1e-4, 1e-4) and torch.allclose(a3, b3, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
