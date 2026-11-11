# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.act_0 = nn.ELU()
        self.act_1 = nn.ELU(alpha=1.3)
        self.act_2 = nn.ELU(alpha=2)

    def forward(self, x, y, z, w, q):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1
        q = q * 2 - 1
        q = F.max_pool2d(q, 1)
        x = self.act_0(x)
        y = self.act_0(y)
        z = self.act_1(z)
        w = self.act_1(w)
        q = self.act_1(q)
        q = self.act_2(q)
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
    mod.save("test_nn_ELU.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_ELU.pt inputshape=[12],[12,64],[12,24,64],[12,24,32,64],[2,3,5,7]")

    # Float spelling is required by ncnn2mem binary parameter conversion.
    alphas = []
    with open("test_nn_ELU.ncnn.param") as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] != "ELU":
                continue
            values = [field[2:] for field in fields[4:] if field.startswith("0=")]
            if len(values) != 1:
                return False
            alpha = values[0]
            if "." not in alpha and "e" not in alpha.lower():
                return False
            alphas.append(float(alpha))
    if 2.0 not in alphas:
        return False

    # ncnn inference
    import test_nn_ELU_ncnn
    b = test_nn_ELU_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
