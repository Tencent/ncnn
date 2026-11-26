# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, r, s):
        x = F.pad(x, (3,4), mode='constant', value=1.3)
        x = F.pad(x, (2,2))

        y = F.pad(y, (5,6), mode='reflect')
        y = F.pad(y, (2,1), mode='replicate')
        y = F.pad(y, (3,4), mode='constant', value=1.3)
        y = F.pad(y, (1,1))

        z = F.pad(z, (3,4,3,4), mode='reflect')
        z = F.pad(z, (2,1,2,0), mode='replicate')
        z = F.pad(z, (1,0,2,0), mode='constant', value=1.3)
        z = F.pad(z, (3,3,3,3))

        #w = F.pad(w, (1,2,3,4,5,6), mode='reflect')
        w = F.pad(w, (5,0,1,2,0,2), mode='replicate')
        w = F.pad(w, (0,2,2,1,3,4), mode='constant', value=1.3)
        w = F.pad(w, (2,2,2,2,2,2))

        r = F.pad(r, (-1,2), mode='constant', value=9.25)

        s = F.pad(s, (-1,2,1,-1,-2,3), mode='constant', value=1.3)

        return x, y, z, w, r, s

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)
    y = torch.rand(12, 2, 16)
    z = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 5, 7, 9, 11)
    r = torch.rand(8)
    s = torch.rand(2, 3, 5, 7)

    a0, a1, a2, a3, a4, a5 = net(x, y, z, w, r, s)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, r, s))
    mod.save("test_F_pad.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_pad.pt inputshape=[1,16],[12,2,16],[1,3,12,16],[1,5,7,9,11],[8],[2,3,5,7]")

    # pnnx inference
    import test_F_pad_pnnx
    b0, b1, b2, b3, b4, b5 = test_F_pad_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3) and torch.equal(a4, b4) and torch.equal(a5, b5)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z, w, r, s))
    torch.export.save(ep, "test_F_pad_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_pad_export.pt2 fp16=0")

    # pnnx inference
    import test_F_pad_export_pnnx
    c0, c1, c2, c3, c4, c5 = test_F_pad_export_pnnx.test_inference()

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2) and torch.equal(a3, c3) and torch.equal(a4, c4) and torch.equal(a5, c5)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
