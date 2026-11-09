# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = F.dropout(x, training=False)
        y = F.dropout(y, training=False)
        z = F.dropout(z, p=0.6, training=False)
        w = F.dropout(w, p=0.1, training=False)
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

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_F_dropout.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_dropout.pt inputshape=[1,16],[12,2,16],[1,3,12,16],[1,5,7,9,11]")

    # pnnx inference
    import test_F_dropout_pnnx
    b0, b1, b2, b3 = test_F_dropout_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z, w))
    torch.export.save(ep, "test_F_dropout_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_dropout_export.pt2 fp16=0")

    # pnnx inference
    import test_F_dropout_export_pnnx
    c0, c1, c2, c3 = test_F_dropout_export_pnnx.test_inference()

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2) and torch.equal(a3, c3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
