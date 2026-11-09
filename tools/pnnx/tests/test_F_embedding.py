# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w1 = nn.Parameter(torch.rand(10, 128))

    def forward(self, x, w0, y):
        x = F.embedding(x, w0)
        y = F.embedding(y, self.w1)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (1, 13), dtype=torch.int)
    w0 = torch.rand(10, 128)
    y = torch.randint(10, (1, 11), dtype=torch.int)

    a0, a1 = net(x, w0, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, w0, y))
    mod.save("test_F_embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_embedding.pt inputshape=[1,13]i32,[10,128],[1,11]i32")

    # pnnx inference
    import test_F_embedding_pnnx
    b0, b1 = test_F_embedding_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, w0, y))
    torch.export.save(ep, "test_F_embedding_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_embedding_export.pt2 fp16=0")

    # pnnx inference
    import test_F_embedding_export_pnnx
    c0, c1 = test_F_embedding_export_pnnx.test_inference()

    return torch.equal(a0, c0) and torch.equal(a1, c1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
