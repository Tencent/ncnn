# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.upsample_bilinear(x, size=(12,12))
        x = F.upsample_bilinear(x, scale_factor=2)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_upsample_bilinear.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_upsample_bilinear.pt inputshape=[1,12,24,64]")

    # pnnx inference
    import test_F_upsample_bilinear_pnnx
    b = test_F_upsample_bilinear_pnnx.test_inference()

    if not (torch.equal(a, b)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_F_upsample_bilinear_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_upsample_bilinear_export.pt2 fp16=0")

    # pnnx inference
    import test_F_upsample_bilinear_export_pnnx
    c = test_F_upsample_bilinear_export_pnnx.test_inference()

    return torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
