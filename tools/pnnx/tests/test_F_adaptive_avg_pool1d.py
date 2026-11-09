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
        x = F.adaptive_avg_pool1d(x, output_size=7)
        x = F.adaptive_avg_pool1d(x, output_size=1)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_adaptive_avg_pool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_adaptive_avg_pool1d.pt inputshape=[1,12,24]")

    # pnnx inference
    import test_F_adaptive_avg_pool1d_pnnx
    b = test_F_adaptive_avg_pool1d_pnnx.test_inference()

    if not (torch.equal(a, b)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_F_adaptive_avg_pool1d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_adaptive_avg_pool1d_export.pt2 fp16=0")

    # pnnx inference
    import test_F_adaptive_avg_pool1d_export_pnnx
    c = test_F_adaptive_avg_pool1d_export_pnnx.test_inference()

    return torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
