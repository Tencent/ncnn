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
        out0 = F.adaptive_avg_pool3d(x, output_size=(7,6,5))
        out1 = F.adaptive_avg_pool3d(x, output_size=1)
        out2 = F.adaptive_avg_pool3d(x, output_size=(None,4,3))
        out3 = F.adaptive_avg_pool3d(x, output_size=(5,None,None))
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 33, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_adaptive_avg_pool3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_adaptive_avg_pool3d.pt inputshape=[1,12,24,33,64]")

    # pnnx inference
    import test_F_adaptive_avg_pool3d_pnnx
    b = test_F_adaptive_avg_pool3d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_F_adaptive_avg_pool3d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_adaptive_avg_pool3d_export.pt2 fp16=0")

    # pnnx inference
    import test_F_adaptive_avg_pool3d_export_pnnx
    c = test_F_adaptive_avg_pool3d_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
