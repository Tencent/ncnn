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
        x, indices = F.adaptive_max_pool1d(x, output_size=7, return_indices=True)
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x, indices

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)

    a0, a1 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_adaptive_max_pool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_adaptive_max_pool1d.pt inputshape=[1,12,24]")

    # pnnx inference
    import test_F_adaptive_max_pool1d_pnnx
    b0, b1 = test_F_adaptive_max_pool1d_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_F_adaptive_max_pool1d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_adaptive_max_pool1d_export.pt2 fp16=0")

    # pnnx inference
    import test_F_adaptive_max_pool1d_export_pnnx
    c0, c1 = test_F_adaptive_max_pool1d_export_pnnx.test_inference()

    return torch.equal(a0, c0) and a1.dtype == c1.dtype and a1.shape == c1.shape and torch.equal(a1, c1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
