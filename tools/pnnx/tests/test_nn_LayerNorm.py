# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LayerNorm(64)
        self.ln_0.weight = nn.Parameter(torch.rand(64))
        self.ln_0.bias = nn.Parameter(torch.rand(64))
        self.ln_1 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)

    def forward(self, x, y, z):
        x = self.ln_0(x)
        x = self.ln_1(x)

        y = self.ln_0(y)
        y = self.ln_1(y)

        z = self.ln_0(z)
        z = self.ln_1(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    z = torch.rand(1, 12, 16, 24, 64)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_LayerNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_LayerNorm.pt inputshape=[1,24,64],[1,12,24,64],[1,12,16,24,64]")

    # pnnx inference
    import test_nn_LayerNorm_pnnx
    b0, b1, b2 = test_nn_LayerNorm_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_nn_LayerNorm_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_LayerNorm_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_LayerNorm_export_pnnx
    c0, c1, c2 = test_nn_LayerNorm_export_pnnx.test_inference()

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
