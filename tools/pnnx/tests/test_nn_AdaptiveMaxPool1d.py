# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveMaxPool1d(output_size=(7), return_indices=True)
        self.pool_1 = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x, indices = self.pool_0(x)
        x = self.pool_1(x)
        return x, indices

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13)

    a0, a1 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AdaptiveMaxPool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AdaptiveMaxPool1d.pt inputshape=[1,128,13]")

    # pnnx inference
    import test_nn_AdaptiveMaxPool1d_pnnx
    b0, b1 = test_nn_AdaptiveMaxPool1d_pnnx.test_inference()

    if not (torch.equal(a0, b0) and torch.equal(a1, b1)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_nn_AdaptiveMaxPool1d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_AdaptiveMaxPool1d_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_AdaptiveMaxPool1d_export_pnnx
    c0, c1 = test_nn_AdaptiveMaxPool1d_export_pnnx.test_inference()

    if a1.dtype != c1.dtype:
        return False
    return torch.equal(a0, c0) and torch.equal(a1, c1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
