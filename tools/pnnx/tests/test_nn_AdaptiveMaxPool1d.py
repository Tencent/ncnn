# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
