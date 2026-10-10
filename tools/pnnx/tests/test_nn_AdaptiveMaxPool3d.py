# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool_0 = nn.AdaptiveMaxPool3d(output_size=(7,6,5), return_indices=True)
        self.pool_1 = nn.AdaptiveMaxPool3d(output_size=1)
        self.pool_2 = nn.AdaptiveMaxPool3d(output_size=(None,4,3))
        self.pool_3 = nn.AdaptiveMaxPool3d(output_size=(5,None,None), return_indices=True)

    def forward(self, x):
        out0, indices0 = self.pool_0(x)
        out1 = self.pool_1(x)
        out2 = self.pool_2(x)
        out3, indices3 = self.pool_3(x)
        return out0, indices0, out1, out2, out3, indices3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 128, 13, 13, 13)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AdaptiveMaxPool3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_AdaptiveMaxPool3d.pt inputshape=[1,128,13,13,13]")

    # pnnx inference
    import test_nn_AdaptiveMaxPool3d_pnnx
    b = test_nn_AdaptiveMaxPool3d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
