# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w0 = nn.Parameter(torch.zeros(1, 12, 52))
        self.w1 = nn.Parameter(torch.ones(1, 24, 52))

    def forward(self, x):
        x = torch.cat((x, self.w0), 1)
        x = torch.cat((x, ), 0)
        y = torch.cat((self.w1, ), 2)
        x = torch.cat((x, y), 0)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 52)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_eliminate_noop_cat.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_eliminate_noop_cat.pt inputshape=[1,12,52]")

    # pnnx inference
    import test_pnnx_eliminate_noop_cat_pnnx
    b = test_pnnx_eliminate_noop_cat_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
