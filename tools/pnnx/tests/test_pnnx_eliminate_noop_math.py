# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w0 = nn.Parameter(torch.zeros(1, 12, 52))
        self.w1 = nn.Parameter(torch.ones(1, 12, 52))
        self.w2 = nn.Parameter(torch.ones(1, 12, 52))

    def forward(self, x):
        x = x + 0
        x = x * 1 / 1
        x = 0 + 1 * x
        x = x + self.w0 * self.w1
        x = x * self.w2
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 52)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_eliminate_noop_math.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_pnnx_eliminate_noop_math.pt inputshape=[1,12,52]")

    # pnnx inference
    import test_pnnx_eliminate_noop_math_pnnx
    b = test_pnnx_eliminate_noop_math_pnnx.test_inference()

    ts_ok = torch.equal(a, b)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,12,52]"], "test_pnnx_eliminate_noop_math")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
