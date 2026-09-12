# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    os.system(os.path.join("..", "src", "pnnx") + " test_F_adaptive_max_pool1d.pt inputshape=[1,12,24]")

    # pnnx inference
    import test_F_adaptive_max_pool1d_pnnx
    b0, b1 = test_F_adaptive_max_pool1d_pnnx.test_inference()

    ts_ok = torch.equal(a0, b0) and torch.equal(a1, b1)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,12,24]"], "test_F_adaptive_max_pool1d")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
