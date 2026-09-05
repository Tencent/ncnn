# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out = torch.mv(x, y)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3)
    y = torch.rand(3)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_mv.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_torch_mv.pt inputshape=[2,3],[3]")

    # pnnx inference
    import test_torch_mv_pnnx
    b = test_torch_mv_pnnx.test_inference()

    ts_ok = torch.equal(a, b)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y), ["[2,3]", "[3]"], "test_torch_mv")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
