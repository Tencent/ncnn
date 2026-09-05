# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.lp_pool2d(x, norm_type=2, kernel_size=3)
        x = F.lp_pool2d(x, norm_type=2, kernel_size=4, stride=2)
        x = F.lp_pool2d(x, norm_type=1, kernel_size=(1,3), stride=1, ceil_mode=False)
        x = F.lp_pool2d(x, norm_type=1, kernel_size=(4,5), stride=(1,2), ceil_mode=True)
        x = F.lp_pool2d(x, norm_type=1.2, kernel_size=(5,3), stride=(2,1), ceil_mode=False)
        x = F.lp_pool2d(x, norm_type=0.5, kernel_size=2, stride=1, ceil_mode=True)
        x = F.lp_pool2d(x, norm_type=0.1, kernel_size=(5,4), stride=1, ceil_mode=False)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128, 128)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_lp_pool2d.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_F_lp_pool2d.pt inputshape=[1,12,128,128]")

    # pnnx inference
    import test_F_lp_pool2d_pnnx
    b = test_F_lp_pool2d_pnnx.test_inference()

    ts_ok = torch.equal(a, b)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,12,128,128]"], "test_F_lp_pool2d")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
