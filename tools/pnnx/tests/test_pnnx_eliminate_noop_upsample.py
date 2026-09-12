# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.up_0 = nn.Upsample(scale_factor=1, mode='nearest')
        self.up_1 = nn.Upsample(size=(12,52), mode='bicubic', align_corners=True)
        self.up_2 = nn.UpsamplingBilinear2d(scale_factor=(1,1))
        self.up_3 = nn.UpsamplingNearest2d(scale_factor=1)

    def forward(self, x):
        x = self.up_0(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = F.upsample(x, scale_factor=1, mode='bilinear')
        x = F.upsample(x, size=(12,52), mode='bicubic', align_corners=True)
        x = F.upsample_bilinear(x, scale_factor=1)
        x = F.upsample_nearest(x, size=(12,52))
        x = F.interpolate(x, scale_factor=(1,1), mode='nearest', recompute_scale_factor=True)
        x = F.interpolate(x, scale_factor=(1,1), mode='bicubic', align_corners=True, recompute_scale_factor=True)
        x = F.interpolate(x, size=(12,52), mode='bicubic', align_corners=False)
        x = F.relu(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 15, 12, 52)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_eliminate_noop_upsample.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_pnnx_eliminate_noop_upsample.pt inputshape=[1,15,12,52]")

    # pnnx inference
    import test_pnnx_eliminate_noop_upsample_pnnx
    b = test_pnnx_eliminate_noop_upsample_pnnx.test_inference()

    ts_ok = torch.equal(a, b)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,15,12,52]"], "test_pnnx_eliminate_noop_upsample")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
