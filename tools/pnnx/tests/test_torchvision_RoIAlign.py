# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.roi_align = torchvision.ops.RoIAlign(output_size=(3,3), spatial_scale=0.25, sampling_ratio=3, aligned=False)
        self.rois = nn.Parameter(torch.tensor([[0, 0, 10, 12, 20]], dtype=torch.float))

    def forward(self, x):
        x = self.roi_align(x, self.rois)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_torchvision_RoIAlign.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torchvision_RoIAlign.pt inputshape=[1,12,64,64]")

    # pnnx inference
    import test_torchvision_RoIAlign_pnnx
    b = test_torchvision_RoIAlign_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
