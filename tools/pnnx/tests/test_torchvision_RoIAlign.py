# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pnnx_test_utils import test_model_formats

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

    return test_model_formats(
        net,
        (x,),
        a,
        "test_torchvision_RoIAlign",
        unsupported_by_torch_export="torchvision::roi_align",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
