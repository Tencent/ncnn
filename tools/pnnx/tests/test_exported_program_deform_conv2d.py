# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        spatial_params = dict(kernel_size=(3,3), stride=(2,1), padding=(1,0), dilation=(1,2))
        self.offset = nn.Conv2d(in_channels=12, out_channels=2*3*3, **spatial_params)
        self.deform = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, **spatial_params)
        self.mask = nn.Conv2d(in_channels=12, out_channels=3*3, **spatial_params)
        self.modulated_deform = torchvision.ops.DeformConv2d(in_channels=12, out_channels=16, **spatial_params)

    def forward(self, x):
        offset = self.offset(x)
        out0 = self.deform(x, offset)
        mask = F.sigmoid(self.mask(x))
        out1 = self.modulated_deform(x, offset, mask)
        return out0, out1


def test():
    if not has_torch_export():
        return True

    torch.manual_seed(0)
    model = Model().eval()
    x = torch.rand(1, 12, 16, 18)
    expected = model(x)
    converted = exported_program_to_pnnx(model, x, "test_exported_program_deform_conv2d")
    actual = converted(x)

    return all(torch.allclose(a, b, rtol=1e-4, atol=1e-5) for a, b in zip(expected, actual))


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
