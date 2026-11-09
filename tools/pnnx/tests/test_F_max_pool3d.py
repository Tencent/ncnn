# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.max_pool3d(x, kernel_size=3)
        x = F.max_pool3d(x, kernel_size=4, stride=2, padding=2, dilation=1)
        x = F.max_pool3d(x, kernel_size=(1,2,3), stride=1, padding=(0,0,1), dilation=1, return_indices=False, ceil_mode=False)
        x = F.max_pool3d(x, kernel_size=(3,4,5), stride=(1,2,2), padding=(1,2,2), dilation=1, return_indices=False, ceil_mode=True)
        x = F.max_pool3d(x, kernel_size=(2,3,3), stride=1, padding=1, dilation=(1,2,2), return_indices=False, ceil_mode=False)
        x = F.max_pool3d(x, kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=True)
        x, indices = F.max_pool3d(x, kernel_size=(5,4,4), stride=1, padding=2, dilation=1, return_indices=True, ceil_mode=False)

        y = F.max_pool3d(y, kernel_size=3)
        y = F.max_pool3d(y, kernel_size=4, stride=2, padding=2, dilation=1)
        y = F.max_pool3d(y, kernel_size=(1,2,3), stride=1, padding=(0,0,1), dilation=1, return_indices=False, ceil_mode=False)
        y = F.max_pool3d(y, kernel_size=(3,4,5), stride=(1,2,2), padding=(1,2,2), dilation=1, return_indices=False, ceil_mode=True)
        y = F.max_pool3d(y, kernel_size=(2,3,3), stride=1, padding=1, dilation=(1,2,2), return_indices=False, ceil_mode=False)
        y = F.max_pool3d(y, kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=True)
        return x, indices, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 96, 128, 128)
    y = torch.rand(12, 96, 128, 128)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_max_pool3d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_max_pool3d.pt inputshape=[1,12,96,128,128],[12,96,128,128]")

    # pnnx inference
    import test_F_max_pool3d_pnnx
    b = test_F_max_pool3d_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_F_max_pool3d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_max_pool3d_export.pt2 fp16=0")

    # pnnx inference
    import test_F_max_pool3d_export_pnnx
    c = test_F_max_pool3d_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype or a0.shape != c0.shape:
            return False
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
