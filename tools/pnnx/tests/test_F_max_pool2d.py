# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, torchscript_to_pnnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.max_pool2d(x, kernel_size=3)
        x = F.max_pool2d(x, kernel_size=4, stride=2, padding=2, dilation=1)
        x = F.max_pool2d(x, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1, return_indices=False, ceil_mode=False)
        x = F.max_pool2d(x, kernel_size=(4,5), stride=(1,2), padding=(1,2), dilation=1, return_indices=False, ceil_mode=True)
        x = F.max_pool2d(x, kernel_size=(2,3), stride=1, padding=1, dilation=(1,2), return_indices=False, ceil_mode=False)
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=True)
        x, indices1 = F.max_pool2d(x, kernel_size=2, padding=1, dilation=1, return_indices=True, ceil_mode=False)
        x, indices2 = F.max_pool2d(x, kernel_size=(5,4), stride=1, padding=2, dilation=1, return_indices=True, ceil_mode=False)

        y = F.max_pool2d(y, kernel_size=3)
        y = F.max_pool2d(y, kernel_size=4, stride=2, padding=2, dilation=1)
        y = F.max_pool2d(y, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1, return_indices=False, ceil_mode=False)
        y = F.max_pool2d(y, kernel_size=(4,5), stride=(1,2), padding=(1,2), dilation=1, return_indices=False, ceil_mode=True)
        y = F.max_pool2d(y, kernel_size=(2,3), stride=1, padding=1, dilation=(1,2), return_indices=False, ceil_mode=False)
        y = F.max_pool2d(y, kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=True)
        return x, indices1, indices2, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128, 127)
    y = torch.rand(12, 128, 127)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_max_pool2d.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_F_max_pool2d", "[1,12,128,127],[12,128,127]")

    # pnnx inference
    b = converted(x, y)

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    if not has_torch_export():
        return True

    converted = exported_program_to_pnnx(net, (x, y), "test_F_max_pool2d_pt2")
    c = converted(x, y)
    return all(torch.equal(a0, c0) for a0, c0 in zip(a, c))

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
