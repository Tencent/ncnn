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
        x = F.avg_pool2d(x, kernel_size=3)
        x = F.avg_pool2d(x, kernel_size=4, stride=2, padding=2)
        x = F.avg_pool2d(x, kernel_size=(1,3), stride=1, padding=(0,1), ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=(4,5), stride=(1,2), padding=(1,2), ceil_mode=True, count_include_pad=False)
        x = F.avg_pool2d(x, kernel_size=(5,3), stride=(2,1), padding=1, ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=(5,4), stride=1, padding=2, ceil_mode=False, count_include_pad=False, divisor_override=18)

        y = F.avg_pool2d(y, kernel_size=3)
        y = F.avg_pool2d(y, kernel_size=4, stride=2, padding=2)
        y = F.avg_pool2d(y, kernel_size=(1,3), stride=1, padding=(0,1), ceil_mode=False, count_include_pad=True)
        y = F.avg_pool2d(y, kernel_size=(4,5), stride=(1,2), padding=(1,2), ceil_mode=True, count_include_pad=False)
        y = F.avg_pool2d(y, kernel_size=(5,3), stride=(2,1), padding=1, ceil_mode=False, count_include_pad=True)
        y = F.avg_pool2d(y, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        y = F.avg_pool2d(y, kernel_size=(5,4), stride=1, padding=2, ceil_mode=False, count_include_pad=False, divisor_override=18)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128, 127)
    y = torch.rand(12, 128, 127)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_avg_pool2d.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_F_avg_pool2d", "[1,12,128,127],[12,128,127]")

    # pnnx inference
    b = converted(x, y)

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    if not has_torch_export():
        return True

    converted = exported_program_to_pnnx(net, (x, y), "test_F_avg_pool2d_pt2")
    c = converted(x, y)
    return all(torch.equal(a0, c0) for a0, c0 in zip(a, c))

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
