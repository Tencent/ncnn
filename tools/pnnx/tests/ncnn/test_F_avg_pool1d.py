# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, q):
        x = F.avg_pool1d(x, kernel_size=3)
        x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=2)
        x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=(0), ceil_mode=False, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=5, stride=2, padding=(2), ceil_mode=True, count_include_pad=False)
        x = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        x = F.avg_pool1d(x, kernel_size=4, stride=1, padding=2, ceil_mode=False, count_include_pad=False)
        q = F.avg_pool1d(q, kernel_size=3)
        q = F.avg_pool1d(q, kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=True)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128)
    q = torch.rand(2, 3, 32)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_F_avg_pool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_avg_pool1d.pt inputshape=[1,12,128],[2,3,32]")

    # ncnn inference
    import test_F_avg_pool1d_ncnn
    b = test_F_avg_pool1d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
