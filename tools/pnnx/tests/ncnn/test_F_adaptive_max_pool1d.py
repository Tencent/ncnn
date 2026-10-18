# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, q):
        x = F.adaptive_max_pool1d(x, output_size=7)
        x = F.adaptive_max_pool1d(x, output_size=1)
        q = F.adaptive_max_pool1d(q, output_size=7)
        q = F.adaptive_max_pool1d(q, output_size=1)
        return x, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    q = torch.rand(2, 12, 24)

    a = net(x, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_F_adaptive_max_pool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_adaptive_max_pool1d.pt inputshape=[1,12,24],[2,12,24]")

    # ncnn inference
    import test_F_adaptive_max_pool1d_ncnn
    b = test_F_adaptive_max_pool1d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        b0 = b0.reshape_as(a0)
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
