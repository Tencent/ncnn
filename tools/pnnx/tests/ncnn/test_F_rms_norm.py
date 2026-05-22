# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w3 = nn.Parameter(torch.rand(24))
        self.w4 = nn.Parameter(torch.rand(12, 16))
        self.w5 = nn.Parameter(torch.rand(3, 12, 16))

    def forward(self, x, y, w):
        x = F.rms_norm(x, (24,), self.w3)

        y = F.rms_norm(y, (16,), None)
        z = F.rms_norm(y, (12,16), self.w4, eps=1e-3)
        w0 = F.rms_norm(w, (16,), None)
        w1 = F.rms_norm(w, (12,16), self.w4, eps=1e-3)
        w2 = F.rms_norm(w, (3,12,16), self.w5, eps=1e-4)
        return x, y, z, w0, w1, w2

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(1, 3, 12, 16)
    w = torch.rand(1, 2, 3, 12, 16)

    a = net(x, y, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, w))
    mod.save("test_F_rms_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_rms_norm.pt inputshape=[1,12,24],[1,3,12,16],[1,2,3,12,16]")

    # ncnn inference
    import test_F_rms_norm_ncnn
    b = test_F_rms_norm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
