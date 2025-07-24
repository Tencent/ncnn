# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.fft.hfft2(x, norm="backward")
        y = torch.fft.hfft2(y, dim=(1,3), norm="forward")
        z = torch.fft.hfft2(z, norm="ortho")
        return x, y, z

def test():
    if version.parse(torch.__version__) < version.parse('1.11'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 120, 120)
    y = torch.rand(1, 100, 2, 120)
    z = torch.rand(1, 20, 20)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_fft_hfft2.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_fft_hfft2.pt inputshape=[1,3,120,120],[1,100,2,120],[1,20,20]")

    # pnnx inference
    import test_torch_fft_hfft2_pnnx
    b = test_torch_fft_hfft2_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
