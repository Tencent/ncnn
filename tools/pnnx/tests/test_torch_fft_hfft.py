# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.fft.hfft(x, norm="backward")
        y = torch.fft.hfft(y, dim=(1), norm="forward")
        z = torch.fft.hfft(z, norm="ortho")
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 120, 120)
    y = torch.rand(1, 100, 2, 120)
    z = torch.rand(1, 20, 20)

    a = net(x, y, z)
    return test_model_formats(
        net,
        (x, y, z),
        a,
        "test_torch_fft_hfft",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
