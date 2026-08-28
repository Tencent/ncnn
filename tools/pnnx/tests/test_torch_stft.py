# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import convert_and_import

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = torch.stft(x, n_fft=64, window=torch.hann_window(44), win_length=44, center=True, normalized=True, return_complex=True)
        out1 = torch.stft(x, n_fft=128, center=False, onesided=True, return_complex=True)
        out2 = torch.stft(y, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, center=True, pad_mode='constant', onesided=True, return_complex=True)
        out3 = torch.stft(y, n_fft=512, center=True, onesided=False, return_complex=True)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 2560)
    y = torch.rand(1000)

    a = net(x, y)

    mod = convert_and_import(
        net,
        (x, y),
        "test_torch_stft",
        pnnx_args=("inputshape=[3,2560],[1000]",),
    )

    b = mod.test_inference()

    passed = True
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            passed = False
    return passed

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
