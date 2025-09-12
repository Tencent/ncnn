# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        out0 = torch.istft(x, n_fft=64, window=torch.hann_window(44), win_length=44, center=True, normalized=True, return_complex=False)
        out1 = torch.istft(y, n_fft=128, center=False, onesided=True, return_complex=False)
        out2 = torch.istft(z, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, center=True, onesided=True, return_complex=False)
        out3 = torch.istft(w, n_fft=512, center=False, onesided=False, return_complex=True)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 33, 161, dtype=torch.complex64)
    y = torch.rand(1, 65, 77, dtype=torch.complex64)
    z = torch.rand(257, 8, dtype=torch.complex64)
    w = torch.rand(512, 4, dtype=torch.complex64)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_istft.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_istft.pt inputshape=[3,33,161]c64,[1,65,77]c64,[257,8]c64,[512,4]c64")

    # pnnx inference
    import test_torch_istft_pnnx
    b = test_torch_istft_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
