# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, q):
        out0 = torch.stft(x, n_fft=64, window=torch.hann_window(44), win_length=44, center=True, normalized=True, return_complex=True)
        out1 = torch.stft(x, n_fft=128, center=False, onesided=True, return_complex=True)
        out2 = torch.stft(y, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, center=True, pad_mode='constant', onesided=True, return_complex=True)
        out3 = torch.stft(y, n_fft=512, center=True, onesided=False, return_complex=True)
        out4 = torch.stft(q, n_fft=64, window=torch.hann_window(44), win_length=44, center=True, normalized=True, return_complex=True)
        out0 = torch.view_as_real(out0)
        out1 = torch.view_as_real(out1)
        out2 = torch.view_as_real(out2)
        out3 = torch.view_as_real(out3)
        out4 = torch.view_as_real(out4)
        return out0, out1, out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2560)
    y = torch.rand(1000)
    q = torch.rand(2, 2560)

    a = net(x, y, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, q))
    mod.save("test_torch_stft.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_stft.pt inputshape=[2560],[1000],[2,2560]")

    # ncnn inference
    import test_torch_stft_ncnn
    b = test_torch_stft_ncnn.test_inference()

    ts_ok = all(torch.allclose(a0, b0, 1e-3, 1e-3) for a0, b0 in zip(a, b))

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx_ncnn
    pt2_ok = test_pnnx_ncnn(net, (x, y, q), ["[2560]", "[1000]", "[2,2560]"], "test_torch_stft")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
