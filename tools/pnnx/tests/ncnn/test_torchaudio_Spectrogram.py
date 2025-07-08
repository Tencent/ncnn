# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.s0 = torchaudio.transforms.Spectrogram(n_fft=64, window_fn=torch.hann_window, win_length=44, hop_length=16, pad=0, center=True, normalized='window', power=1)
        if version.parse(torchaudio.__version__) < version.parse('0.11.0'):
            # return_complex=False with power=None, skip it
            self.s1 = torchaudio.transforms.Spectrogram(n_fft=128, window_fn=torch.hann_window, win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=1)
        else:
            self.s1 = torchaudio.transforms.Spectrogram(n_fft=128, window_fn=torch.hann_window, win_length=128, hop_length=3, pad=0, center=False, onesided=True, normalized=False, power=None)
        self.s2 = torchaudio.transforms.Spectrogram(n_fft=512, window_fn=torch.hamming_window, win_length=256, hop_length=128, pad=0, center=True, pad_mode='constant', onesided=True, normalized='frame_length', power=2)
        self.s3 = torchaudio.transforms.Spectrogram(n_fft=512, window_fn=torch.hamming_window, win_length=512, hop_length=128, pad=32, center=True, onesided=False, normalized=False, power=2)

    def forward(self, x, y):
        out0 = self.s0(x)
        out1 = self.s1(x)
        out2 = self.s2(y)
        out3 = self.s3(y)
        if version.parse(torchaudio.__version__) >= version.parse('0.11.0'):
            out1 = torch.view_as_real(out1)
        return out0, out1, out2, out3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2560)
    y = torch.rand(1000)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torchaudio_Spectrogram.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torchaudio_Spectrogram.pt inputshape=[2560],[1000]")

    # ncnn inference
    import test_torchaudio_Spectrogram_ncnn
    b = test_torchaudio_Spectrogram_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
