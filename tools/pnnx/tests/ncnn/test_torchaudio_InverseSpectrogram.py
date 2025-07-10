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

        self.s0 = torchaudio.transforms.InverseSpectrogram(n_fft=64, window_fn=torch.hann_window, win_length=44, hop_length=16, pad=0, center=True, normalized='window')
        self.s1 = torchaudio.transforms.InverseSpectrogram(n_fft=128, window_fn=torch.hann_window, win_length=128, hop_length=3, pad=0, center=True, onesided=True, normalized=False)
        self.s2 = torchaudio.transforms.InverseSpectrogram(n_fft=512, window_fn=torch.hamming_window, win_length=256, hop_length=128, pad=0, center=True, onesided=True, normalized='frame_length')
        self.s3 = torchaudio.transforms.InverseSpectrogram(n_fft=1024, window_fn=torch.hamming_window, win_length=512, hop_length=128, pad=0, center=True, onesided=True, normalized=False)

    def forward(self, x, y, z, w):
        x = torch.view_as_complex(x)
        y = torch.view_as_complex(y)
        z = torch.view_as_complex(z)
        w = torch.view_as_complex(w)
        out0 = self.s0(x)
        out1 = self.s1(y)
        out2 = self.s2(z)
        out3 = self.s3(w)
        return out0, out1, out2, out3

def test():
    if version.parse(torchaudio.__version__) < version.parse('0.10.0'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(33, 161, 2)
    y = torch.rand(65, 77, 2)
    z = torch.rand(257, 8, 2)
    w = torch.rand(513, 4, 2)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torchaudio_InverseSpectrogram.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torchaudio_InverseSpectrogram.pt inputshape=[33,161,2],[65,77,2],[257,8,2],[513,4,2]")

    # ncnn inference
    import test_torchaudio_InverseSpectrogram_ncnn
    b = test_torchaudio_InverseSpectrogram_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
