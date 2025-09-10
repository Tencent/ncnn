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

    def forward(self, x, y, z, w):
        out0 = torchaudio.functional.inverse_spectrogram(x, n_fft=64, window=torch.hann_window(44), win_length=44, hop_length=16, pad=0, center=True, normalized='window', length=None)
        out1 = torchaudio.functional.inverse_spectrogram(y, n_fft=128, window=torch.hann_window(128), win_length=128, hop_length=3, pad=0, center=True, onesided=True, normalized=False, length=None)
        out2 = torchaudio.functional.inverse_spectrogram(z, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, pad=0, center=True, onesided=True, normalized='frame_length', length=None)
        out3 = torchaudio.functional.inverse_spectrogram(w, n_fft=512, window=torch.hamming_window(512), win_length=512, hop_length=128, pad=0, center=True, onesided=False, normalized=False, length=None)
        return out0, out1, out2, out3

def test():
    if version.parse(torchaudio.__version__) < version.parse('0.10.0'):
        return True

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
    mod.save("test_torchaudio_F_inverse_spectrogram.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torchaudio_F_inverse_spectrogram.pt inputshape=[3,33,161]c64,[1,65,77]c64,[257,8]c64,[512,4]c64")

    # pnnx inference
    import test_torchaudio_F_inverse_spectrogram_pnnx
    b = test_torchaudio_F_inverse_spectrogram_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
