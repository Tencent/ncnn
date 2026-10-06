# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torchaudio

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(n_fft=32, win_length=24, hop_length=8, n_mels=8)
        self.magnitude = torchaudio.transforms.MelSpectrogram(n_fft=32, win_length=24, hop_length=8, n_mels=8, power=1)
        self.normalized = torchaudio.transforms.MelSpectrogram(n_fft=32, win_length=24, hop_length=8, n_mels=8, normalized=True)

    def forward(self, x, y, z):
        out0 = self.mel(x)
        out1 = self.mel(y)
        out2 = self.mel(z)
        out3 = self.magnitude(x)
        out4 = self.magnitude(y)
        out5 = self.magnitude(z)
        out6 = self.normalized(x)
        out7 = self.normalized(y)
        out8 = self.normalized(z)
        return out0, out1, out2, out3, out4, out5, out6, out7, out8

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    inputs = (torch.rand(128), torch.rand(1, 128), torch.rand(2, 3, 128))
    inputs2 = (torch.rand(160), torch.rand(1, 160), torch.rand(2, 3, 160))

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save("test_torchaudio_MelSpectrogram.pt")

    # torchscript to pnnx
    import os
    if os.system("../src/pnnx test_torchaudio_MelSpectrogram.pt inputshape=[128],[1,128],[2,3,128] inputshape2=[160],[1,160],[2,3,160]") != 0:
        return False

    # pnnx inference
    import test_torchaudio_MelSpectrogram_pnnx
    converted = test_torchaudio_MelSpectrogram_pnnx.Model()
    converted.eval()
    for args in (inputs, inputs2):
        a = net(*args)
        b = converted(*args)
        for a0, b0 in zip(a, b):
            if a0.shape != b0.shape or not torch.allclose(a0, b0, 1e-4, 1e-4):
                return False

    with open("test_torchaudio_MelSpectrogram.pnnx.param") as f:
        if f.read().count("torchaudio.functional.spectrogram ") != 9:
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
