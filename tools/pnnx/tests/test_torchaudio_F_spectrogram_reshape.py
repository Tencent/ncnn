# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.register_buffer("window", torch.hann_window(32))

    def forward(self, x, y, z):
        x = torch.stft(x.reshape(-1, 128), n_fft=32, hop_length=8, window=self.window, return_complex=True)
        y = torch.stft(y.reshape(-1, 128), n_fft=32, hop_length=8, window=self.window, return_complex=True)
        z = torch.stft(z.reshape(-1, 128), n_fft=32, hop_length=8, window=self.window, return_complex=True)
        out0 = x.reshape(17, 17).abs().pow(2)
        out1 = y.reshape(1, 17, 17).abs().pow(2)
        out2 = z.reshape(1, 1, 17, 17).abs().pow(2)
        return out0, out1, out2

class ReshapeModel(nn.Module):
    def __init__(self):
        super(ReshapeModel, self).__init__()

        self.register_buffer("window", torch.hann_window(32))

    def forward(self, x):
        shape = x.size()
        packed = x.reshape(-1, shape[-1])
        spec = torch.stft(packed, n_fft=32, hop_length=8, window=self.window, return_complex=True)
        return spec.reshape(1, 1, 1, -1).abs().pow(2)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    inputs = (torch.rand(128), torch.rand(1, 128), torch.rand(1, 1, 128))
    inputs2 = (torch.rand(160), torch.rand(1, 160), torch.rand(1, 1, 160))

    # export torchscript with constant shape lists
    mod = torch.jit.trace(net, inputs)
    mod.save("test_torchaudio_F_spectrogram_reshape.pt")

    # torchscript to pnnx
    import os
    if os.system("../src/pnnx test_torchaudio_F_spectrogram_reshape.pt inputshape=[128],[1,128],[1,1,128]") != 0:
        return False

    # pnnx inference
    import test_torchaudio_F_spectrogram_reshape_pnnx
    converted = test_torchaudio_F_spectrogram_reshape_pnnx.Model()
    converted.eval()
    a = net(*inputs)
    b = converted(*inputs)
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape or not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    with open("test_torchaudio_F_spectrogram_reshape.pnnx.param") as f:
        if f.read().count("torchaudio.functional.spectrogram ") != 3:
            return False

    # flattening the frequency and time axes must keep the reshape
    net = ReshapeModel()
    net.eval()
    mod = torch.jit.trace(net, inputs[2])
    mod.save("test_torchaudio_F_spectrogram_reshape_flatten.pt")
    if os.system("../src/pnnx test_torchaudio_F_spectrogram_reshape_flatten.pt inputshape=[1,1,128] inputshape2=[1,1,160]") != 0:
        return False

    import test_torchaudio_F_spectrogram_reshape_flatten_pnnx
    converted = test_torchaudio_F_spectrogram_reshape_flatten_pnnx.Model()
    converted.eval()
    for args in (inputs, inputs2):
        a = net(args[2])
        b = converted(args[2])
        if a.shape != b.shape or not torch.allclose(a, b, 1e-4, 1e-4):
            return False
    with open("test_torchaudio_F_spectrogram_reshape_flatten.pnnx.param") as f:
        if "torchaudio.functional.spectrogram " in f.read():
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
