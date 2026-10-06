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
    inputs = (torch.rand(128), torch.rand(1, 128), torch.rand(1, 1, 128))
    inputs2 = (torch.rand(160), torch.rand(1, 160), torch.rand(1, 1, 160))

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save("test_torchaudio_MelSpectrogram.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_torchaudio_MelSpectrogram.pt inputshape=[128],[1,128],[1,1,128] inputshape2=[160],[1,160],[1,1,160] fp16=0") != 0:
        return False

    # ncnn inference
    import ncnn
    with ncnn.Net() as ncnn_net:
        ncnn_net.opt.num_threads = 2
        if ncnn_net.load_param("test_torchaudio_MelSpectrogram.ncnn.param") != 0 or ncnn_net.load_model("test_torchaudio_MelSpectrogram.ncnn.bin") != 0:
            return False
        for args in (inputs, inputs2):
            expected = net(*args)
            with ncnn_net.create_extractor() as ex:
                for i, x in enumerate(args):
                    if ex.input("in%d" % i, ncnn.Mat(x.numpy(), batch_index=233 if x.dim() == 1 else 0).clone()) != 0:
                        return False
                for i, a in enumerate(expected):
                    ret, out = ex.extract("out%d" % i)
                    if ret != 0:
                        return False
                    b = torch.from_numpy(out.numpy(batch_index=233 if a.dim() == 2 else 0).copy())
                    if a.shape != b.shape or not torch.allclose(a, b, 1e-3, 1e-3):
                        return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
