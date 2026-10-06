# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torchaudio

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = torchaudio.functional.spectrogram(x, n_fft=32, window=torch.hann_window(24), win_length=24, hop_length=8, pad=0, center=True, normalized="window", power=1)
        out1 = torchaudio.functional.spectrogram(x, n_fft=32, window=torch.hann_window(32), win_length=32, hop_length=8, pad=0, center=True, normalized=False, power=None)
        out2 = torchaudio.functional.spectrogram(y, n_fft=32, window=torch.hamming_window(24), win_length=24, hop_length=8, pad=8, center=False, pad_mode="constant", onesided=False, normalized="frame_length", power=2)
        out3 = torchaudio.functional.spectrogram(x, n_fft=32, window=torch.hann_window(24), win_length=24, hop_length=8, pad=0, center=True, normalized=False, power=2)
        out4 = torchaudio.functional.spectrogram(x.transpose(0, 1), n_fft=32, window=torch.hann_window(24), win_length=24, hop_length=8, pad=0, center=True, normalized=False, power=2)
        return out0, torch.view_as_real(out1), out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    inputs = (torch.rand(2, 3, 128), torch.rand(2, 3, 2, 128))
    inputs2 = (torch.rand(3, 4, 160), torch.rand(3, 2, 4, 160))

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save("test_torchaudio_F_spectrogram_batch.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_torchaudio_F_spectrogram_batch.pt inputshape=[2,3,128],[2,3,2,128] inputshape2=[3,4,160],[3,2,4,160] fp16=0") != 0:
        return False

    # ncnn inference
    import ncnn
    with ncnn.Net() as ncnn_net:
        ncnn_net.opt.num_threads = 2
        if ncnn_net.load_param("test_torchaudio_F_spectrogram_batch.ncnn.param") != 0 or ncnn_net.load_model("test_torchaudio_F_spectrogram_batch.ncnn.bin") != 0:
            return False
        for args in (inputs, inputs2):
            a = net(*args)
            with ncnn_net.create_extractor() as ex:
                for i, x in enumerate(args):
                    if ex.input("in%d" % i, ncnn.Mat(x.numpy(), batch_index=0).clone()) != 0:
                        return False
                for i, a0 in enumerate(a):
                    ret, out = ex.extract("out%d" % i)
                    if ret != 0:
                        return False
                    b0 = torch.from_numpy(out.numpy(batch_index=0).copy())
                    if a0.shape != b0.shape or not torch.allclose(a0, b0, 1e-3, 1e-3):
                        return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
