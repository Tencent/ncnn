# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        out0 = torch.stft(x, n_fft=64, window=torch.hann_window(44), win_length=44, center=True, normalized=True, return_complex=True)
        out1 = torch.stft(x, n_fft=128, center=False, onesided=True, return_complex=True)
        out2 = torch.stft(y, n_fft=512, window=torch.hamming_window(256), win_length=256, hop_length=128, center=True, pad_mode='constant', onesided=True, return_complex=True)
        out3 = torch.stft(y, n_fft=512, center=True, onesided=False, return_complex=True)
        out0 = torch.view_as_real(out0)
        out1 = torch.view_as_real(out1)
        out2 = torch.view_as_real(out2)
        out3 = torch.view_as_real(out3)
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
    mod.save("test_torch_stft.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_stft.pt inputshape=[2560],[1000]")

    # ncnn inference
    import test_torch_stft_ncnn
    b = test_torch_stft_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
