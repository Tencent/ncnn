# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

    def forward(self, x, y, z, w):
        out0 = torch.istft(x, n_fft=64, center=True, normalized=True, return_complex=False)
        out1 = torch.istft(y, n_fft=128, center=False, onesided=True, return_complex=False)
        out2 = torch.istft(z, n_fft=512, center=True, onesided=True, return_complex=False)
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
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
