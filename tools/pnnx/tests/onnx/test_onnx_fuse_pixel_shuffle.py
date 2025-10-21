# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class pixel_shuffle(nn.Module):
    def __init__(self, upscale=2):
        super(pixel_shuffle, self).__init__()
        self.upscale = upscale
                                
    def forward(self, x):
        n, c, h, w = x.shape
        upscale = self.upscale
        c //= upscale ** 2
        x = x.view(n, c, upscale, upscale, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(n, c, h * upscale, w * upscale)

        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.up_0 = pixel_shuffle(2)
        self.up_1 = pixel_shuffle(4)

    def forward(self, x):
        x = self.up_0(x)
        x = F.relu(x)
        x = self.up_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64, 15, 15)

    a0 = net(x)

    # export onnx
    torch.onnx.export(net, (x,), "test_pnnx_fuse_pixel_shuffle.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_pnnx_fuse_pixel_shuffle.onnx inputshape=[1,64,15,15]")

    # pnnx inference
    import test_pnnx_fuse_pixel_shuffle_pnnx
    b0 = test_pnnx_fuse_pixel_shuffle_pnnx.test_inference()

    return torch.equal(a0, b0)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
