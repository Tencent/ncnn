# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class pixel_unshuffle(nn.Module):
    def __init__(self, scale=2):
        super(pixel_unshuffle, self).__init__()
        self.scale = scale
                                
    def forward(self, x):
        n, c, h, w = x.shape
        x = torch.reshape(x, (n, c, h // self.scale, self.scale, w // self.scale, self.scale))
        x = x.permute((0, 1, 3, 5, 2, 4))
        x = torch.reshape(x, (n, c * self.scale * self.scale, h // self.scale, w // self.scale))

        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.down_0 = pixel_unshuffle(2)
        self.down_1 = pixel_unshuffle(4)

    def forward(self, x):
        x = self.down_0(x)
        x = self.down_1(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128)

    a0 = net(x)

    return test_model_formats(net, (x,), a0, "test_pnnx_fuse_pixel_unshuffle")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
