# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=20, out_channels=8, kernel_size=1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 32, 32)

    a = net(x)

    program = torch.export.export(net, (x,))
    torch.export.save(program, "test_nn_Conv2d.pt2")

    from pnnxutil import run_pnnx
    run_pnnx("test_nn_Conv2d.pt2", "inputshape=[1,12,32,32]")

    import test_nn_Conv2d_pnnx
    b = test_nn_Conv2d_pnnx.test_inference()

    return torch.allclose(a, b, 1e-3, 1e-3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
