# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fold_0 = nn.Fold(output_size=22, kernel_size=3)
        self.fold_1 = nn.Fold(output_size=(17,18), kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        self.fold_2 = nn.Fold(output_size=(5,11), kernel_size=(2,3), stride=1, padding=(2,4), dilation=(1,2))

    def forward(self, x, y, z):
        x = self.fold_0(x)
        y = self.fold_1(y)
        z = self.fold_2(z)

        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 108, 400)
    y = torch.rand(1, 96, 190)
    z = torch.rand(1, 36, 120)

    a0, a1, a2 = net(x, y, z)

    return test_model_formats(net, (x, y, z), (a0, a1, a2), "test_nn_Fold")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
