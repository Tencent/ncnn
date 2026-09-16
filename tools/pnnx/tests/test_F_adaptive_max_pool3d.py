# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0, indices0 = F.adaptive_max_pool3d(x, output_size=(7,6,5), return_indices=True)
        out1 = F.adaptive_max_pool3d(x, output_size=1)
        out2 = F.adaptive_max_pool3d(x, output_size=(None,4,3))
        out3, indices3 = F.adaptive_max_pool3d(x, output_size=(5,None,None), return_indices=True)
        return out0, indices0, out1, out2, out3, indices3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 33, 64)

    a = net(x)

    return test_model_formats(net, (x,), a, "test_F_adaptive_max_pool3d")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
