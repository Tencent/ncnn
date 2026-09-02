# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = x.new_zeros((2,2))
        out1 = x.new_zeros(3)
        out2 = x.new_zeros((4,5,6,7,8))
        out3 = x.new_zeros((1,2,1))
        out4 = x.new_zeros((3,3,3,3), dtype=torch.long)
        return out0, out1, out2, out3, out4

class NoInputModel(nn.Module):
    def __init__(self, model, x):
        super(NoInputModel, self).__init__()
        self.model = model
        self.register_buffer("x", x)

    def forward(self):
        return self.model(self.x)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    wrapped = NoInputModel(net, x)
    wrapped.eval()

    compare = lambda a0, b0: a0.shape == b0.shape
    return test_model_formats(wrapped, (), a, "test_Tensor_new_zeros", compare)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
