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
        out0 = x.new_full((2,2), 1.5)
        out1 = x.new_full((3,), 3)
        out2 = x.new_full((4,5,6,7,8), -0.5)
        out3 = x.new_full((1,2,1), 0)
        out4 = x.new_full((3,3,3,3), 1, dtype=torch.long)
        return out0, out1, out2, out3, out4

class InputAnchoredModel(nn.Module):
    def __init__(self, model):
        super(InputAnchoredModel, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        z = x.sum() * 0
        return y[0] + z, y[1] + z, y[2] + z, y[3] + z, y[4] + z.to(dtype=y[4].dtype)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    wrapped = InputAnchoredModel(net)
    wrapped.eval()

    compare = lambda a0, b0: a0.shape == b0.shape
    return test_model_formats(wrapped, (x,), a, "test_Tensor_new_full", compare)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
