# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x[:,:12,1:14:2]
        x = x[...,1:]
        x = x[:,:,:x.size(2)-1]
        y = y[0:,1:,5:,3:]
        y = y[:,:,1:13:2,:14]
        y = y[:1,:y.size(1):,:,:]
        z = z[4:]
        z = z[:2,:,:,:,2:-2:3]
        z = z[:,:,:,z.size(3)-3:,:]
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 13, 26)
    y = torch.rand(1, 15, 19, 21)
    z = torch.rand(14, 18, 15, 19, 20)

    a = net(x, y, z)

    return test_model_formats(net, (x, y, z), a, "test_Tensor_slice")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
