# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = x[:12,1:14:1]
        x = x[...,1:]
        x = x[:,:x.size(1)-1]
        y = y[1:,5:,3:]
        y = y[:,1:13:1,:14]
        y = y[:y.size(1):,:,:]
        z = z[4:]
        z = z[:2,:,:,2:-2:1]
        z = z[:,:,z.size(2)-3:,:]
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(13, 26)
    y = torch.rand(15, 19, 21)
    z = torch.rand(18, 15, 19, 20)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_slice.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_slice.pt inputshape=[13,26],[15,19,21],[18,15,19,20]")

    # ncnn inference
    import test_Tensor_slice_ncnn
    b = test_Tensor_slice_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
