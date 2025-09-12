# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x.clone()
        y = y.clone()
        z = z.clone()
        w = w.clone()
        xx = x[1]
        x[...,1] = x[...,-1] * 3
        x[:,:,3,:2].clamp_(0, 0.5)
        x[:,:,3,:2] = x[:,:,3,:2].exp_()
        xx[2:4,...] += 1
        x[:,:,:,:] = x[:,:,:,:] / 2
        y[...,-1,-5:-1] = y[...,-4,1:5] - 11
        z[:1] = z[-1:] * z[3:4]
        w[100:] = w[4:24] + 23
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18, 15, 19, 20)
    y = torch.rand(15, 19, 20)
    z = torch.rand(19, 20)
    w = torch.rand(120)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_Tensor_slice_copy.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_slice_copy.pt inputshape=[18,15,19,20],[15,19,20],[19,20],[120]")

    # ncnn inference
    import test_Tensor_slice_copy_ncnn
    b = test_Tensor_slice_copy_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
