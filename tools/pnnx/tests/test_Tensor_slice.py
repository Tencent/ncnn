# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

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

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_slice.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_slice.pt inputshape=[1,13,26],[1,15,19,21],[14,18,15,19,20]")

    # pnnx inference
    import test_Tensor_slice_pnnx
    b = test_Tensor_slice_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y, z))
    torch.export.save(ep, "test_Tensor_slice_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_Tensor_slice_export.pt2 fp16=0")

    # pnnx inference
    import test_Tensor_slice_export_pnnx
    c = test_Tensor_slice_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
