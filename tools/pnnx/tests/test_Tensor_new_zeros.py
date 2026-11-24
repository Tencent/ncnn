# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

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

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_new_zeros.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_new_zeros.pt inputshape=[1,16]")

    # pnnx inference
    import test_Tensor_new_zeros_pnnx
    b = test_Tensor_new_zeros_pnnx.test_inference()

    # test shape only for uninitialized data
    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not a0.shape == b0.shape:
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_Tensor_new_zeros_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_Tensor_new_zeros_export.pt2 fp16=0")

    # pnnx inference
    import test_Tensor_new_zeros_export_pnnx
    c = test_Tensor_new_zeros_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype or not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
