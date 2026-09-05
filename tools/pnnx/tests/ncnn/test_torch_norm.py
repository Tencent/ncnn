# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x = torch.norm(x)
        y = torch.norm(y, p=2, dim=(1,2), keepdim=False)
        z = torch.norm(z, p=1, dim=None, keepdim=True)
        q = F.max_pool2d(q, 1)
        q0 = torch.norm(q, p=2, dim=(1,2), keepdim=False)
        q1 = torch.norm(q, p=1, dim=-1, keepdim=True)
        return x, y, z, q0, q1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_torch_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_norm.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_norm_ncnn
    b = test_torch_norm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
