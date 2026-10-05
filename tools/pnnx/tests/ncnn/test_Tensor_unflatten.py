# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x = x.unflatten(dim=0, sizes=(2,1,2,-1))
        y = y.unflatten(dim=1, sizes=(3,4))
        z = z.unflatten(dim=-2, sizes=(3,-1))
        q = F.max_pool2d(q, 3, stride=1, padding=1)
        q = q.unflatten(dim=1, sizes=(3,4))
        return x, y, z, q

class ModelMiddleBatch(nn.Module):
    def __init__(self):
        super(ModelMiddleBatch, self).__init__()

    def forward(self, x):
        x = x.unflatten(dim=0, sizes=(3,2))
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x

def test():
    if version.parse(torch.__version__) < version.parse('1.13'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(9, 12)
    z = torch.rand(8, 9, 10)
    q = torch.rand(2, 12, 5, 7)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_Tensor_unflatten.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_unflatten.pt inputshape=[16],[9,12],[8,9,10],[2,12,5,7]")

    # ncnn inference
    import test_Tensor_unflatten_ncnn
    b = test_Tensor_unflatten_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    net = ModelMiddleBatch()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, (x,))
    mod.save("test_Tensor_unflatten_middle_batch.pt")

    # torchscript to pnnx
    os.system("../../src/pnnx test_Tensor_unflatten_middle_batch.pt inputshape=[6,5,7]")

    import test_Tensor_unflatten_middle_batch_ncnn
    b = test_Tensor_unflatten_middle_batch_ncnn.test_inference()

    if not torch.equal(a, b):
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
