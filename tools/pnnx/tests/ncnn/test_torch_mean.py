# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.mean(x, dim=0, keepdim=False)
        y = torch.mean(y, dim=(1,2), keepdim=False)
        z = torch.mean(z, dim=(0,3), keepdim=True)
        return x, y, z

class ModelBatch(nn.Module):
    def __init__(self):
        super(ModelBatch, self).__init__()

    def forward(self, x):
        x0 = torch.mean(x, dim=2, keepdim=False)
        x1 = torch.mean(x, dim=(1,3), keepdim=True)
        return x0, x1


class ModelEmptyDims(nn.Module):
    def forward(self, x):
        return (
            torch.mean(x, dim=[], keepdim=True),
            torch.mean(x, dim=(), keepdim=True),
            torch.mean(x, dim=[], keepdim=False),
            torch.mean(x, dim=(), keepdim=False),
            torch.mean(x, dim=(0, 1), keepdim=True),
            torch.mean(x, dim=(0, 1), keepdim=False),
        )

def test_empty_dims():
    net = ModelEmptyDims()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3)
    a = net(x)

    mod = torch.jit.trace(net, x)
    mod.save("test_torch_mean_empty_dims.pt")

    import os
    os.system("../../src/pnnx test_torch_mean_empty_dims.pt inputshape=[2,3]")

    import test_torch_mean_empty_dims_ncnn
    b = test_torch_mean_empty_dims_ncnn.test_inference()

    if len(b) != len(a):
        return False

    for i, (a0, b0) in enumerate(zip(a, b)):
        if i < 4:
            control = b[4] if i < 2 else b[5]
            if b0.shape != control.shape:
                return False
            if i < 2 and b0.shape != a0.shape:
                return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_mean.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_mean.pt inputshape=[3,16],[5,9,11],[8,5,9,10]")

    # ncnn inference
    import test_torch_mean_ncnn
    b = test_torch_mean_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return test_empty_dims() and test_batch()

def test_batch():
    net = ModelBatch()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_torch_mean_batch.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_mean_batch.pt inputshape=[2,3,5,7]")

    # ncnn inference
    import test_torch_mean_batch_ncnn
    b = test_torch_mean_batch_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
