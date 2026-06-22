# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 3, 1)
        self.conv3 = nn.Conv2d(3, 4, 1)
        self.conv4 = nn.Conv2d(3, 4, 1)
        self.conv5 = nn.Conv2d(3, 4, 1)

    def forward(self, x, y, z, w, v):
        x = x.permute(1, 0)
        x = x.permute(0, 1)
        y = y.permute(2, 1, 0)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 3, 0, 2)
        z = z.permute(2, 0, 3, 1)
        ww = self.conv(w)
        w0 = ww.permute(1, 0, 2, 3).reshape(8, 5, 7)
        w1 = ww.permute(1, 0, 3, 2).reshape(8, 7, 5)
        w2 = self.conv3(w)
        w2 = w2.permute(1, 0, 2, 3)
        w2 = torch.clone(w2)
        w2 = w2.permute(0, 2, 3, 1)
        w2 = torch.clone(w2)
        w2 = w2.permute(0, 3, 1, 2).reshape(8, 5, 7)
        w3 = self.conv4(w)
        w3 = w3.permute(1, 0, 2, 3)
        w4 = self.conv5(w)
        w4 = w4.permute(2, 1, 0, 3)
        w4 = F.relu(w4)
        v = v.reshape(4, 2, 5, 7)
        v = v.permute(1, 0, 2, 3)
        v = torch.clone(v)
        v = v.permute(0, 1, 3, 2)
        v = self.conv2(v)
        x = F.relu(x)
        y = F.relu(y)
        z = F.relu(z)
        return x, y, z, w0, w1, w2, w3, w4, v

class ModelMultiConsumer(nn.Module):
    def __init__(self):
        super(ModelMultiConsumer, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(1, 0, 2, 3)
        y = x.reshape(8, 5, 7)
        z = x.permute(0, 1, 3, 2)
        return y, z

class ModelComputeBarrier(nn.Module):
    def __init__(self):
        super(ModelComputeBarrier, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(1, 0, 2, 3)
        x = F.relu(x)
        x = x.reshape(8, 5, 7)
        return x

class ModelOutputBoundary(nn.Module):
    def __init__(self):
        super(ModelOutputBoundary, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        return x.permute(2, 0, 1, 3)

def test_conservative():
    import os

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)

    net = ModelMultiConsumer()
    net.eval()
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_permute_multi_consumer.pt")
    if os.system("../../src/pnnx test_Tensor_permute_multi_consumer.pt inputshape=[2,3,5,7]") != 0:
        return False

    with open("test_Tensor_permute_multi_consumer.ncnn.param") as f:
        lines = f.readlines()
        if any(line.startswith("Reshape") and ("12=1" in line or "12=2" in line) for line in lines):
            return False

    net = ModelComputeBarrier()
    net.eval()
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_permute_compute_barrier.pt")
    if os.system("../../src/pnnx test_Tensor_permute_compute_barrier.pt inputshape=[2,3,5,7]") != 0:
        return False

    with open("test_Tensor_permute_compute_barrier.ncnn.param") as f:
        lines = f.readlines()
        if any(line.startswith("Reshape") and ("12=1" in line or "12=2" in line) for line in lines):
            return False

    net = ModelOutputBoundary()
    net.eval()
    a = net(x)
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_permute_output_boundary.pt")
    if os.system("../../src/pnnx test_Tensor_permute_output_boundary.pt inputshape=[2,3,5,7]") != 0:
        return False

    with open("test_Tensor_permute_output_boundary.ncnn.param") as f:
        lines = f.readlines()
        if any(line.startswith("Reshape") and ("12=1" in line or "12=2" in line) for line in lines):
            return False
        if sum(1 for line in lines if line.startswith("Permute")) != 1:
            return False

    import test_Tensor_permute_output_boundary_ncnn
    b = test_Tensor_permute_output_boundary_ncnn.test_inference()
    if not torch.allclose(a, b, 1e-3, 1e-3):
        return False

    return True

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)
    v = torch.rand(280)

    a = net(x, y, z, w, v)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, v))
    mod.save("test_Tensor_permute.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_permute.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7],[280]")

    with open("test_Tensor_permute.ncnn.param") as f:
        lines = f.readlines()
        if sum(1 for line in lines if line.startswith("Reshape") and "12=1" in line) != 3:
            return False
        if sum(1 for line in lines if line.startswith("Reshape") and "12=2" in line) != 1:
            return False
        if sum(1 for line in lines if line.startswith("Permute")) != 10:
            return False

    # ncnn inference
    import test_Tensor_permute_ncnn
    b = test_Tensor_permute_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return test_conservative()

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
