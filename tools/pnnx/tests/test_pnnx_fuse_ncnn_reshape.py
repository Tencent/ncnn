# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(16, 32)
        self.l2 = nn.Linear(32, 8)

    def forward(self, x, y):
        # consecutive linear, 4d input
        x = self.l1(x)
        x = self.l2(x)
        # linear + relu + linear, 4d input
        y = self.l1(y)
        y = F.relu(y)
        y = self.l2(y)
        return x, y

class ModelReshapeChain(nn.Module):
    def __init__(self):
        super(ModelReshapeChain, self).__init__()
        self.l1 = nn.Linear(16, 32)
        self.l2 = nn.Linear(32, 8)

    def forward(self, x):
        # reshape chain feeding linear + relu + linear
        x = x.reshape(1, 3, -1, 16)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

class Model5D(nn.Module):
    def __init__(self):
        super(Model5D, self).__init__()
        self.l1 = nn.Linear(16, 32)
        self.l2 = nn.Linear(32, 8)

    def forward(self, x, y):
        # consecutive linear and linear + relu + linear, 5d input
        x = self.l1(x)
        x = self.l2(x)
        y = self.l1(y)
        y = F.relu(y)
        y = self.l2(y)
        return x, y

class ModelSqueeze(nn.Module):
    def __init__(self):
        super(ModelSqueeze, self).__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        x = self.conv(x)          # [2,8,1,16]
        a = x.squeeze(0)          # batch-axis no-op, must be eliminated
        b = x.squeeze()           # real squeeze, must be kept
        c = x.squeeze(2)          # real squeeze of the size-1 dim, must be kept
        return a + 1, b + 1, c + 1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 16)
    y = torch.rand(1, 3, 4, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_pnnx_fuse_ncnn_reshape.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_ncnn_reshape.pt inputshape=[1,3,4,16],[1,3,4,16]")

    # pnnx inference
    import test_pnnx_fuse_ncnn_reshape_pnnx
    b = test_pnnx_fuse_ncnn_reshape_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    # the reshape pairs inserted around 4d linear must be eliminated,
    # only the head/tail reshapes remain
    with open("test_pnnx_fuse_ncnn_reshape.ncnn.param", "r") as f:
        param = f.read()

    if param.count("Reshape") != 4:
        return False
    if param.count("Gemm") != 4:
        return False
    if param.count("ReLU") != 1:
        return False

    # reshape chain into linear + relu + linear
    net2 = ModelReshapeChain()
    net2.eval()

    torch.manual_seed(0)
    x2 = torch.rand(1, 3, 4, 16)

    a2 = net2(x2)

    mod2 = torch.jit.trace(net2, (x2,))
    mod2.save("test_pnnx_fuse_ncnn_reshape_chain.pt")

    os.system("../src/pnnx test_pnnx_fuse_ncnn_reshape_chain.pt inputshape=[1,3,4,16]")

    import test_pnnx_fuse_ncnn_reshape_chain_pnnx
    b2 = test_pnnx_fuse_ncnn_reshape_chain_pnnx.test_inference()

    if not torch.equal(a2, b2):
        return False

    with open("test_pnnx_fuse_ncnn_reshape_chain.ncnn.param", "r") as f:
        param2 = f.read()

    # only the head/tail reshapes around the two gemm remain
    if param2.count("Reshape") != 2:
        return False
    if param2.count("Gemm") != 2:
        return False
    if param2.count("ReLU") != 1:
        return False

    # 5d input
    net3 = Model5D()
    net3.eval()

    torch.manual_seed(0)
    x3 = torch.rand(1, 2, 3, 4, 16)
    y3 = torch.rand(1, 2, 3, 4, 16)

    a3 = net3(x3, y3)

    mod3 = torch.jit.trace(net3, (x3, y3))
    mod3.save("test_pnnx_fuse_ncnn_reshape_5d.pt")

    os.system("../src/pnnx test_pnnx_fuse_ncnn_reshape_5d.pt inputshape=[1,2,3,4,16],[1,2,3,4,16]")

    import test_pnnx_fuse_ncnn_reshape_5d_pnnx
    b3 = test_pnnx_fuse_ncnn_reshape_5d_pnnx.test_inference()

    for a0, b0 in zip(a3, b3):
        if not torch.equal(a0, b0):
            return False

    with open("test_pnnx_fuse_ncnn_reshape_5d.ncnn.param", "r") as f:
        param3 = f.read()

    if param3.count("Reshape") != 4:
        return False
    if param3.count("Gemm") != 4:
        return False
    if param3.count("ReLU") != 1:
        return False

    # squeeze on the batch axis is a no-op in the ncnn conversion and must be
    # eliminated, while real squeezes must be kept
    net4 = ModelSqueeze()
    net4.eval()

    torch.manual_seed(0)
    x4 = torch.rand(2, 3, 1, 16)

    a4 = net4(x4)

    mod4 = torch.jit.trace(net4, (x4,))
    mod4.save("test_pnnx_fuse_ncnn_reshape_squeeze.pt")

    os.system("../src/pnnx test_pnnx_fuse_ncnn_reshape_squeeze.pt inputshape=[2,3,1,16]")

    import test_pnnx_fuse_ncnn_reshape_squeeze_pnnx
    b4 = test_pnnx_fuse_ncnn_reshape_squeeze_pnnx.test_inference()

    for a0, b0 in zip(a4, b4):
        if not torch.equal(a0, b0):
            return False

    with open("test_pnnx_fuse_ncnn_reshape_squeeze.ncnn.param", "r") as f:
        param4 = f.read()

    if param4.count("Squeeze") != 2:
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
