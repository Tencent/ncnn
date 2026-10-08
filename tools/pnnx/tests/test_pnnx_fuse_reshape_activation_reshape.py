# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        # unsqueeze + relu + reshape -> reshape + relu
        x = x.unsqueeze(1).relu().reshape(2, 3, 4)
        # reshape + sigmoid + squeeze -> reshape + sigmoid
        y = y.reshape(1, 24).sigmoid().squeeze(0)
        # reshape chain + tanh + reshape chain -> reshape + tanh
        z = z.reshape(2, 6).reshape(2, 2, 3).tanh().reshape(3, 4).reshape(2, 6)
        # softmax is dim-sensitive, should NOT be merged
        w = w.reshape(2, 3, 4).softmax(dim=1).reshape(4, 6)
        return x, y, z, w

class ModelDynamic(nn.Module):
    def __init__(self):
        super(ModelDynamic, self).__init__()

    def forward(self, x):
        # adjacent reshapes with dynamic dims
        x = x.reshape(2, -1).reshape(-1, 4)
        # reshape + relu + reshape with dynamic dims
        x = x.reshape(-1, 8).relu().reshape(4, -1)
        # unsqueeze + sigmoid + squeeze roundtrip
        x = x.unsqueeze(0).sigmoid().squeeze(0)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(24)
    y = torch.rand(2, 3, 4)
    z = torch.rand(12)
    w = torch.rand(24)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_pnnx_fuse_reshape_activation_reshape.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_pnnx_fuse_reshape_activation_reshape.pt inputshape=[24],[2,3,4],[12],[24]")

    # pnnx inference
    import test_pnnx_fuse_reshape_activation_reshape_pnnx
    b = test_pnnx_fuse_reshape_activation_reshape_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    # verify the fused graph structure
    with open("test_pnnx_fuse_reshape_activation_reshape_pnnx.py", "r") as f:
        code = f.read()

    # x/y/z each merged to one reshape, softmax case keeps two reshapes
    if code.count(".reshape(") != 5:
        return False
    if code.count("F.relu(") != 1:
        return False
    if code.count("F.sigmoid(") != 1:
        return False
    if code.count("F.tanh(") != 1:
        return False
    if code.count("F.softmax(") != 1:
        return False

    # dynamic shape model, pnnx run without inputshape
    net2 = ModelDynamic()
    net2.eval()

    torch.manual_seed(0)
    x2 = torch.rand(8, 6)

    a2 = net2(x2)

    mod2 = torch.jit.trace(net2, (x2,))
    mod2.save("test_pnnx_fuse_reshape_activation_reshape_dynamic.pt")

    os.system("../src/pnnx test_pnnx_fuse_reshape_activation_reshape_dynamic.pt")

    import test_pnnx_fuse_reshape_activation_reshape_dynamic_pnnx
    b2 = test_pnnx_fuse_reshape_activation_reshape_dynamic_pnnx.test_inference()

    if not torch.equal(a2, b2):
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
