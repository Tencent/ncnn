# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = torch.round(x * 10)
        y = torch.round(y * 10)
        z = torch.round(z * 10)
        w = F.max_pool2d(w, 1)
        w = torch.round(w * 10)
        return x, y, z, w

class ModelWithDecimals(nn.Module):
    def __init__(self, decimals, method, embedded):
        super(ModelWithDecimals, self).__init__()
        self.decimals = decimals
        self.method = method
        self.embedded = embedded

    def forward(self, x):
        y = x.round(decimals=self.decimals) if self.method else torch.round(x, decimals=self.decimals)
        return y + 1 if self.embedded else y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)
    w = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_torch_round.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_round.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_round_ncnn
    b = test_torch_round_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    import ncnn
    x = torch.tensor([[-12.75, 12.75, -1.25], [1.25, 2.65, -2.65]])
    for i, args in enumerate([(1, False, False), (-1, True, False), (2, False, True), (-2, True, True)]):
        mod = torch.jit.trace(ModelWithDecimals(*args).eval(), x)
        name = "test_torch_round_decimals_%d" % i
        mod.save(name + ".pt")
        if os.system("../../src/pnnx %s.pt inputshape=[2,3]" % name) != 0:
            return False
        with open(name + ".ncnn.param", encoding="utf-8") as f:
            if "aten::round" not in f.read():
                return False
        with ncnn.Net() as net:
            net.opt.use_vulkan_compute = False
            if net.load_param(name + ".ncnn.param") != -1:
                return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
