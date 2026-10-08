# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x = F.normalize(x, dim=0)
        x = F.normalize(x, dim=0, eps=1e-3)

        y = F.normalize(y, dim=0)
        y = F.normalize(y, dim=0, eps=1e-4)

        z = F.normalize(z, dim=0)
        z = F.normalize(z, dim=0, eps=1e-4)
        q = F.max_pool2d(q, 1)
        q0 = F.normalize(q, dim=1)
        q1 = F.normalize(q, dim=1, eps=1e-4)
        return x, y, z, q0, q1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(64)
    y = torch.rand(12, 24, 64)
    z = torch.rand(12, 3, 24, 64)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, q))
    mod.save("test_F_normalize.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_normalize.pt inputshape=[64],[12,24,64],[12,3,24,64],[2,3,5,7]")

    # ncnn inference
    import test_F_normalize_ncnn
    b = test_F_normalize_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

class NormalizeContractModel(nn.Module):
    def __init__(self, p, dim, batched):
        super(NormalizeContractModel, self).__init__()
        self.p = p
        self.dim = dim
        self.batched = batched

    def forward(self, x):
        if self.batched:
            x = F.max_pool2d(x, 1)
        return F.normalize(x, p=self.p, dim=self.dim, eps=1e-3)

def convert_contract_case(name, shape, p, dim, batched):
    import subprocess
    net = NormalizeContractModel(p, dim, batched).eval()
    torch.manual_seed(0)
    x = torch.rand(*shape)
    mod = torch.jit.trace(net, x)
    mod.save(name + ".pt")
    inputshape = "inputshape=[" + ",".join(str(d) for d in shape) + "]"
    subprocess.check_call(["../../src/pnnx", name + ".pt", inputshape])
    return net(x)

def test_supported_negative_axes():
    import importlib
    cases = [
        ((11,), 2.0, -1, False),
        ((5, 3, 7), 2, -3, False),
        ((5, 2, 3, 7), 2.0, -4, False),
        ((2, 5, 3, 7), 2, -3, True),
    ]
    for i, (shape, p, dim, batched) in enumerate(cases):
        name = "test_F_normalize_supported_" + str(i)
        expected = convert_contract_case(name, shape, p, dim, batched)
        actual = importlib.import_module(name + "_ncnn").test_inference()
        if not torch.allclose(expected, actual, 1e-4, 1e-4):
            return False
    return True

def test_unsupported():
    import ncnn
    cases = [
        ((11,), 1, 0, False),
        ((11,), 3, 0, False),
        ((11,), float("inf"), 0, False),
        ((5, 7), 2, 0, False),
        ((5, 3, 7), 2, 1, False),
        ((5, 2, 3, 7), 2, -1, False),
        ((2, 5, 3, 7), 2, 0, True),
        ((2, 5, 3, 7), 2, -4, True),
    ]
    for i, (shape, p, dim, batched) in enumerate(cases):
        name = "test_F_normalize_unsupported_" + str(i)
        convert_contract_case(name, shape, p, dim, batched)
        with open(name + ".ncnn.param") as f:
            layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
        if "F.normalize" not in layers or "Normalize" in layers:
            return False
        with ncnn.Net() as net:
            if net.load_param(name + ".ncnn.param") == 0:
                return False
    return True

if __name__ == "__main__":
    if test() and test_supported_negative_axes() and test_unsupported():
        exit(0)
    else:
        exit(1)
