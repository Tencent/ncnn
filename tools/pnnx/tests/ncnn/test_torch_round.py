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

class ModelWithDynamicDecimals(nn.Module):
    def forward(self, x):
        return torch.round(x, decimals=int(x[0, 0]))

def run_ncnn_model(name, x):
    import ncnn
    with ncnn.Net() as net:
        assert net.load_param(name + ".ncnn.param") == 0
        assert net.load_model(name + ".ncnn.bin") == 0
        with net.create_extractor() as ex:
            assert ex.input("in0", ncnn.Mat(x.numpy(), batch_index=233).clone()) == 0
            ret, value = ex.extract("out0")
            assert ret == 0
            return torch.from_numpy(value.numpy(batch_index=233).copy())

def test_round_decimals():
    try:
        torch.round(torch.tensor([0.]), decimals=0)
    except TypeError:
        return True

    decimals_input = torch.tensor([[-12.75, 12.75, -1.25], [1.25, 2.65, -2.65]])
    cases = [(1, False, False), (-1, True, False), (2, False, True),
             (-2, True, True), (0, False, False), (0, True, True)]
    for i, args in enumerate(cases):
        decimals_model = ModelWithDecimals(*args).eval()
        expected = decimals_model(decimals_input)
        mod = torch.jit.trace(decimals_model, decimals_input)
        name = "test_torch_round_decimals_%d" % i
        mod.save(name + ".pt")
        import subprocess
        proc = subprocess.run(["../../src/pnnx", name + ".pt", "inputshape=[2,3]", "fp16=0"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or b"ignoring decimals" in proc.stderr:
            return False
        with open(name + ".pnnx.param", encoding="utf-8") as f:
            graph = f.read()
            if "aten::round" in graph or "round(" not in graph:
                return False
        actual = run_ncnn_model(name, decimals_input)
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)):
            return False

    # Runtime decimals are unsupported: keep a usable plain-round fallback.
    dynamic_input = decimals_input.clone()
    dynamic_input[0, 0] = 1.25
    mod = torch.jit.script(ModelWithDynamicDecimals().eval())
    name = "test_torch_round_dynamic_decimals"
    mod.save(name + ".pt")
    proc = subprocess.run(["../../src/pnnx", name + ".pt", "inputshape=[2,3]", "fp16=0"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or b"unsupported torch.round non-constant decimals" not in proc.stderr:
        return False
    with open(name + ".pnnx.param", encoding="utf-8") as f:
        graph = f.read()
        if "aten::round" in graph or "round(" not in graph:
            return False
    actual = run_ncnn_model(name, dynamic_input)
    expected = torch.round(dynamic_input)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not torch.equal(actual, expected)):
        return False

    return True

def test():
    if not test_round_decimals():
        return False

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
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
