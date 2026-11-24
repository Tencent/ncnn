# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1
        q = q * 2 - 1
        q = F.max_pool2d(q, 1)
        x_wide = F.log_softmax(x * 64, 0)
        y_wide = F.log_softmax(y * 64, 1)
        z_wide = F.log_softmax(z * 64, 2)
        q_wide = F.log_softmax(q * 64, -1)
        w_wide = F.log_softmax(w * 64, -1)
        x = F.log_softmax(x, 0)
        y = F.log_softmax(y, 1)
        z = F.log_softmax(z, 2)
        w = F.log_softmax(w, -1)
        q0 = F.log_softmax(q, 1)
        q1 = F.log_softmax(q, -1)
        return x, y, z, w, q0, q1, x_wide, y_wide, z_wide, q_wide, w_wide

class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return F.log_softmax(x, -1)

def test_dynamic():
    import subprocess
    import numpy as np
    import ncnn
    torch.manual_seed(0)
    net = DynamicModel().eval()
    data = torch.rand(2, 3, 4, 5) * 128 - 64
    name = "test_F_log_softmax_dynamic"
    mod = torch.jit.trace(net, data)
    mod.save(name + ".pt")
    proc = subprocess.run(["../../src/pnnx", name + ".pt",
                           "inputshape=[2,3,4,5]", "inputshape2=[2,3,6,7]", "fp16=0"])
    if proc.returncode != 0:
        return False
    with open(name + ".ncnn.param", encoding="utf-8") as f:
        graph = f.read()
    reductions = [line for line in graph.splitlines() if line.startswith("Reduction")]
    subtractions = [line for line in graph.splitlines() if line.startswith("BinaryOp")]
    if (len(reductions) != 1 or "0=10" not in reductions[0].split()
            or len(subtractions) != 1 or "0=1" not in subtractions[0].split()):
        return False
    with ncnn.Net() as runtime:
        runtime.opt.use_vulkan_compute = False
        runtime.opt.num_threads = 1
        if runtime.load_param(name + ".ncnn.param") != 0:
            return False
        if runtime.load_model(name + ".ncnn.bin") != 0:
            return False
        for shape in [(2, 3, 4, 5), (2, 3, 6, 7)]:
            data = torch.rand(shape) * 128 - 64
            expected = net(data)
            with runtime.create_extractor() as extractor:
                extractor.input("in0", ncnn.Mat(data.numpy(), batch_index=233).clone())
                status, value = extractor.extract("out0")
                if status != 0:
                    return False
                actual = torch.from_numpy(np.array(value.numpy(batch_index=233), copy=True))
            if (actual.shape != expected.shape or actual.dtype != expected.dtype
                    or not torch.allclose(actual, expected, 1e-4, 1e-4)):
                return False
    return True

def test():
    if not test_dynamic():
        return False

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(16)
    y = torch.rand(2, 16)
    z = torch.rand(3, 12, 16)
    w = torch.rand(5, 7, 9, 11)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q))
    mod.save("test_F_log_softmax.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_log_softmax.pt inputshape=[16],[2,16],[3,12,16],[5,7,9,11],[2,3,5,7]")

    # ncnn inference
    import test_F_log_softmax_ncnn
    b = test_F_log_softmax_ncnn.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape or not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
