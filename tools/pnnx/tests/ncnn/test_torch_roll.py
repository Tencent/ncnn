# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, q):
        x0 = torch.roll(x, shifts=(2,1), dims=(1,1))
        x1 = torch.roll(x, shifts=(9,10), dims=(-1,-1))
        y0 = torch.roll(y, shifts=(2,-2), dims=(2,-1))
        z0 = torch.roll(z, shifts=(-2,-1), dims=(0,-4))
        x = torch.roll(x, 3, 1)
        y = torch.roll(y, -2, -1)
        z = torch.roll(z, shifts=(2,1), dims=(0,1))
        q = F.max_pool2d(q, 1)
        q0 = torch.roll(q, 2, 1)
        q1 = torch.roll(q, shifts=(1,-2), dims=(2,3))
        q2 = torch.roll(q, shifts=(2,1), dims=(3,-1))
        q3 = torch.roll(q, shifts=(1,1), dims=(1,-3))
        return x, y, z, q0, q1, x0, x1, y0, z0, q2, q3

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
    mod.save("test_torch_roll.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_torch_roll.pt inputshape=[3,16],[5,9,11],[8,5,9,10],[2,3,5,7]")

    # ncnn inference
    import test_torch_roll_ncnn
    b = test_torch_roll_ncnn.test_inference()

    print(x)
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            print(a0)
            print(b0)
            return False
    return True

class DynamicModel(nn.Module):
    def forward(self, x):
        return (torch.roll(x, shifts=(3,-2), dims=(1,-1)),
                torch.roll(x, shifts=(1,2), dims=(0,0)))

def test_dynamic():
    import ncnn
    import subprocess

    model = DynamicModel().eval()
    torch.manual_seed(0)
    inputs = [torch.rand(3, 8), torch.rand(5, 11), torch.rand(4, 9)]
    torch.jit.trace(model, inputs[0]).save("test_torch_roll_dynamic.pt")
    if subprocess.call(["../../src/pnnx", "test_torch_roll_dynamic.pt",
                        "inputshape=[3,8]", "inputshape2=[5,11]", "fp16=0"]) != 0:
        return False

    with ncnn.Net() as net:
        if net.load_param("test_torch_roll_dynamic.ncnn.param") != 0:
            return False
        if net.load_model("test_torch_roll_dynamic.ncnn.bin") != 0:
            return False

        # Include a third shape that was not used during conversion.
        for x in inputs:
            expected = model(x)
            with net.create_extractor() as ex:
                if ex.input("in0", ncnn.Mat(x.numpy()).clone()) != 0:
                    return False
                for i, a in enumerate(expected):
                    ret, out = ex.extract("out" + str(i))
                    if ret != 0:
                        return False
                    b = torch.from_numpy(out.numpy().copy())
                    if a.shape != b.shape or not torch.equal(a, b):
                        return False
    return True

class BatchRollModel(nn.Module):
    def __init__(self, dims):
        super(BatchRollModel, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        return torch.roll(x, shifts=(1,1), dims=self.dims)

def test_unsupported_batch():
    import ncnn
    import subprocess

    x = torch.rand(2, 3, 5, 7)
    for dims in [(0,0), (0,1), (1,-4)]:
        model = BatchRollModel(dims).eval()
        torch.jit.trace(model, x).save("test_torch_roll_batch.pt")
        if subprocess.call(["../../src/pnnx", "test_torch_roll_batch.pt",
                            "inputshape=[2,3,5,7]", "fp16=0"]) != 0:
            return False
        with ncnn.Net() as net:
            if net.load_param("test_torch_roll_batch.ncnn.param") == 0:
                return False
    return True

if __name__ == "__main__":
    if test() and test_dynamic() and test_unsupported_batch():
        exit(0)
    else:
        exit(1)
