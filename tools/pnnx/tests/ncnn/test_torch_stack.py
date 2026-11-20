# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w, q, r):
        z = F.max_pool2d(z, kernel_size=(2,2))
        w = F.max_pool2d(w, kernel_size=(2,2))
        q = F.max_pool2d(q, 1)
        r = F.max_pool2d(r, 1)
        out0 = torch.stack((x, y), dim=0)
        out1 = torch.stack((x, y), dim=2)
        out2 = torch.stack((z, w), dim=2)
        out3 = torch.stack((z, w), dim=-1)
        out4 = torch.stack((q, r), dim=1)
        out5 = torch.stack((q, r), dim=-1)
        out0.relu_()
        out1.relu_()
        out2.relu_()
        out3.relu_()
        out4.relu_()
        out5.relu_()
        return out0, out1, out2, out3, out4, out5

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)
    z = torch.rand(1, 5, 10, 4)
    w = torch.rand(1, 5, 10, 4)
    q = torch.rand(2, 3, 5, 7)
    r = torch.rand(2, 3, 5, 7)

    a = net(x, y, z, w, q, r)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w, q, r))
    mod.save("test_torch_stack.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_stack.pt inputshape=[3,16],[3,16],[1,5,10,4],[1,5,10,4],[2,3,5,7],[2,3,5,7]")

    # ncnn inference
    import test_torch_stack_ncnn
    b = test_torch_stack_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()

    def forward(self, x, y, z, w):
        z = F.max_pool2d(z, 1)
        w = F.max_pool2d(w, 1)
        return (torch.stack((x, y), dim=0).relu(),
                torch.stack((x, y), dim=1).relu(),
                torch.stack((x, y), dim=-1).relu(),
                torch.stack((z, w), dim=1).relu(),
                torch.stack((z, w), dim=-3).relu(),
                torch.stack((z, w), dim=-1).relu())

def test_dynamic():
    import ncnn
    import subprocess

    model = DynamicModel().eval()
    torch.manual_seed(0)
    inputs = []
    for shape, batch_shape in [((3, 5), (2, 3, 5, 7)),
                               ((4, 7), (3, 4, 6, 9)),
                               ((5, 8), (4, 2, 7, 10))]:
        inputs.append((torch.rand(shape), torch.rand(shape),
                       torch.rand(batch_shape), torch.rand(batch_shape)))

    mod = torch.jit.trace(model, inputs[0])
    mod.save("test_torch_stack_dynamic.pt")

    def input_shapes(data):
        return ",".join("[" + ",".join(str(d) for d in x.shape) + "]" for x in data)

    if subprocess.call(["../../src/pnnx", "test_torch_stack_dynamic.pt",
                        "inputshape=" + input_shapes(inputs[0]),
                        "inputshape2=" + input_shapes(inputs[1]), "fp16=0"]) != 0:
        return False

    with ncnn.Net() as net:
        if net.load_param("test_torch_stack_dynamic.ncnn.param") != 0:
            return False
        if net.load_model("test_torch_stack_dynamic.ncnn.bin") != 0:
            return False

        # The third shape is not used during conversion.
        for data in inputs:
            expected = model(*data)
            with net.create_extractor() as ex:
                for i, x in enumerate(data):
                    batch_axis = 233 if i < 2 else 0
                    if ex.input("in" + str(i), ncnn.Mat(x.numpy(), batch_index=batch_axis).clone()) != 0:
                        return False
                for i, a in enumerate(expected):
                    ret, out = ex.extract("out" + str(i))
                    if ret != 0:
                        return False
                    batch_axis = 233 if i < 3 else 0
                    b = torch.from_numpy(out.numpy(batch_index=batch_axis).copy())
                    if a.shape != b.shape or not torch.equal(a, b):
                        return False

    return True

if __name__ == "__main__":
    if test() and test_dynamic():
        exit(0)
    else:
        exit(1)
