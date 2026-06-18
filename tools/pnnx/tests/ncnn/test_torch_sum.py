# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.sum(x, dim=0, keepdim=False)
        y = torch.sum(y, dim=(1,2), keepdim=False)
        z = torch.sum(z, dim=(0,3), keepdim=True)
        return x, y, z

class ModelBatchWarning(nn.Module):
    def __init__(self):
        super(ModelBatchWarning, self).__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sum(x, dim=0)
        return x

def test_warning():
    import subprocess

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)

    net = ModelBatchWarning()
    net.eval()
    mod = torch.jit.trace(net, x)
    mod.save("test_torch_sum_batch_warning.pt")

    p = subprocess.run("../../src/pnnx test_torch_sum_batch_warning.pt inputshape=[2,3,5,7]", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        return False
    if "sum along batch axis is not supported yet" not in p.stdout:
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
    mod.save("test_torch_sum.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_sum.pt inputshape=[3,16],[5,9,11],[8,5,9,10]")

    # ncnn inference
    import test_torch_sum_ncnn
    b = test_torch_sum_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return test_warning()

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
