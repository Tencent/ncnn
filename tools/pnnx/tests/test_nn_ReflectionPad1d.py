# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pad_0 = nn.ReflectionPad1d(2)
        self.pad_1 = nn.ReflectionPad1d(padding=(3,4))
        self.pad_2 = nn.ReflectionPad1d(padding=(1,0))

    def forward(self, x):
        x = self.pad_0(x)
        x = self.pad_1(x)
        x = self.pad_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 13)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_ReflectionPad1d.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_ReflectionPad1d.pt inputshape=[1,12,13]")

    # pnnx inference
    import test_nn_ReflectionPad1d_pnnx
    b = test_nn_ReflectionPad1d_pnnx.test_inference()

    if not (torch.equal(a, b)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_nn_ReflectionPad1d_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_ReflectionPad1d_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_ReflectionPad1d_export_pnnx
    c = test_nn_ReflectionPad1d_export_pnnx.test_inference()

    return torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
