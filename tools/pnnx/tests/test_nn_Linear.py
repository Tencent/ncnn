# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from pnnx_test_utils import exported_program_to_pnnx, has_torch_export, torchscript_to_pnnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(in_features=64, out_features=16, bias=False)
        self.linear_1 = nn.Linear(in_features=16, out_features=13, bias=True)

        self.linear_2 = nn.Linear(in_features=13, out_features=17, bias=True)
        if version.parse(torch.__version__) < version.parse('1.9'):
            # weight_norm on torch 1.8 produces wrong output shape, skip it
            pass
        elif version.parse(torch.__version__) < version.parse('2.1'):
            self.linear_2 = torch.nn.utils.weight_norm(self.linear_2)
        else:
            self.linear_2 = torch.nn.utils.parametrizations.weight_norm(self.linear_2)

    def forward(self, x, y, z):
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.linear_2(x)

        y = self.linear_0(y)
        y = self.linear_1(y)
        y = self.linear_2(y)

        z = self.linear_0(z)
        z = self.linear_1(z)
        z = self.linear_2(z)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 64)
    y = torch.rand(12, 64)
    z = torch.rand(1, 3, 12, 64)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_nn_Linear.pt")

    # torchscript to pnnx
    converted = torchscript_to_pnnx("test_nn_Linear", "[1,64],[12,64],[1,3,12,64]")

    # pnnx inference
    b = converted(x, y, z)

    for a0, b0 in zip(a, b):
        b0 = b0.reshape_as(a0)
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    if not has_torch_export():
        return True

    if hasattr(net.linear_2, "parametrizations"):
        torch.nn.utils.parametrize.remove_parametrizations(net.linear_2, "weight", leave_parametrized=True)
    a = net(x, y, z)
    converted = exported_program_to_pnnx(net, (x, y, z), "test_nn_Linear_pt2")
    c = converted(x, y, z)

    return all(torch.allclose(a0, c0.reshape_as(a0), 1e-3, 1e-3) for a0, c0 in zip(a, c))

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
