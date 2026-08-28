# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.repeat_interleave(x, 2)
        y = torch.repeat_interleave(y, 3, dim=1)
        if version.parse(torch.__version__) >= version.parse('1.10'):
            z = torch.repeat_interleave(z, torch.tensor([2, 1, 3]), dim=0, output_size=6)
        else:
            z = torch.repeat_interleave(z, torch.tensor([2, 1, 3]), dim=0)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3)
    y = torch.rand(4, 5)
    z = torch.rand(3, 7, 8)

    a = net(x, y, z)

    # export pt2
    mod = torch.export.export(net, (x, y, z))
    torch.export.save(mod, "test_torch_repeat_interleave.pt2")

    # pt2 to pnnx
    import os
    os.system("../../src/pnnx test_torch_repeat_interleave.pt2 inputshape=[3],[4,5],[3,7,8]")

    # pnnx inference
    import test_torch_repeat_interleave_pnnx
    b = test_torch_repeat_interleave_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
