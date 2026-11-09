# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import re

import torch
import torch.nn as nn
from packaging import version
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = torch.arange(x[0])
        out1 = torch.arange(x[1], 33, dtype=None)
        out2 = torch.arange(x[3], x[4], x[6] * 0.1, dtype=torch.float)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (16,), dtype=torch.int)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_torch_arange.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_arange.pt inputshape=[16]i32")

    # pnnx inference
    import test_torch_arange_pnnx
    b = test_torch_arange_pnnx.test_inference()

    if len(a) != len(b):
        return False
    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
    try:
        ep = torch.export.export(net, (x,))
    except GuardOnDataDependentSymNode as e:
        # torch.export cannot guard the sign of a data-dependent floating step.
        if not re.search(r'data-dependent expression zuf\d+ > 0\.0\b', str(e)) or 'in arange)' not in str(e):
            raise
        print(f"skip pt2 export: upstream arange floating step limitation: {e}")
        return True
    torch.export.save(ep, "test_torch_arange_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_torch_arange_export.pt2 fp16=0")

    # pnnx inference
    import test_torch_arange_export_pnnx
    c = test_torch_arange_export_pnnx.test_inference()

    if len(a) != len(c):
        return False
    for a0, c0 in zip(a, c):
        if a0.dtype != c0.dtype or not torch.equal(a0, c0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
