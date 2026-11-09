# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        mask = x > y
        out = x.masked_fill(mask, 2)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_Tensor_masked_fill.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_masked_fill.pt inputshape=[6,16],[6,16]")

    # pnnx inference
    import test_Tensor_masked_fill_pnnx
    b = test_Tensor_masked_fill_pnnx.test_inference()

    if not torch.equal(a, b):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x, y))
    torch.export.save(ep, "test_Tensor_masked_fill_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_Tensor_masked_fill_export.pt2 fp16=0")

    # pnnx inference
    import test_Tensor_masked_fill_export_pnnx
    c = test_Tensor_masked_fill_export_pnnx.test_inference()

    return torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
