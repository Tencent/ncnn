# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        mask = x > 0.5
        out0 = x.masked_fill(mask, -1.0)
        out1 = x.masked_fill(x > 0.3, 0.0)
        out2 = x.masked_fill(x < 0.2, 100.0)
        return out0, out1, out2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 10)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, (x,))
    mod.save("test_torch_masked_fill.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_masked_fill.pt inputshape=[1,10]")

    # ncnn inference
    import test_torch_masked_fill_ncnn
    b = test_torch_masked_fill_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
