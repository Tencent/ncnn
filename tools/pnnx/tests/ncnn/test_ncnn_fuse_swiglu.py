# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gate0 = nn.Linear(16, 32)
        self.up0 = nn.Linear(16, 32)
        self.gate1 = nn.Linear(16, 32)
        self.up1 = nn.Linear(16, 32)

    def forward(self, x):
        gate0 = self.gate0(x)
        up0 = self.up0(x)
        gate1 = self.gate1(x)
        up1 = self.up1(x)

        # both operand orders of the final mul must be fused into SwiGLU
        return F.silu(gate0) * up0, up1 * F.silu(gate1)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)

    x = torch.rand(2, 5, 16)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_ncnn_fuse_swiglu.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_ncnn_fuse_swiglu.pt inputshape=[2,5,16] fp16=0") != 0:
        return False

    # both swish + mul pairs must be fused
    with open("test_ncnn_fuse_swiglu.ncnn.param") as f:
        layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
    assert layers.count("SwiGLU") == 2
    assert "Swish" not in layers
    assert "BinaryOp" not in layers

    # ncnn inference
    import test_ncnn_fuse_swiglu_ncnn
    b = test_ncnn_fuse_swiglu_ncnn.test_inference()

    assert len(a) == len(b)
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
