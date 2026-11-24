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
        # the silu output is consumed by the mul and returned as a graph output,
        # so it must not be absorbed into SwiGLU
        shared = F.silu(self.gate0(x))
        shared_out = shared * self.up0(x)

        # broadcasting mul must not become a SwiGLU with mismatched shapes
        broadcast = F.silu(self.gate1(x)) * self.up1(x)[:, :1, :]

        return shared_out, shared, broadcast

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)

    x = torch.rand(2, 5, 16)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_ncnn_fuse_swiglu_negative.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_ncnn_fuse_swiglu_negative.pt inputshape=[2,5,16] fp16=0") != 0:
        return False

    # neither the shared silu nor the broadcast mul may be fused
    with open("test_ncnn_fuse_swiglu_negative.ncnn.param") as f:
        layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
    assert "SwiGLU" not in layers

    # ncnn inference
    import test_ncnn_fuse_swiglu_negative_ncnn
    b = test_ncnn_fuse_swiglu_negative_ncnn.test_inference()

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
