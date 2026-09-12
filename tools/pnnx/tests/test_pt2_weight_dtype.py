# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# non-f32 weight/buffer dtypes (float16 / bfloat16 / int32 / int64) through the
# torchscript and pt2 (torch.export) channels. the loader must decode each
# attribute with its own element size and the generated pnnx python must be able
# to rebuild a bf16 buffer (numpy has no native bfloat16) before the .float()
# cast; a silent truncation or a missing dtype shows up as a numeric mismatch.

import os

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('w16', torch.arange(8, dtype=torch.float16).reshape(2, 4))
        self.register_buffer('wbf', torch.arange(8, dtype=torch.bfloat16).reshape(2, 4))
        self.register_buffer('wi32', torch.arange(8, dtype=torch.int32).reshape(2, 4))
        self.register_buffer('wi64', torch.arange(8, dtype=torch.int64).reshape(2, 4))

    def forward(self, x):
        y0 = x + self.w16.float()
        y1 = x + self.wbf.float()
        y2 = x + self.wi32.float()
        y3 = x + self.wi64.float()
        return y0, y1, y2, y3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 4)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, (x,))
    mod.save("test_pt2_weight_dtype.pt")

    # torchscript to pnnx
    os.system(os.path.join("..", "src", "pnnx") + " test_pt2_weight_dtype.pt inputshape=[2,4]")

    # pnnx inference
    import test_pt2_weight_dtype_pnnx
    b = test_pt2_weight_dtype_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, atol=1e-4, rtol=1e-4):
            return False
    ts_ok = True

    # pt2 path
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[2,4]"], "test_pt2_weight_dtype")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
