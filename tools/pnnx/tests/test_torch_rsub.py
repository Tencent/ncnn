# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = x * 8 - 4
        a = torch.rsub(x, 2.5)
        b = 2.5 - x
        c = torch.rsub(x, 2.5, alpha=2)
        d = torch.rsub(x, 2.5, alpha=0.5)
        e = torch.rsub(x, 2.5, alpha=-1)
        f = torch.rsub(x, 2.5, alpha=0)
        g = 2.5 - 2 * x
        h = torch.sin(torch.rsub(x, 2.5, alpha=-0.75))
        return a, b, c, d, e, f, g, h

def test():
    net = Model().eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save('test_torch_rsub.pt')

    # torchscript to pnnx
    import os
    os.system('../src/pnnx test_torch_rsub.pt inputshape=[3,16] fp16=0')

    # pnnx inference
    import test_torch_rsub_pnnx
    b = test_torch_rsub_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-6, 1e-6):
            return False
    return True

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
