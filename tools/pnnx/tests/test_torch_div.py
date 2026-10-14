# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x * 8 - 4
        y = y * 2 + 1
        a = torch.div(x, y)
        b = torch.div(x, y, rounding_mode='floor')
        c = torch.div(x, y, rounding_mode='trunc')
        d = torch.div(x, 2.5, rounding_mode='floor')
        e = torch.div(x, 2.5, rounding_mode='trunc')
        f = x.div(-y, rounding_mode='floor')
        g = x.div(-y, rounding_mode='trunc')
        h = torch.sin(torch.div(x, y, rounding_mode='floor')) * 2
        i = torch.sin(torch.div(x, y, rounding_mode='trunc')) * 2
        return a, b, c, d, e, f, g, h, i

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(3, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save('test_torch_div.pt')

    # torchscript to pnnx
    import os
    os.system('../src/pnnx test_torch_div.pt inputshape=[3,16],[3,16] fp16=0')

    # pnnx inference
    import test_torch_div_pnnx
    b = test_torch_div_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-6, 1e-6):
            return False
    return True

if __name__ == '__main__':
    if test():
        exit(0)
    else:
        exit(1)
