# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w3 = nn.Parameter(torch.rand(24))
        self.b3 = nn.Parameter(torch.rand(24))
        self.w4 = nn.Parameter(torch.rand(12, 16))
        self.b4 = nn.Parameter(torch.rand(12, 16))
        self.w5 = nn.Parameter(torch.rand(24))
        self.b5 = nn.Parameter(torch.rand(24))

    def forward(self, x, y, z, w0, b0, w1, b1, w2, b2):
        x = F.layer_norm(x, (24,), w0, b0)
        if version.parse(torch.__version__) < version.parse('2.1') or version.parse(torch.__version__) >= version.parse('2.2'):
            x = F.layer_norm(x, (12,24), None, None)
        x = F.layer_norm(x, (24,), self.w3, self.b3)

        if version.parse(torch.__version__) < version.parse('2.1') or version.parse(torch.__version__) >= version.parse('2.2'):
            y = F.layer_norm(y, (16,), None, None, eps=1e-3)
        y = F.layer_norm(y, (12,16), w1, b1)
        y = F.layer_norm(y, (12,16), self.w4, self.b4)

        z = F.layer_norm(z, (24,), w2, b2)
        if version.parse(torch.__version__) < version.parse('2.1') or version.parse(torch.__version__) >= version.parse('2.2'):
            z = F.layer_norm(z, (12,16,24), None, None, eps=1e-2)
        z = F.layer_norm(z, (24,), self.w5, self.b5)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)
    w0 = torch.rand(24)
    b0 = torch.rand(24)
    w1 = torch.rand(12, 16)
    b1 = torch.rand(12, 16)
    w2 = torch.rand(24)
    b2 = torch.rand(24)

    a0, a1, a2 = net(x, y, z, w0, b0, w1, b1, w2, b2)

    # export onnx
    torch.onnx.export(net, (x, y, z, w0, b0, w1, b1, w2, b2), "test_F_layer_norm.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_layer_norm.onnx inputshape=[1,12,24],[2,3,12,16],[1,10,12,16,24],[24],[24],[12,16],[12,16],[24],[24]")

    # pnnx inference
    import test_F_layer_norm_pnnx
    b0, b1, b2 = test_F_layer_norm_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
