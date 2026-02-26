# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, m):
        x = F.scaled_dot_product_attention(q, k, v)
        q = q + 3
        k = k - 3
        v = v * 3
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return x, y

def test():
    if version.parse(torch.__version__) < version.parse('2.1'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    q = torch.rand(3, 8, 128, 64)
    k = torch.rand(3, 8, 48, 64)
    v = torch.rand(3, 8, 48, 77)
    m = torch.rand(3, 8, 128, 48)

    a = net(q, k, v, m)

    # export onnx
    torch.onnx.export(net, (q, k, v, m), "test_F_scaled_dot_product_attention.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_F_scaled_dot_product_attention.onnx inputshape=[3,8,128,64],[3,8,48,64],[3,8,48,77],[3,8,128,48]")

    # pnnx inference
    import test_F_scaled_dot_product_attention_pnnx
    b = test_F_scaled_dot_product_attention_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
