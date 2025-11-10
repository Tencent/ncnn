# Copyright 2025 Tencent
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=1)
        return x, y

def test():
    if version.parse(torch.__version__) < version.parse('2.0'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    q = torch.rand(1, 8, 128, 64)
    k = torch.rand(1, 8, 48, 64)
    v = torch.rand(1, 8, 48, 77)
    m = torch.rand(1, 8, 128, 48)

    a = net(q, k, v, m)

    # export torchscript
    mod = torch.jit.trace(net, (q, k, v, m))
    mod.save("test_F_scaled_dot_product_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_scaled_dot_product_attention.pt inputshape=[1,8,128,64],[1,8,48,64],[1,8,48,77],[1,8,128,48]")

    # ncnn inference
    import test_F_scaled_dot_product_attention_ncnn
    b = test_F_scaled_dot_product_attention_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
