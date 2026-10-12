# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, m, k2, v2, m2, q3, k3, v3, q4, k4, v4, q5, k5, v5, m5, q6, k6, v6, q7, k7, v7):
        x = F.scaled_dot_product_attention(q, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        w = F.scaled_dot_product_attention(q3, k3, v3)
        u = F.scaled_dot_product_attention(q4, k4, v4)
        t = F.scaled_dot_product_attention(q5, k5, v5, attn_mask=m5)
        r = F.scaled_dot_product_attention(q6, k6, v6)
        s = F.scaled_dot_product_attention(q7, k7, v7)

        if version.parse(torch.__version__) >= version.parse('2.5'):
            z = F.scaled_dot_product_attention(q, k2, v2, enable_gqa=True)
            z2 = F.scaled_dot_product_attention(q, k2, v2, attn_mask=m2, enable_gqa=True)
        else:
            k2_stack = k2.repeat_interleave(q.size(-3)//k2.size(-3), -3)
            v2_stack = v2.repeat_interleave(q.size(-3)//v2.size(-3), -3)
            z = F.scaled_dot_product_attention(q, k2_stack, v2_stack)
            k2_stack = k2.clone().repeat_interleave(q.size(-3)//k2.size(-3), -3)
            v2_stack = v2.clone().repeat_interleave(q.size(-3)//v2.size(-3), -3)
            z2 = F.scaled_dot_product_attention(q, k2_stack, v2_stack, attn_mask=m2)

        return x, y, z, z2, w, u, t, r, s

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
    k2 = torch.rand(1, 2, 48, 64)
    v2 = torch.rand(1, 2, 48, 77)
    m2 = torch.rand(1, 1, 128, 48)
    q3 = torch.rand(2, 4, 16, 8)
    k3 = torch.rand(2, 4, 7, 8)
    v3 = torch.rand(2, 4, 7, 9)
    q4 = torch.rand(4, 16, 8)
    k4 = torch.rand(4, 7, 8)
    v4 = torch.rand(4, 7, 9)
    q5 = torch.rand(2, 4, 16, 8)
    k5 = torch.rand(4, 7, 8)
    v5 = torch.rand(4, 7, 9)
    m5 = torch.rand(16, 7)
    q6 = torch.rand(4, 16, 8)
    k6 = torch.rand(2, 4, 7, 8)
    v6 = torch.rand(2, 4, 7, 9)
    q7 = torch.rand(16, 8)
    k7 = torch.rand(7, 8)
    v7 = torch.rand(7, 9)

    a = net(q, k, v, m, k2, v2, m2, q3, k3, v3, q4, k4, v4, q5, k5, v5, m5, q6, k6, v6, q7, k7, v7)

    # export torchscript
    mod = torch.jit.trace(net, (q, k, v, m, k2, v2, m2, q3, k3, v3, q4, k4, v4, q5, k5, v5, m5, q6, k6, v6, q7, k7, v7))
    mod.save("test_F_scaled_dot_product_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_scaled_dot_product_attention.pt inputshape=[1,8,128,64],[1,8,48,64],[1,8,48,77],[1,8,128,48],[1,2,48,64],[1,2,48,77],[1,1,128,48],[2,4,16,8],[2,4,7,8],[2,4,7,9],[4,16,8],[4,7,8],[4,7,9],[2,4,16,8],[4,7,8],[4,7,9],[16,7],[4,16,8],[2,4,7,8],[2,4,7,9],[16,8],[7,8],[7,9]")

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
