# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, m, k2, v2, m2):
        x = F.scaled_dot_product_attention(q, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)

        if version.parse(torch.__version__) >= version.parse('2.5'):
            z = F.scaled_dot_product_attention(q, k2, v2, enable_gqa=True)
            z2 = F.scaled_dot_product_attention(q, k2, v2, attn_mask=m2, enable_gqa=True)
        else:
            k2_stack = k2.repeat_interleave(q.size(-3)//k2.size(-3), -3)
            v2_stack = v2.repeat_interleave(q.size(-3)//v2.size(-3), -3)
            z = F.scaled_dot_product_attention(q, k2_stack, v2_stack)
            z2 = F.scaled_dot_product_attention(q, k2_stack, v2_stack, attn_mask=m2)

        return x, y, z, z2

def test():
    if version.parse(torch.__version__) < version.parse('2.0'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    q = torch.rand(3, 8, 128, 64)
    k = torch.rand(3, 8, 48, 64)
    v = torch.rand(3, 8, 48, 77)
    m = torch.rand(3, 8, 128, 48)
    k2 = torch.rand(3, 2, 48, 64)
    v2 = torch.rand(3, 2, 48, 77)
    m2 = torch.rand(3, 1, 128, 48)

    a = net(q, k, v, m, k2, v2, m2)

    # export torchscript
    mod = torch.jit.trace(net, (q, k, v, m, k2, v2, m2))
    mod.save("test_F_scaled_dot_product_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_scaled_dot_product_attention.pt inputshape=[3,8,128,64],[3,8,48,64],[3,8,48,77],[3,8,128,48],[3,2,48,64],[3,2,48,77],[3,1,128,48]")

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
