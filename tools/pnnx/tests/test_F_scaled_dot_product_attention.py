# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, m):
        x = F.scaled_dot_product_attention(q, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    q = torch.rand(3, 8, 128, 64)
    k = torch.rand(3, 8, 48, 64)
    v = torch.rand(3, 8, 48, 77)
    m = torch.rand(3, 8, 128, 48)

    a = net(q, k, v, m)

    # export torchscript
    mod = torch.jit.trace(net, (q, k, v, m))
    mod.save("test_F_scaled_dot_product_attention.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_scaled_dot_product_attention.pt inputshape=[3,8,128,64],[3,8,48,64],[3,8,48,77],[3,8,128,48]")

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
