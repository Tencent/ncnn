# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

    def forward(self, x, y, z):
        x = F.local_response_norm(x, 3)
        x = F.local_response_norm(x, size=5, alpha=0.001, beta=0.8, k=0.9)

        y = F.local_response_norm(y, 4)
        y = F.local_response_norm(y, size=4, alpha=0.01, beta=0.2, k=1.9)

        z = F.local_response_norm(z, 5)
        z = F.local_response_norm(z, size=3, alpha=0.1, beta=0.3, k=0.2)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)

    a0, a1, a2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_local_response_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_local_response_norm.pt inputshape=[1,12,24],[2,3,12,16],[1,10,12,16,24]")

    # pnnx inference
    import test_F_local_response_norm_pnnx
    b0, b1, b2 = test_F_local_response_norm_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
