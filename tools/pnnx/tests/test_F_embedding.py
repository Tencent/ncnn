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

        self.w1 = nn.Parameter(torch.rand(10, 128))

    def forward(self, x, w0, y):
        x = F.embedding(x, w0)
        y = F.embedding(y, self.w1)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (1, 13), dtype=torch.int)
    w0 = torch.rand(10, 128)
    y = torch.randint(10, (1, 11), dtype=torch.int)

    a0, a1 = net(x, w0, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, w0, y))
    mod.save("test_F_embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_embedding.pt inputshape=[1,13]i32,[10,128],[1,11]i32")

    # pnnx inference
    import test_F_embedding_pnnx
    b0, b1 = test_F_embedding_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
