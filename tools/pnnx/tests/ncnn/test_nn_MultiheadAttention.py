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

        self.attention_0_0 = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        if torch.__version__ >= '1.9':
            self.attention_1_0 = nn.MultiheadAttention(embed_dim=40, num_heads=4, batch_first=True)

    def forward(self, x, y):
        x0, _ = self.attention_0_0(x, x, x)

        if torch.__version__ < '1.9':
            return x0

        y0, _ = self.attention_1_0(y, y, y)

        return x0, y0

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, 64)
    y = torch.rand(1, 15, 40)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_nn_MultiheadAttention.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_MultiheadAttention.pt inputshape=[1,1,64],[1,15,40]")

    # ncnn inference
    import test_nn_MultiheadAttention_ncnn
    b = test_nn_MultiheadAttention_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
