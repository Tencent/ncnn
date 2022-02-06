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

    def forward(self, x, y):
        x0, x1 = torch.chunk(x, chunks=2, dim=0)
        y0, y1, y2 = torch.chunk(y, chunks=3, dim=1)
        return x0, x1, y0, y1, y2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(4, 16)
    y = torch.rand(5, 9, 11)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_chunk.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_chunk.pt inputshape=[4,16],[5,9,11]")

    # ncnn inference
    import test_torch_chunk_ncnn
    b = test_torch_chunk_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            print(a0.shape)
            print(b0.shape)
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
