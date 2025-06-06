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

    def forward(self, x, y, z, w):
        x = x.clone()
        z = z.clone()
        x = x.index_put(indices=[torch.tensor([10,2])], values=y, accumulate=False)
        z.index_put_(indices=[torch.tensor([1,0,0]), torch.tensor([3,2,1])], values=w, accumulate=True)

        x[torch.tensor([1], dtype=torch.int64)] = torch.tensor(45).float()
        x[torch.tensor([], dtype=torch.int64)] = torch.tensor(233).float()
        return x, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(2)
    z = torch.rand(6,9)
    w = torch.rand(3)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_Tensor_index_put.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_index_put.pt inputshape=[12],[2],[6,9],[3]")

    # pnnx inference
    import test_Tensor_index_put_pnnx
    b = test_Tensor_index_put_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
