# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

    def forward(self, x, y0, y1, z0, z1, w, r0, r1, r2):
        a = torch.einsum('ii', x)
        b = torch.einsum('ii->i', x)
        c = torch.einsum('i,j->ij', y0, y1)
        d = torch.einsum('bij,bjk->bik', z0, z1)
        e = torch.einsum('...ij->...ji', w)
        f = torch.einsum('bn,anm,bm->ba', r0, r1, r2)
        return a, b, c, d, e, f

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(4, 4)
    y0 = torch.rand(5)
    y1 = torch.rand(4)
    z0 = torch.rand(3, 2, 5)
    z1 = torch.rand(3, 5, 4)
    w = torch.rand(2, 3, 4, 5)
    r0 = torch.rand(2, 5)
    r1 = torch.rand(3, 5, 4)
    r2 = torch.rand(2, 4)

    a = net(x, y0, y1, z0, z1, w, r0, r1, r2)

    # export torchscript
    mod = torch.jit.trace(net, (x, y0, y1, z0, z1, w, r0, r1, r2))
    mod.save("test_torch_einsum.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_einsum.pt inputshape=[4,4],[5],[4],[3,2,5],[3,5,4],[2,3,4,5],[2,5],[3,5,4],[2,4]")

    # pnnx inference
    import test_torch_einsum_pnnx
    b = test_torch_einsum_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
