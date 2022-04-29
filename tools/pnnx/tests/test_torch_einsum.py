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
        # identity
        a0 = torch.einsum('i', y0)
        a1 = torch.einsum('ij', x)
        a2 = torch.einsum('ijk', z0)
        a3 = torch.einsum('ijkl', w)

        # permute
        b0 = torch.einsum('ij->ji', x)
        b1 = torch.einsum('ba', x)
        b2 = torch.einsum('jki', z0)
        b3 = torch.einsum('ijk->kij', z0)
        b4 = torch.einsum('kjil', w)
        b5 = torch.einsum('ijkl->jilk', w)
        b6 = torch.einsum('...ij->...ji', w)
        b7 = torch.einsum('abc...->cba...', w)

        # trace
        c = torch.einsum('ii', x)

        # sum
        d0 = torch.einsum('ij->', x)
        d1 = torch.einsum('xyz->', z0)
        d2 = torch.einsum('ijkl->', w)

        # sum axis
        e0 = torch.einsum('ij->i', x)
        e1 = torch.einsum('ij->j', x)
        e2 = torch.einsum('ijk->i', z0)
        e3 = torch.einsum('ijk->j', z0)
        e4 = torch.einsum('ijk->k', z0)
        e5 = torch.einsum('ijk->ij', z0)
        e6 = torch.einsum('ijk->jk', z0)
        e7 = torch.einsum('ijk->ik', z0)
        e8 = torch.einsum('ijkl->i', w)
        e9 = torch.einsum('ijkl->j', w)
        e10 = torch.einsum('ijkl->k', w)
        e11 = torch.einsum('ijkl->l', w)
        e12 = torch.einsum('ijkl->ij', w)
        e13 = torch.einsum('ijkl->jk', w)
        e14 = torch.einsum('ijkl->kl', w)
        e15 = torch.einsum('ijkl->il', w)
        e16 = torch.einsum('ijkl->ijk', w)
        e17 = torch.einsum('ijkl->jkl', w)
        e18 = torch.einsum('ijkl->ijl', w)

        # matrix-vector
        f0 = torch.einsum('ij,j->i', r0, y0)
        f1 = torch.einsum('i,jki->jk', y1, r1)

        # vector-vector outer product
        g = torch.einsum('i,j->ij', y0, y1)

        # batch mm
        h0 = torch.einsum('bij,bjk->bik', z0, z1)
        h1 = torch.einsum('bjk,bij->bik', z1, z0)

        # bilinear
        i = torch.einsum('bn,anm,bm->ba', r0, r1, r2)
        return a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7, c, d0, d1, d2, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e17, f0, f1, g, h0, h1, i

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
