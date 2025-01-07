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

    def forward(self, x, y, z, d):
        x0, i0 = torch.topk(x, 4)
        y1, i1 = torch.topk(y, k=2, dim=0, largest=True)
        y2, i2 = torch.topk(y, k=2, dim=1, largest=False)
        # 3D
        z1, i3 = torch.topk(z, k=2, dim=0)
        z1, i4 = torch.topk(z, k=3, dim=1)
        z1, i5 = torch.topk(z, k=1, dim=2)
        # 4D
        # d0, i6 = torch.topk(
        #     d,
        #     k=2,
        #     dim=0,
        # )
        # d1, i7 = torch.topk(
        #     d,
        #     k=2,
        #     dim=1,
        # )
        d2, i8 = torch.topk(
            d,
            k=2,
            dim=2,
        )
        d3, i9 = torch.topk(d, k=2, dim=3, sorted=True)
        # return x0, y1, y2, z1, i3, i4, i5, d0, d1, d2, d3, i6, i7, i8, i9
        return x0, y1, y2, i0, i1, i2, z1, i3, i4, i5, d2, d3, i8, i9


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(36)  # 1D
    y = torch.rand(4, 7)  # 2D
    z = torch.rand(3, 4, 5)  # 3D
    d = torch.rand(4, 2, 6, 7)  # 4D

    a = net(x, y, z, d)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, d))
    mod.save("test_torch_topk.pt")

    # torchscript to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_topk.pt inputshape=[36],[4,7],[3,4,5],[4,2,6,7]"
    )

    # pnnx inference
    import test_torch_topk_ncnn

    b = test_torch_topk_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if a0.dtype != torch.float:
            a0 = a0.to(torch.int32)  # i64 --> i32
            b0 = b0.view(torch.int32)  # f32 --> i32
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
