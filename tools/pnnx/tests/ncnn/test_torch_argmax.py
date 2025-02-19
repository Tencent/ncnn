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

    def forward(self, x, y, z, d):
        # 1D
        x0 = torch.argmax(x, 0, keepdim=True)
        # 2D
        y0 = torch.argmax(y, 0, keepdim=True)
        y1 = torch.argmax(y, 1, keepdim=False)
        # 3D
        z0 = torch.argmax(z, -3, keepdim=False)
        z1 = torch.argmax(z, -2, keepdim=True)
        z2 = torch.argmax(z, -1, keepdim=False)
        # 4D
        d0 = torch.argmax(d, 0, keepdim=True)
        d1 = torch.argmax(d, 1, keepdim=False)
        d2 = torch.argmax(d, 2)
        d3 = torch.argmax(d, 3, keepdim=False)

        return x0, y0, y1, z0, z1, z2, d0, d1, d2, d3


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(36)  # 1D
    y = torch.rand(5, 7)  # 2D
    z = torch.rand(4, 5, 8)  # 3D
    d = torch.rand(5, 8, 6, 7)  # 4D

    a = net(x, y, z, d)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, d))

    a = net(x, y, z, d)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_argmax.pt")

    # torchscript to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_argmax.pt inputshape=[36],[5,7],[4,5,8],[5,8,6,7]"
    )

    # ncnn inference
    import test_torch_argmax_ncnn

    b = test_torch_argmax_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if a0.dtype != torch.float:
            a0 = a0.to(torch.int32)  # i64 --> i32
            b0 = b0.view(torch.int32)  # f32 --> i32
        if not torch.equal(a0, b0):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
