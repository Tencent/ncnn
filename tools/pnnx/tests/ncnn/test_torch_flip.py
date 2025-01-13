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
        # 1D
        x0 = torch.flip(x, [0])
        # 2D
        y0 = torch.flip(y, [0])
        y1 = torch.flip(y, [1])
        y2 = torch.flip(y, [-2, -1])
        # 3D
        z0 = torch.flip(z, [0])
        z1 = torch.flip(z, [1])
        z2 = torch.flip(z, [2])
        z3 = torch.flip(z, [0, 1])
        z4 = torch.flip(z, [0, 2])
        z5 = torch.flip(z, [1, 2])
        z6 = torch.flip(z, [0, 1, 2])
        # 4D
        d0 = torch.flip(d, [-1])
        d1 = torch.flip(d, [-2])
        d2 = torch.flip(d, [-3])
        d3 = torch.flip(d, [-4])
        d4 = torch.flip(d, [0, 1])
        d5 = torch.flip(d, [0, 2])
        d6 = torch.flip(d, [0, 3])
        d7 = torch.flip(d, [1, 2])
        d8 = torch.flip(d, [1, 3])
        d9 = torch.flip(d, [2, 3])
        d10 = torch.flip(d, [0, 1, 2])
        d11 = torch.flip(d, [0, 1, 3])
        d12 = torch.flip(d, [0, 2, 3])
        d13 = torch.flip(d, [1, 2, 3])
        d14 = torch.flip(d, [0, 1, 2, 3])

        return (
            x0,
            y0,
            y1,
            y2,
            z0,
            z1,
            z2,
            z3,
            z4,
            z5,
            z6,
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
        )


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
    mod.save("test_torch_flip.pt")

    # torchscript to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_flip.pt inputshape=[36],[4,7],[3,4,5],[4,2,6,7]"
    )

    # pnnx inference
    import test_torch_flip_ncnn

    b = test_torch_flip_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
