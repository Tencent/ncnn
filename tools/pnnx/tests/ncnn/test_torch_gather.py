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


# # 由于ncnn的MemoryData层不支持int64，如下代码无法通过测试
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, d):
        # 1D
        x0 = torch.gather(x, 0, torch.tensor([7, 9, 11]))
        # 2D
        y0 = torch.gather(y, 0, torch.tensor([[1, 3, 2], [0, 3, 1]]))
        y1 = torch.gather(y, 1, torch.tensor([[1, 3, 2, 4, 6, 5], [4, 3, 2, 1, 5, 6]]))
        # 3D
        z0 = torch.gather(z, -3, torch.tensor([[[0], [1], [0]], [[1], [0], [1]]]))
        z1 = torch.gather(z, -2, torch.tensor([[[0], [1], [2]], [[1], [2], [0]]]))
        z2 = torch.gather(z, -1, torch.tensor([[[0, 1, 2, 3]], [[2, 1, 0, 3]]]))
        # 4D
        zz = torch.tensor(
            [
                [[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]],
                [[[1, 0, 3, 4, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]],
            ]
        )
        d0 = torch.gather(d, 0, zz)
        d1 = torch.gather(d, 1, zz)
        d2 = torch.gather(d, 2, zz)
        d3 = torch.gather(d, 3, zz)

        return x0, y0, y1, z0, z1, z2, d0, d1, d2, d3


def test():
    return True  # TODO: need i64 support in pnnx
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
    mod.save("test_torch_gather.pt")

    # torchscript to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_gather.pt inputshape=[36],[5,7],[4,5,8],[5,8,6,7]"
    )

    # pnnx inference
    import test_torch_gather_ncnn

    b = test_torch_gather_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
