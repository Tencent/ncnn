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


# 由于ncnn的MemoryData层不支持int64，所以先浮点存档，再转回int64
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#     def forward(self, x, y, z, d):
#         # 1D
#         x0 = torch.index_select(x, 0, torch.tensor([7, 9, 11]))
#         # 2D
#         y0 = torch.index_select(y, 0, torch.tensor([1, 0]))
#         y1 = torch.index_select(y, 1, torch.tensor([1, 3, 2]))
#         # 3D
#         z0 = torch.index_select(z, -3, torch.tensor([1, 0]))
#         z1 = torch.index_select(z, -2, torch.tensor([2, 0]))
#         z2 = torch.index_select(z, -1, torch.tensor([1, 2]))
#         # 4D
#         d0 = torch.index_select(d, 0, torch.tensor([1, 0]))
#         d1 = torch.index_select(d, 1, torch.tensor([1, 0]))
#         d2 = torch.index_select(d, 2, torch.tensor([1, 0]))
#         d3 = torch.index_select(d, 3, torch.tensor([1, 0]))

#         return x0, y0, y1, z0, z1, z2, d0, d1, d2, d3


# 成功
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 注册所有需要的索引缓冲区
        self.register_buffer("idx_x", torch.tensor([7.0, 9.0, 11.0]))
        self.register_buffer("idx_y0", torch.tensor([1.0, 0.0]))
        self.register_buffer("idx_y1", torch.tensor([1.0, 3.0, 2.0]))
        self.register_buffer("idx_z0", torch.tensor([1.0, 0.0]))
        self.register_buffer("idx_z1", torch.tensor([2.0, 0.0]))
        self.register_buffer("idx_z2", torch.tensor([1.0, 2.0, 0.0]))
        self.register_buffer("idx_d0", torch.tensor([1.0, 0.0, 3.0]))
        self.register_buffer("idx_d1", torch.tensor([0.0, 1.0]))
        self.register_buffer("idx_d2", torch.tensor([4.0, 3.0, 0.0]))
        self.register_buffer("idx_d3", torch.tensor([3.0, 6.0, 2.0]))

    def float_to_int(self, idx_float):
        mask = torch.ones_like(idx_float)
        return (torch.max(idx_float * mask, mask * 0)).int()

    def forward(self, x, y, z, d):
        # 使用辅助函数进行转换
        x0 = torch.index_select(x, 0, self.float_to_int(self.idx_x))
        y0 = torch.index_select(y, 0, self.float_to_int(self.idx_y0))
        y1 = torch.index_select(y, 1, self.float_to_int(self.idx_y1))
        z0 = torch.index_select(z, -3, self.float_to_int(self.idx_z0))
        z1 = torch.index_select(z, -2, self.float_to_int(self.idx_z1))
        z2 = torch.index_select(z, -1, self.float_to_int(self.idx_z2))
        d0 = torch.index_select(d, 0, self.float_to_int(self.idx_d0))
        d1 = torch.index_select(d, 1, self.float_to_int(self.idx_d1))
        d2 = torch.index_select(d, 2, self.float_to_int(self.idx_d2))
        d3 = torch.index_select(d, 3, self.float_to_int(self.idx_d3))

        return x0, y0, y1, z0, z1, z2, d0, d1, d2, d3


def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(36)  # 1D
    y = torch.rand(5, 7)  # 2D
    z = torch.rand(3, 5, 8)  # 3D
    d = torch.rand(4, 3, 6, 7)  # 4D

    a = net(x, y, z, d)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, d))
    mod.save("test_torch_index_select.pt")

    # torchscript to pnnx
    import os

    os.system(
        "../../src/pnnx test_torch_index_select.pt inputshape=[36],[5,7],[3,5,8],[4,3,6,7]"
    )

    # pnnx inference
    import test_torch_index_select_ncnn

    b = test_torch_index_select_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
