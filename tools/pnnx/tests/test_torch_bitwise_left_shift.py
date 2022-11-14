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

    def forward(self, x, y):
        out = torch.bitwise_left_shift(x, y)
        return out

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (3, 16), dtype=torch.int)
    y = torch.randint(10, (3, 16), dtype=torch.int)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_torch_bitwise_left_shift.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_bitwise_left_shift.pt inputshape=[3,16]i32,[3,16]i32")

    # pnnx inference
    import test_torch_bitwise_left_shift_pnnx
    b = test_torch_bitwise_left_shift_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
