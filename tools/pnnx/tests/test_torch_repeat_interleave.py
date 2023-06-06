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
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.repeat_interleave(x, 2)
        y = torch.repeat_interleave(y, 3, dim=1)
        if version.parse(torch.__version__) >= version.parse('1.10'):
            z = torch.repeat_interleave(z, torch.tensor([2, 1, 3]), dim=0, output_size=6)
        else:
            z = torch.repeat_interleave(z, torch.tensor([2, 1, 3]), dim=0)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3)
    y = torch.rand(4, 5)
    z = torch.rand(3, 7, 8)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_repeat_interleave.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_repeat_interleave.pt inputshape=[3],[4,5],[3,7,8]")

    # pnnx inference
    import test_torch_repeat_interleave_pnnx
    b = test_torch_repeat_interleave_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
