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

    def forward(self, x, y):
        x = F.dropout3d(x, training=False)
        z = F.dropout3d(y, p=0.6, training=False)
        x = F.relu(x)
        y = F.relu(y)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12, 4, 2, 16)
    y = torch.rand(3, 5, 12, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_dropout3d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_dropout3d.pt inputshape=[12,4,2,16],[3,5,12,16]")

    # ncnn inference
    import test_F_dropout3d_ncnn
    b = test_F_dropout3d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
