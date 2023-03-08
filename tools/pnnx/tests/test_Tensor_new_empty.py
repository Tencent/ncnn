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

    def forward(self, x):
        out0 = x.new_empty((2,2))
        out1 = x.new_empty(3)
        out2 = x.new_empty((4,5,6,7,8))
        out3 = x.new_empty((1,2,1))
        out4 = x.new_empty((3,3,3,3))
        return out0, out1, out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_new_empty.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_new_empty.pt inputshape=[1,16]")

    # pnnx inference
    import test_Tensor_new_empty_pnnx
    b = test_Tensor_new_empty_pnnx.test_inference()

    # test shape only for uninitialized data
    for a0, b0 in zip(a, b):
        if not a0.shape == b0.shape:
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
