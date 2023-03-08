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

    def forward(self, x, y, z):
        x = x[:,:12,1:14:2]
        x = x[...,1:]
        x = x[:,:,:x.size(2)-1]
        y = y[0:,1:,5:,3:]
        y = y[:,:,1:13:2,:14]
        y = y[:1,:y.size(1):,:,:]
        z = z[4:]
        z = z[:2,:,:,:,2:-2:3]
        z = z[:,:,:,z.size(3)-3:,:]
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 13, 26)
    y = torch.rand(1, 15, 19, 21)
    z = torch.rand(14, 18, 15, 19, 20)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_slice.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_slice.pt inputshape=[1,13,26],[1,15,19,21],[14,18,15,19,20]")

    # pnnx inference
    import test_Tensor_slice_pnnx
    b = test_Tensor_slice_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
