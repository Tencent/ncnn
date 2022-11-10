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

    def forward(self, x):
        x = x.clone()
        x[2:10,...] += 1
        x[...,1] = x[...,-1] * 3
        y = x.clone()
        x[:,:,3,::2].clamp_(0, 0.5)
        x[:,:,3,::2] = x[:,:,3,::2].exp_()
        x[:,:,::2,:] = y[:,:,::2,:].pow(2)
        x[:,:,:,:] = x[:,:,:,:] / 2
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18, 15, 19, 20)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_slice_copy.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_slice_copy.pt inputshape=[18,15,19,20]")

    # pnnx inference
    import test_Tensor_slice_copy_pnnx
    b = test_Tensor_slice_copy_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
