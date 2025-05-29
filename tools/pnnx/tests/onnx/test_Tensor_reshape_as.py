# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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
        x = x.reshape_as(y)
        y = y.reshape_as(z)
        z = z.reshape_as(x)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(6, 2, 2, 2)
    z = torch.rand(48)

    a = net(x, y, z)

    # export onnx
    torch.onnx.export(net, (x, y, z), "test_Tensor_reshape_as.onnx", input_names = ['x','y','z'], output_names = ['a','b','c'],
                      dynamic_axes={'x' : {0 : 'x0', 1 : 'x1', 2 : 'x2'}, 'y' : {0 : 'y0', 1 : 'y1', 2 : 'y2', 3 : 'y3'}, 'z' : {0 : 'z0'}})

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_Tensor_reshape_as.onnx inputshape=[1,3,16],[6,2,2,2],[48] inputshape2=[1,3,8],[6,1,2,2],[24]")

    # pnnx inference
    import test_Tensor_reshape_as_pnnx
    netb = test_Tensor_reshape_as_pnnx.Model().float().eval()
    b = netb(x, y, z)

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
