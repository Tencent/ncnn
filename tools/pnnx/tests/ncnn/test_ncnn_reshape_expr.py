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

    def forward(self, x, y):
        x = x.reshape(x.size(0) // 128, y.size(1), -1)
        return x * 3.3

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(128)
    y0 = torch.rand(16, 64)

    x1 = torch.rand(256)
    y1 = torch.rand(32, 128)

    a0 = net(x0, y0)
    a1 = net(x1, y1)

    # export torchscript
    mod = torch.jit.trace(net, (x0, y0), _store_inputs=False)
    mod.save("test_ncnn_reshape_expr.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_reshape_expr.pt inputshape=[128],[16,64] inputshape2=[256],[32,128]")

    # ncnn inference
    import numpy as np
    import ncnn
    with ncnn.Net() as net:
        net.load_param("test_ncnn_reshape_expr.ncnn.param")
        net.load_model("test_ncnn_reshape_expr.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.numpy()).clone())
            ex.input("in1", ncnn.Mat(y0.numpy()).clone())

            _, out0 = ex.extract("out0")
            b0 = torch.from_numpy(np.array(out0))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.numpy()).clone())
            ex.input("in1", ncnn.Mat(y1.numpy()).clone())

            _, out0 = ex.extract("out0")
            b1 = torch.from_numpy(np.array(out0))

    return torch.equal(a0, b0) and torch.equal(a1, b1)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
