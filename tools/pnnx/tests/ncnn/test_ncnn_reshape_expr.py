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
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1)

    def forward(self, x, y, z):
        x = x.reshape(x.size(0) // 128, y.size(1), -1)
        x = x * 2
        x = x.reshape(-1, y.size(0) // 2, 4)
        x = x * 3.3

        z = self.conv0(z)
        z = z.reshape(z.size(0), -1, z.size(3) * 3, z.size(2) // 3)
        z = self.conv1(z)
        z = z.permute(1, 0, 2, 3)
        z = z.reshape(-1, 1, z.size(2) // 3, z.size(3) * 3)
        z = z.permute(1, 0, 3, 2)
        z = self.conv2(z)
        z = z * 1.5

        return x, z

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(128)
    y0 = torch.rand(64, 16)
    z0 = torch.rand(1, 3, 39, 16)

    x1 = torch.rand(256)
    y1 = torch.rand(32, 128)
    z1 = torch.rand(1, 3, 15, 33)

    a0 = net(x0, y0, z0)
    a1 = net(x1, y1, z1)

    # export torchscript
    if version.parse(torch.__version__) < version.parse('2.0'):
        mod = torch.jit.trace(net, (x0, y0, z0))
    else:
        mod = torch.jit.trace(net, (x0, y0, z0), _store_inputs=False)
    mod.save("test_ncnn_reshape_expr.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_reshape_expr.pt inputshape=[128],[64,16],[1,3,39,16] inputshape2=[256],[32,128],[1,3,15,33]")

    # ncnn inference
    import numpy as np
    import ncnn
    b0 = []
    b1 = []
    with ncnn.Net() as net:
        net.load_param("test_ncnn_reshape_expr.ncnn.param")
        net.load_model("test_ncnn_reshape_expr.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.numpy()).clone())
            ex.input("in1", ncnn.Mat(y0.numpy()).clone())
            ex.input("in2", ncnn.Mat(z0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            b0.append(torch.from_numpy(np.array(out0)))
            b0.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.numpy()).clone())
            ex.input("in1", ncnn.Mat(y1.numpy()).clone())
            ex.input("in2", ncnn.Mat(z1.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            b1.append(torch.from_numpy(np.array(out0)))
            b1.append(torch.from_numpy(np.array(out1)).unsqueeze(0))

    for aa, bb in zip(a0, b0):
        if not torch.allclose(aa, bb, 1e-4, 1e-4):
            return False

    for aa, bb in zip(a1, b1):
        if not torch.allclose(aa, bb, 1e-4, 1e-4):
            return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
