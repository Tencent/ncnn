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

    def forward(self, x, y, z):
        out0 = F.interpolate(x, size=(x.size(2)//6), mode='nearest')
        out1 = F.interpolate(x, size=(y.size(1)*2), mode='linear', align_corners=True)
        out2 = F.interpolate(y, size=(z.size(3)+2,z.size(2)-2), mode='nearest')
        out3 = F.interpolate(z, size=(y.size(1)*2,y.size(1)*4), mode='bilinear', align_corners=True)
        out4 = F.interpolate(z, size=(y.size(3),y.size(2)), mode='bicubic', align_corners=False)

        return out0, out1, out2, out3, out4

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(1, 13, 120)
    y0 = torch.rand(1, 3, 32, 32)
    z0 = torch.rand(1, 1, 64, 48)

    x1 = torch.rand(1, 15, 40)
    y1 = torch.rand(1, 5, 14, 12)
    z1 = torch.rand(1, 2, 50, 44)

    a0 = net(x0, y0, z0)
    a1 = net(x1, y1, z1)

    # export torchscript
    mod = torch.jit.trace(net, (x0, y0, z0))
    mod.save("test_ncnn_interp_expr.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_interp_expr.pt inputshape=[1,13,120],[1,3,32,32],[1,1,64,48] inputshape2=[1,15,40],[1,5,14,12],[1,2,50,44]")

    # ncnn inference
    import numpy as np
    import ncnn
    b0 = []
    b1 = []
    with ncnn.Net() as net:
        net.load_param("test_ncnn_interp_expr.ncnn.param")
        net.load_model("test_ncnn_interp_expr.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(y0.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(z0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out2 = ex.extract("out2")
            _, out3 = ex.extract("out3")
            _, out4 = ex.extract("out4")
            b0.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out2)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out3)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out4)).unsqueeze(0))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(y1.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(z1.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out2 = ex.extract("out2")
            _, out3 = ex.extract("out3")
            _, out4 = ex.extract("out4")
            b1.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out2)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out3)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out4)).unsqueeze(0))

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
