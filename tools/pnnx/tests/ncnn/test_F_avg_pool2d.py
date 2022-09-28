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
        y = x.reshape(12, 128, 127)

        x = F.avg_pool2d(x, kernel_size=3)
        x = F.avg_pool2d(x, kernel_size=4, stride=2, padding=2)
        x = F.avg_pool2d(x, kernel_size=(1,3), stride=1, padding=(0,1), ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=(4,5), stride=(1,2), padding=(1,2), ceil_mode=True, count_include_pad=False)
        x = F.avg_pool2d(x, kernel_size=(5,3), stride=(2,1), padding=1, ceil_mode=False, count_include_pad=True)
        x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        #x = F.avg_pool2d(x, kernel_size=(5,4), stride=1, padding=2, ceil_mode=False, count_include_pad=False, divisor_override=18)

        y = F.avg_pool2d(y, kernel_size=3)
        y = F.avg_pool2d(y, kernel_size=4, stride=2, padding=2)
        y = F.avg_pool2d(y, kernel_size=(1,3), stride=1, padding=(0,1), ceil_mode=False, count_include_pad=True)
        y = F.avg_pool2d(y, kernel_size=(4,5), stride=(1,2), padding=(1,2), ceil_mode=True, count_include_pad=False)
        y = F.avg_pool2d(y, kernel_size=(5,3), stride=(2,1), padding=1, ceil_mode=False, count_include_pad=True)
        y = F.avg_pool2d(y, kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        #y = F.avg_pool2d(y, kernel_size=(5,4), stride=1, padding=2, ceil_mode=False, count_include_pad=False, divisor_override=18)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128, 127)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_F_avg_pool2d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_avg_pool2d.pt inputshape=[1,12,128,127]")

    # ncnn inference
    import test_F_avg_pool2d_ncnn
    b = test_F_avg_pool2d_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
