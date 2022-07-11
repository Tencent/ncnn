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

        self.conv = nn.Conv2d(in_channels=14, out_channels=24, kernel_size=1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def forward(self, x):
        x = self.conv(x)
        x0, x1 = self.channel_shuffle(x)
        return x0, x1

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 14, 5, 6)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_ncnn_fuse_shufflechannel_slice.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_shufflechannel_slice.pt inputshape=[1,14,5,6]")

    # ncnn inference
    import test_ncnn_fuse_shufflechannel_slice_ncnn
    b = test_ncnn_fuse_shufflechannel_slice_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
