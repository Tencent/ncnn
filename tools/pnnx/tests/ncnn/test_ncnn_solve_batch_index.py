# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = x.permute(2, 3, 0, 1)
        x = x.reshape(-1, x.size(0) // 8, 1, 3, 5)
        x = x.transpose(0, 1)
        x = x.reshape(-1, 1, 15)
        x = x.unsqueeze(-1)
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(1, -1)
        x = x.unsqueeze(0)
        x = x.relu()
        return x

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(1, 3, 32, 32)

    x1 = torch.rand(1, 3, 64, 64)

    a0 = net(x0)
    a1 = net(x1)

    # export torchscript
    mod = torch.jit.trace(net, x0)
    mod.save("test_ncnn_solve_batch_index.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_solve_batch_index.pt inputshape=[1,3,32,32] inputshape2=[1,3,64,64]")

    # ncnn inference
    import numpy as np
    import ncnn
    with ncnn.Net() as net:
        net.load_param("test_ncnn_solve_batch_index.ncnn.param")
        net.load_model("test_ncnn_solve_batch_index.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            b0 = torch.from_numpy(np.array(out0)).unsqueeze(0)

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            b1 = torch.from_numpy(np.array(out0)).unsqueeze(0)

    return torch.allclose(a0, b0, 1e-4, 1e-4) and torch.allclose(a1, b1, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
