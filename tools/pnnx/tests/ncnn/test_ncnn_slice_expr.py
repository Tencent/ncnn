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

    def forward(self, x, y, z, q):
        out0 = x[x.size(0)//128:x.size(0)-3]
        out0 = out0.clone() * 2

        out1 = y[...,y.size(0)//8]
        out1 = out1.clone() * 3.3

        z = self.conv0(z)
        out2 = z[:,x.size(0)//64-y.size(1)//128,z.size(3)-3:-z.size(2)//7,z.size(2)//3]
        out2 = out2.clone() * 1.5

        out3 = z[...,x.size(0)//128:-z.size(2)//10,z.size(3)//8:-z.size(3)//8]
        out3 = out3.clone() * -10

        q = self.conv0(q)
        out4 = q[:,:,q.size(2)//3:q.size(2)-q.size(2)//3,q.size(3)//4:q.size(3)-q.size(3)//4]
        out4 = out4.clone() * 0.5

        return out0, out1, out2, out3, out4

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(128)
    y0 = torch.rand(64, 16)
    z0 = torch.rand(1, 3, 39, 16)
    q0 = torch.rand(2, 3, 39, 16)

    x1 = torch.rand(256)
    y1 = torch.rand(32, 128)
    z1 = torch.rand(1, 3, 15, 33)
    q1 = torch.rand(2, 3, 15, 33)

    a0 = net(x0, y0, z0, q0)
    a1 = net(x1, y1, z1, q1)

    # export torchscript
    if version.parse(torch.__version__) < version.parse('2.0'):
        mod = torch.jit.trace(net, (x0, y0, z0, q0))
    else:
        mod = torch.jit.trace(net, (x0, y0, z0, q0), _store_inputs=False)
    mod.save("test_ncnn_slice_expr.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_slice_expr.pt inputshape=[128],[64,16],[1,3,39,16],[2,3,39,16] inputshape2=[256],[32,128],[1,3,15,33],[2,3,15,33]")

    # ncnn inference
    import numpy as np
    import ncnn
    b0 = []
    b1 = []
    with ncnn.Net() as net:
        net.load_param("test_ncnn_slice_expr.ncnn.param")
        net.load_model("test_ncnn_slice_expr.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.numpy()).clone())
            ex.input("in1", ncnn.Mat(y0.numpy()).clone())
            ex.input("in2", ncnn.Mat(z0.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(q0.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out4 = ex.extract("out4")
            b0.append(torch.from_numpy(np.array(out0)))
            b0.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b0.append(torch.from_numpy(out4.numpy(batch_index=0)))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.numpy()).clone())
            ex.input("in1", ncnn.Mat(y1.numpy()).clone())
            ex.input("in2", ncnn.Mat(z1.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(q1.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out4 = ex.extract("out4")
            b1.append(torch.from_numpy(np.array(out0)))
            b1.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b1.append(torch.from_numpy(out4.numpy(batch_index=0)))

    for aa, bb in zip(a0[:2], b0[:2]):
        if not torch.allclose(aa, bb, 1e-4, 1e-4):
            return False

    if not torch.allclose(a0[4], b0[2], 1e-4, 1e-4):
        return False

    for aa, bb in zip(a1[:2], b1[:2]):
        if not torch.allclose(aa, bb, 1e-4, 1e-4):
            return False

    if not torch.allclose(a1[4], b1[2], 1e-4, 1e-4):
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
