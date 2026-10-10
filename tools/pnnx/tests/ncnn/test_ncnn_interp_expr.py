# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, qx, qy, qz):
        out0 = F.interpolate(x, size=(x.size(2)//6), mode='nearest')
        out1 = F.interpolate(x, size=(y.size(1)*2), mode='linear', align_corners=True)
        out2 = F.interpolate(y, size=(z.size(3)+2,z.size(2)-2), mode='nearest')
        out3 = F.interpolate(z, size=(y.size(1)*2,y.size(1)*4), mode='bilinear', align_corners=True)
        out4 = F.interpolate(z, size=(y.size(3),y.size(2)), mode='bicubic', align_corners=False)

        out5 = F.interpolate(qx, size=(qx.size(2)//6), mode='nearest')
        out6 = F.interpolate(qx, size=(qy.size(1)*2), mode='linear', align_corners=True)
        out7 = F.interpolate(qy, size=(qz.size(3)+2,qz.size(2)-2), mode='nearest')
        out8 = F.interpolate(qz, size=(qy.size(1)*2,qy.size(1)*4), mode='bilinear', align_corners=True)
        out9 = F.interpolate(qz, size=(qy.size(3),qy.size(2)), mode='bicubic', align_corners=False)

        return out0, out1, out2, out3, out4, out5, out6, out7, out8, out9

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(1, 13, 120)
    y0 = torch.rand(1, 3, 32, 32)
    z0 = torch.rand(1, 1, 64, 48)
    qx0 = torch.rand(2, 13, 120)
    qy0 = torch.rand(2, 3, 32, 32)
    qz0 = torch.rand(2, 1, 64, 48)

    x1 = torch.rand(1, 15, 40)
    y1 = torch.rand(1, 5, 14, 12)
    z1 = torch.rand(1, 2, 50, 44)
    qx1 = torch.rand(2, 15, 40)
    qy1 = torch.rand(2, 5, 14, 12)
    qz1 = torch.rand(2, 2, 50, 44)

    a0 = net(x0, y0, z0, qx0, qy0, qz0)
    a1 = net(x1, y1, z1, qx1, qy1, qz1)

    # export torchscript
    mod = torch.jit.trace(net, (x0, y0, z0, qx0, qy0, qz0))
    mod.save("test_ncnn_interp_expr.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_interp_expr.pt inputshape=[1,13,120],[1,3,32,32],[1,1,64,48],[2,13,120],[2,3,32,32],[2,1,64,48] inputshape2=[1,15,40],[1,5,14,12],[1,2,50,44],[2,15,40],[2,5,14,12],[2,2,50,44]")

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
            ex.input("in3", ncnn.Mat(qx0.numpy(), batch_index=0).clone())
            ex.input("in4", ncnn.Mat(qy0.numpy(), batch_index=0).clone())
            ex.input("in5", ncnn.Mat(qz0.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out2 = ex.extract("out2")
            _, out3 = ex.extract("out3")
            _, out4 = ex.extract("out4")
            _, out5 = ex.extract("out5")
            _, out6 = ex.extract("out6")
            _, out7 = ex.extract("out7")
            _, out8 = ex.extract("out8")
            _, out9 = ex.extract("out9")
            b0.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out2)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out3)).unsqueeze(0))
            b0.append(torch.from_numpy(np.array(out4)).unsqueeze(0))
            b0.append(torch.from_numpy(out5.numpy(batch_index=0)))
            b0.append(torch.from_numpy(out6.numpy(batch_index=0)))
            b0.append(torch.from_numpy(out7.numpy(batch_index=0)))
            b0.append(torch.from_numpy(out8.numpy(batch_index=0)))
            b0.append(torch.from_numpy(out9.numpy(batch_index=0)))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(y1.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(z1.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(qx1.numpy(), batch_index=0).clone())
            ex.input("in4", ncnn.Mat(qy1.numpy(), batch_index=0).clone())
            ex.input("in5", ncnn.Mat(qz1.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            _, out1 = ex.extract("out1")
            _, out2 = ex.extract("out2")
            _, out3 = ex.extract("out3")
            _, out4 = ex.extract("out4")
            _, out5 = ex.extract("out5")
            _, out6 = ex.extract("out6")
            _, out7 = ex.extract("out7")
            _, out8 = ex.extract("out8")
            _, out9 = ex.extract("out9")
            b1.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out1)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out2)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out3)).unsqueeze(0))
            b1.append(torch.from_numpy(np.array(out4)).unsqueeze(0))
            b1.append(torch.from_numpy(out5.numpy(batch_index=0)))
            b1.append(torch.from_numpy(out6.numpy(batch_index=0)))
            b1.append(torch.from_numpy(out7.numpy(batch_index=0)))
            b1.append(torch.from_numpy(out8.numpy(batch_index=0)))
            b1.append(torch.from_numpy(out9.numpy(batch_index=0)))

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
