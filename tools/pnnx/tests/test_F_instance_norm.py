# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.m3 = torch.rand(12)
        self.v3 = torch.rand(12)
        self.w3 = nn.Parameter(torch.rand(12))
        self.b3 = nn.Parameter(torch.rand(12))
        self.m4 = torch.rand(3)
        self.v4 = torch.rand(3)
        self.w4 = nn.Parameter(torch.rand(3))
        self.b4 = nn.Parameter(torch.rand(3))
        self.m5 = torch.rand(10)
        self.v5 = torch.rand(10)
        self.w5 = nn.Parameter(torch.rand(10))
        self.b5 = nn.Parameter(torch.rand(10))

    def forward(self, x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2):
        x = F.instance_norm(x, m0, v0, w0, b0)
        x = F.instance_norm(x, m0, v0, None, None)
        x = F.instance_norm(x, self.m3, self.v3, self.w3, self.b3)

        y = F.instance_norm(y, m1, v1, w1, b1, eps=1e-3)
        y = F.instance_norm(y, m1, v1, None, None)
        y = F.instance_norm(y, self.m4, self.v4, self.w4, self.b4)

        z = F.instance_norm(z, m2, v2, w2, b2)
        z = F.instance_norm(z, m2, v2, None, None, eps=1e-2)
        z = F.instance_norm(z, self.m5, self.v5, self.w5, self.b5)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)
    m0 = torch.rand(12)
    v0 = torch.rand(12)
    w0 = torch.rand(12)
    b0 = torch.rand(12)
    m1 = torch.rand(3)
    v1 = torch.rand(3)
    w1 = torch.rand(3)
    b1 = torch.rand(3)
    m2 = torch.rand(10)
    v2 = torch.rand(10)
    w2 = torch.rand(10)
    b2 = torch.rand(10)

    if version.parse(torch.__version__) >= version.parse('2.11'):
        # instance_norm updates running statistics even when net is in eval mode.
        m0_initial, v0_initial = m0.clone(), v0.clone()
        m1_initial, v1_initial = m1.clone(), v1.clone()
        m2_initial, v2_initial = m2.clone(), v2.clone()
        m3_initial, v3_initial = net.m3.clone(), net.v3.clone()
        m4_initial, v4_initial = net.m4.clone(), net.v4.clone()
        m5_initial, v5_initial = net.m5.clone(), net.v5.clone()

    a0, a1, a2 = net(x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2))
    mod.save("test_F_instance_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_instance_norm.pt inputshape=[1,12,24],[2,3,12,16],[1,10,12,16,24],[12],[12],[12],[12],[3],[3],[3],[3],[10],[10],[10],[10]")

    # pnnx inference
    import test_F_instance_norm_pnnx
    ts_b0, ts_b1, ts_b2 = test_F_instance_norm_pnnx.test_inference()

    if not (torch.equal(a0, ts_b0) and torch.equal(a1, ts_b1) and torch.equal(a2, ts_b2)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # Restore the running statistics used for the original reference.
    m0.copy_(m0_initial)
    v0.copy_(v0_initial)
    m1.copy_(m1_initial)
    v1.copy_(v1_initial)
    m2.copy_(m2_initial)
    v2.copy_(v2_initial)
    net.m3.copy_(m3_initial)
    net.v3.copy_(v3_initial)
    net.m4.copy_(m4_initial)
    net.v4.copy_(v4_initial)
    net.m5.copy_(m5_initial)
    net.v5.copy_(v5_initial)

    # export exported program
    ep = torch.export.export(net, (x, y, z, m0, v0, w0, b0, m1, v1, w1, b1, m2, v2, w2, b2))
    torch.export.save(ep, "test_F_instance_norm_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_F_instance_norm_export.pt2 fp16=0")

    # pnnx inference
    import test_F_instance_norm_export_pnnx
    c0, c1, c2 = test_F_instance_norm_export_pnnx.test_inference()

    return torch.equal(a0, c0) and torch.equal(a1, c1) and torch.equal(a2, c2)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
