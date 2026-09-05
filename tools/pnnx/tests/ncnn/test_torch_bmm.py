# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a0, a1, q0, q1):
        a = torch.bmm(a0, a1)
        q = torch.bmm(q0, q1)
        return a, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    a0 = torch.rand(10, 23, 14)
    a1 = torch.rand(10, 14, 5)
    q0 = torch.rand(2, 7, 11)
    q1 = torch.rand(2, 11, 13)

    a = net(a0, a1, q0, q1)

    # export torchscript
    mod = torch.jit.trace(net, (a0, a1, q0, q1))
    mod.save("test_torch_bmm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_bmm.pt inputshape=[10,23,14],[10,14,5],[2,7,11],[2,11,13]")

    # ncnn inference
    import test_torch_bmm_ncnn
    b = test_torch_bmm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
