# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.instance_norm(x, None, None, None, None, True, 0.1, 1e-5)
        y = F.instance_norm(y, None, None, None, None, True, 0.1, 1e-3)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 10, 12, 16)
    y = torch.rand(1, 10, 3, 12, 16)

    a = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_instance_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_instance_norm.pt inputshape=[1,10,12,16],[1,10,3,12,16]")

    # ncnn inference
    import test_F_instance_norm_ncnn
    b = test_F_instance_norm_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
