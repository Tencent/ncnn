# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w1 = nn.Parameter(torch.rand(10, 128))

    def forward(self, y):
        y = F.embedding(y, self.w1)
        return y

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    y = torch.randint(10, (1, 11), dtype=torch.int)

    a = net(y)

    # export torchscript
    mod = torch.jit.trace(net, (y))
    mod.save("test_F_embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_embedding.pt inputshape=[1,11]i32")

    # ncnn inference
    import test_F_embedding_ncnn
    b = test_F_embedding_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
