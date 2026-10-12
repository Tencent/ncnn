# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, q):
        x = F.feature_alpha_dropout(x, training=False)
        y = F.feature_alpha_dropout(y, p=0.6, training=False)
        q = F.max_pool2d(q, 1)
        q = F.feature_alpha_dropout(q, training=False)
        x = F.relu(x)
        y = F.relu(y)
        q = F.relu(q)
        return x, y, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 6, 12, 16)
    y = torch.rand(5, 7, 9, 11)
    q = torch.rand(2, 3, 5, 7)

    a = net(x, y, q)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, q))
    mod.save("test_F_feature_alpha_dropout.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_F_feature_alpha_dropout.pt inputshape=[3,6,12,16],[5,7,9,11],[2,3,5,7]")

    # ncnn inference
    import test_F_feature_alpha_dropout_ncnn
    b = test_F_feature_alpha_dropout_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
