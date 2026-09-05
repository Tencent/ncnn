# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = F.feature_alpha_dropout(x, training=False)
        y = F.feature_alpha_dropout(y, p=0.6, training=False)
        return x, y

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 12, 16)
    y = torch.rand(1, 5, 7, 9, 11)

    a0, a1 = net(x, y)

    # export torchscript
    mod = torch.jit.trace(net, (x, y))
    mod.save("test_F_feature_alpha_dropout.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_F_feature_alpha_dropout.pt inputshape=[1,3,4,12,16],[1,5,7,9,11]")

    # pnnx inference
    import test_F_feature_alpha_dropout_pnnx
    b0, b1 = test_F_feature_alpha_dropout_pnnx.test_inference()

    ts_ok = torch.equal(a0, b0) and torch.equal(a1, b1)

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y), ["[1,3,4,12,16]", "[1,5,7,9,11]"], "test_F_feature_alpha_dropout")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
