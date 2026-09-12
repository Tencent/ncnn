# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        out0 = x.new_empty((2,2))
        out1 = x.new_empty(3)
        out2 = x.new_empty((4,5,6,7,8))
        out3 = x.new_empty((1,2,1))
        out4 = x.new_empty((3,3,3,3), dtype=torch.long)
        return out0, out1, out2, out3, out4

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_Tensor_new_empty.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_Tensor_new_empty.pt inputshape=[1,16]")

    # pnnx inference
    import test_Tensor_new_empty_pnnx
    b = test_Tensor_new_empty_pnnx.test_inference()

    # test shape only for uninitialized data
    for a0, b0 in zip(a, b):
        if not a0.shape == b0.shape:
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically); new_empty is
    # uninitialized, compare shape and dtype only like the torchscript path
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[1,16]"], "test_Tensor_new_empty", shape_only=True)

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
