# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z, w):
        x = x.clone()
        z = z.clone()
        x = x.index_put(indices=[torch.tensor([10,2])], values=y, accumulate=False)
        z.index_put_(indices=[torch.tensor([1,0,0]), torch.tensor([3,2,1])], values=w, accumulate=True)

        x[torch.tensor([1], dtype=torch.int64)] = torch.tensor(45).float()
        x[torch.tensor([], dtype=torch.int64)] = torch.tensor(233).float()
        return x, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(12)
    y = torch.rand(2)
    z = torch.rand(6,9)
    w = torch.rand(3)

    a = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_Tensor_index_put.pt")

    # torchscript to pnnx
    import os
    os.system(os.path.join("..", "src", "pnnx") + " test_Tensor_index_put.pt inputshape=[12],[2],[6,9],[3]")

    # pnnx inference
    import test_Tensor_index_put_pnnx
    b = test_Tensor_index_put_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    ts_ok = True

    # pt2 path (torch.export fails, skip automatically)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x, y, z, w), ["[12]", "[2]", "[6,9]", "[3]"], "test_Tensor_index_put")

    return ts_ok and (pt2_ok is not False)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
