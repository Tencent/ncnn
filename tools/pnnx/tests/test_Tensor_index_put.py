# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

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

    return test_model_formats(net, (x, y, z, w), a, "test_Tensor_index_put")

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
