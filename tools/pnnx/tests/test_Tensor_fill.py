# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x[:2,:].fill_(z[0])
        y[:1,:].fill_(0.22)
        return x + y.fill_(7)

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(6, 16)
    y = torch.rand(6, 16)
    z = torch.rand(1)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_Tensor_fill.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_Tensor_fill.pt inputshape=[6,16],[6,16],[1]")

    # pnnx inference
    import test_Tensor_fill_pnnx
    b = test_Tensor_fill_pnnx.test_inference()

    if not torch.equal(a, b):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # This model writes directly to the caller's x and y tensors. The pt2
    # frontend accepts inference graphs without external storage mutation.
    print("pt2 not applicable: Tensor.fill writes external input storage")
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
