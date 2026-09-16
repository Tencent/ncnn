# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnnx_test_utils import convert_and_import_ncnn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.ln_0 = nn.LayerNorm(64)
        self.ln_0.weight = nn.Parameter(torch.rand(64))
        self.ln_0.bias = nn.Parameter(torch.rand(64))
        self.ln_1 = nn.LayerNorm(normalized_shape=(24,64), eps=1e-2, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(normalized_shape=(3,24,64), eps=1e-3)

    def forward(self, x, y, w, q):
        x = self.ln_0(x)
        y = self.ln_0(y)
        z = self.ln_1(y)
        w0 = self.ln_0(w)
        w1 = self.ln_1(w)
        w2 = self.ln_2(w)
        q = self.ln_0(q)
        return x, y, z, w0, w1, w2, q

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 24, 64)
    y = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 2, 3, 24, 64)
    q = torch.rand(2, 24, 64)

    a = net(x, y, w, q)

    module = convert_and_import_ncnn(net, (x, y, w, q), "test_nn_LayerNorm", pnnx_args=("inputshape=[1,24,64],[1,12,24,64],[1,2,3,24,64],[2,24,64]",))

    # ncnn inference
    b = module.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-3, 1e-3):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
