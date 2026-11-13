# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn
import ncnn

class Model(nn.Module):
    def __init__(self, alpha, beta):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, mat1, mat2):
        return torch.addmm(input, mat1, mat2, alpha=self.alpha, beta=self.beta)

def test():
    torch.manual_seed(0)
    mat1 = torch.rand(2, 3)
    mat2 = torch.rand(3, 4)

    for i, (shape, alpha, beta) in enumerate((([4], 0, 1), ([2, 4], 0.0, 1.4), ([1, 4], -0.0, 0.0), ([2, 1], 0.0, -2.0))):
        input = torch.rand(shape)
        net = Model(alpha, beta).eval()
        mod = torch.jit.trace(net, (input, mat1, mat2))
        name = "test_torch_addmm_zero_alpha_%d" % i
        mod.save(name + ".pt")

        # Unsupported zero alpha must not become a Gemm with beta / alpha.
        if os.system("../../src/pnnx %s.pt inputshape=%s,[2,3],[3,4]" % (name, str(shape).replace(" ", ""))) != 0:
            return False
        with open(name + ".ncnn.param", "r") as f:
            param = f.read()
        if "torch.addmm" not in param:
            return False
        with ncnn.Net() as ncnn_net:
            if ncnn_net.load_param(name + ".ncnn.param") == 0:
                return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
