# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import torch
import torch.nn as nn

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

        # Keep conversion available; zero-alpha inference remains unsupported.
        proc = subprocess.run(
            ["../../src/pnnx", name + ".pt", "inputshape=%s,[2,3],[3,4]" % str(shape).replace(" ", "")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or b"addmm with alpha=0 is not supported yet" not in proc.stderr:
            return False
        with open(name + ".ncnn.param", "r") as f:
            layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
        if "Gemm" not in layers or "torch.addmm" in layers:
            return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
