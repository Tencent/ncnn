# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import math
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
        if proc.returncode != 0 or b"unsupported torch.addmm alpha=0" not in proc.stderr:
            return False
        with open(name + ".ncnn.param", "r") as f:
            layers = [line.split() for line in f.readlines()[2:] if line.strip()]
        if not any(layer[0] == "Gemm" for layer in layers) or any(layer[0] == "torch.addmm" for layer in layers):
            return False
        gemm = next(layer for layer in layers if layer[0] == "Gemm")
        params = dict(token.split("=", 1) for token in gemm if "=" in token)
        if "0" not in params or "1" not in params:
            return False
        if not all(math.isfinite(float(value)) for value in params.values()):
            return False
        if float(params["0"]) != 0.0 or not math.isclose(float(params["1"]), beta, rel_tol=1e-6, abs_tol=1e-6):
            return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
