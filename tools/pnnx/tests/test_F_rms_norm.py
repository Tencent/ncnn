# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class T5LayerNorm_without_gamma(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.w3 = nn.Parameter(torch.rand(24))
        self.w4 = nn.Parameter(torch.rand(12, 16))
        self.w5 = nn.Parameter(torch.rand(24))

        self.rmsnorm = T5LayerNorm(66)
        self.rmsnorm_2 = T5LayerNorm_without_gamma(66)

    def forward(self, x, y, z, w0, w1, w2, x2):
        x = F.rms_norm(x, (24,), w0)
        x = F.rms_norm(x, (12,24), None)
        x = F.rms_norm(x, (24,), self.w3)

        y = F.rms_norm(y, (16,), None, eps=1e-3)
        y = F.rms_norm(y, (12,16), w1)
        y = F.rms_norm(y, (12,16), self.w4)

        z = F.rms_norm(z, (24,), w2)
        z = F.rms_norm(z, (12,16,24), None, eps=1e-2)
        z = F.rms_norm(z, (24,), self.w5)

        x2 = self.rmsnorm(x2)
        x2 = self.rmsnorm_2(x2)
        return x, y, z, x2

def test():
    if version.parse(torch.__version__) < version.parse('2.4'):
        return True

    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24)
    y = torch.rand(2, 3, 12, 16)
    z = torch.rand(1, 10, 12, 16, 24)
    w0 = torch.rand(24)
    w1 = torch.rand(12, 16)
    w2 = torch.rand(24)
    x2 = torch.rand(3, 22, 66)

    a = net(x, y, z, w0, w1, w2, x2)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w0, w1, w2, x2))
    mod.save("test_F_rms_norm.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_rms_norm.pt inputshape=[1,12,24],[2,3,12,16],[1,10,12,16,24],[24],[12,16],[24],[3,22,66]")

    # pnnx inference
    import test_F_rms_norm_pnnx
    b = test_F_rms_norm_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
