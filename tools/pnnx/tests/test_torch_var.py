# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version


class Model(nn.Module):
    def forward(self, x):
        population = torch.var(x, dim=1, unbiased=False, keepdim=True)
        if version.parse(torch.__version__) >= version.parse("2.0"):
            sample = torch.var(x, dim=1, unbiased=True, keepdim=True)
            corrected = torch.var(x, dim=1, correction=2, keepdim=True)
            fractional_correction = torch.var(x, dim=1, correction=1.5, keepdim=True)
        else:
            sample = torch.var(x, dim=1, unbiased=True, keepdim=True)
            corrected = sample
            fractional_correction = sample
        return population, sample, corrected, fractional_correction


def test():
    net = Model().eval()

    torch.manual_seed(0)
    x = torch.rand(3, 6)

    expected = net(x)

    mod = torch.jit.trace(net, x)
    mod.save("test_torch_var.pt")

    import os

    command = "../src/pnnx test_torch_var.pt inputshape=[3,6]"
    result = os.system(command)
    if result != 0:
        raise RuntimeError(f"pnnx conversion failed with status {result}: {command}")

    import test_torch_var_pnnx

    actual = test_torch_var_pnnx.test_inference()
    if len(expected) != len(actual):
        return False
    return all(
        torch.equal(expected_value, actual_value)
        for expected_value, actual_value in zip(expected, actual)
    )


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
