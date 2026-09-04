# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from pnnx_test_utils import torchscript_to_pnnx


class Model(nn.Module):
    def forward(self, x):
        return torch.tril(x), torch.tril(x, diagonal=2), torch.tril(x, diagonal=-1)


def test():
    net = Model().eval()

    torch.manual_seed(0)
    x = torch.rand(2, 5, 7)
    expected = net(x)

    traced = torch.jit.trace(net, x)
    traced.save("test_torch_tril.pt")

    converted = torchscript_to_pnnx("test_torch_tril", "[2,5,7]")
    actual = converted(x)
    return all(torch.equal(a, b) for a, b in zip(expected, actual))


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
