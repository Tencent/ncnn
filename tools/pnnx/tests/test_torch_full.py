# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnnx_test_utils import test_model_formats

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        w = torch.full(x.size(), 3.0, dtype=torch.float64)
        x = torch.full(x.size(), 1.5)
        y = torch.full(y.size(), 3)
        z = torch.full(z.size(), -2.2)
        return x, y, z, w

class InputAnchoredModel(nn.Module):
    def __init__(self, model):
        super(InputAnchoredModel, self).__init__()
        self.model = model

    def forward(self, x, y, z):
        out = self.model(x, y, z)
        zx = x.sum() * 0
        zy = y.sum() * 0
        zz = z.sum() * 0
        # Keep the anchors from promoting the int64 factory result to float32.
        return (
            out[0] + zx.to(out[0].dtype),
            out[1] + zy.to(out[1].dtype),
            out[2] + zz.to(out[2].dtype),
            out[3] + zx.to(out[3].dtype),
        )

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 3, 16)
    y = torch.rand(1, 5, 9, 11)
    z = torch.rand(14, 8, 5, 9, 10)

    a = net(x, y, z)
    wrapped = InputAnchoredModel(net)
    wrapped.eval()

    assert tuple(t.dtype for t in a) == (torch.get_default_dtype(), torch.int64,
                                       torch.get_default_dtype(), torch.float64)
    anchored = wrapped(x, y, z)
    assert all(expected.dtype == actual.dtype and torch.equal(expected, actual)
               for expected, actual in zip(a, anchored))

    return test_model_formats(
        wrapped,
        (x, y, z),
        a,
        "test_torch_full",
    )

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
