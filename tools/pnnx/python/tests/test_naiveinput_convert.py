# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pnnx

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x

def test_export():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 16)

    a0 = net(x)

    mod = torch.jit.trace(net, x)
    mod.save("test_F_relu_nconvert.pt")

    net2 = pnnx.convert("test_F_relu_nconvert.pt", x)

    b0 = net2(x)

    assert torch.equal(a0, b0)

@pytest.mark.parametrize("fp16", [True, False])
def test_convert_input_shapes_respects_fp16(tmp_path, monkeypatch, fp16):
    monkeypatch.chdir(tmp_path)
    net = nn.Linear(4, 3).eval()
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.001, 2.002, 3.003, 4.004]] * 3))
        net.bias.fill_(0.125)
    x = torch.ones(1, 4)
    torch.jit.trace(net, x).save("model.pt")

    pnnx.convert("model.pt", inputs=x, fp16=fp16, ncnnbin="reference.bin")
    pnnx.convert(
        "model.pt",
        input_shapes=[[1, 4]],
        input_types=["f32"],
        fp16=fp16,
        ncnnbin="observed.bin",
    )

    assert (tmp_path / "reference.bin").read_bytes() == (
        tmp_path / "observed.bin"
    ).read_bytes()
