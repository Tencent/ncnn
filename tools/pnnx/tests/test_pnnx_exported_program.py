# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys

import torch
import torch.nn as nn

from pnnx_test_utils import export_convert_import, has_exported_program


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        y = self.linear(x)
        return torch.relu(y) * self.scale, y


def test():
    if not has_exported_program():
        print("SKIP: torch.export.save is unavailable in torch " + torch.__version__)
        return True

    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 3)
    expected = model(x)

    torchscript_model = export_convert_import(model, (x,), "test_pnnx_exported_program", "torchscript")
    pt2_model = export_convert_import(model, (x,), "test_pnnx_exported_program", "pt2")

    torchscript_output = torchscript_model(x)
    pt2_output = pt2_model(x)
    return all(
        torch.allclose(eager, torchscript)
        and torch.allclose(eager, exported)
        and torch.allclose(torchscript, exported)
        for eager, torchscript, exported in zip(expected, torchscript_output, pt2_output)
    )


if __name__ == "__main__":
    sys.exit(0 if test() else 1)