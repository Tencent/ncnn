# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Reproduce the pt2 legacy(<2.8) fixtures exported with torch 2.7.1.
# Run with a torch 2.7.x interpreter, e.g.
#   python3 -m venv /tmp/pt27 && /tmp/pt27/bin/pip install \
#       --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
#   /tmp/pt27/bin/python generate.py

import os
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))


class M1(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(8, 8))
        self.b = nn.Parameter(torch.randn(8))
        self.register_buffer("pbuf", torch.randn(8))
        self.register_buffer("nbuf", torch.randn(8), persistent=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.w, self.b) + self.pbuf + self.nbuf


class M2(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.fc = nn.Linear(8, 8)
        self.c = torch.arange(8, dtype=torch.float32)

    def forward(self, x):
        return self.fc(x) + self.c


class M3(nn.Module):
    # conv weights (bias optional)
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.conv = nn.Conv2d(1, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class M4(nn.Module):
    # bfloat16 weights
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.fc = nn.Linear(8, 8, dtype=torch.bfloat16)

    def forward(self, x):
        return self.fc(x)


class M5(nn.Module):
    # float16 weights
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.fc = nn.Linear(8, 8, dtype=torch.float16)

    def forward(self, x):
        return self.fc(x)


class M6(nn.Module):
    # int64 vector buffer + int64 scalar buffer + an int64 tensor attribute
    # that actually participates in the graph (tensor_constant payload)
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.fc = nn.Linear(8, 8)
        self.register_buffer("idx", torch.arange(8, dtype=torch.int64))
        self.register_buffer("cnt", torch.tensor(3, dtype=torch.int64))
        self.lut = torch.arange(64, dtype=torch.int64).reshape(8, 8)

    def forward(self, x):
        # cnt.index_add-style use keeps cnt as a live scalar buffer; lut feeds
        # a runtime op so it is serialized as a tensor_constant
        return self.fc(x) + self.idx.float() + self.lut.float().mean(0) + self.cnt.float()


def main():
    torch.manual_seed(7)
    x = torch.randn(4, 8)
    x_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
    x_f16 = torch.randn(4, 8, dtype=torch.float16)
    x_img = torch.randn(1, 1, 8, 8)
    models = (
        ("linear_params", M1(), x),
        ("linear_const", M2(), x),
        ("conv_bn_params", M3(), x_img),
        ("bf16_weights", M4(), x_bf16),
        ("f16_weights", M5(), x_f16),
        ("i64_values", M6(), x),
    )
    for tag, m, xin in models:
        m.eval()
        ep = torch.export.export(m, (xin,))
        path = os.path.join(OUT, tag + "_pt2_7.pt2")
        torch.export.save(ep, path)
        print("wrote", path, os.path.getsize(path), "bytes; torch", torch.__version__)


if __name__ == "__main__":
    main()
