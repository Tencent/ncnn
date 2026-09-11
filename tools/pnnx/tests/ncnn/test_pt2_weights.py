# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from testutil_pt2 import run_pt2_test

ATOL = 1e-3


class MLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        return self.fc(x)


class MConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class MConv2dGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1, groups=2, bias=False)

    def forward(self, x):
        return self.conv(x)


class MBatchNorm2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(x)


class MLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        return self.ln(x)


class MLinearFloat16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 5).to(torch.float16)

    def forward(self, x):
        return self.fc(x)


class MLinearBFloat16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 5).to(torch.bfloat16)

    def forward(self, x):
        return self.fc(x)


class MConvBnRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


CASES = [
    ("test_pt2_w_linear", MLinear, "[1,4,8]", (torch.rand(1, 4, 8),), ATOL),
    ("test_pt2_w_conv2d", MConv2d, "[1,3,8,8]", (torch.rand(1, 3, 8, 8),), ATOL),
    ("test_pt2_w_conv2d_shape2", MConv2d, "[1,3,5,7]", (torch.rand(1, 3, 5, 7),), ATOL),
    ("test_pt2_w_conv2d_groups", MConv2dGroups, "[1,4,8,8]", (torch.rand(1, 4, 8, 8),), ATOL),
    ("test_pt2_w_batchnorm", MBatchNorm2d, "[1,4,8,8]", (torch.rand(1, 4, 8, 8),), ATOL),
    ("test_pt2_w_layernorm", MLayerNorm, "[1,4,8]", (torch.rand(1, 4, 8),), ATOL),
    ("test_pt2_w_linear_f16", MLinearFloat16, "[1,4,8]", (torch.rand(1, 4, 8, dtype=torch.float16),), 5e-4),
    ("test_pt2_w_linear_bf16", MLinearBFloat16, "[1,4,8]", (torch.rand(1, 4, 8, dtype=torch.bfloat16),), 4e-3),
    ("test_pt2_w_conv_bn_relu", MConvBnRelu, "[1,3,8,8]", (torch.rand(1, 3, 8, 8),), ATOL),
    ("test_pt2_w_smoke", None, "[1,3,4,4],[1,3,4,4]", None, ATOL),
]


def test():
    torch.manual_seed(0)
    failures = []
    for name, cls, shape_str, inputs, atol in CASES:
        if cls is None:
            from test_pt2_smoke import Model as SmokeModel
            x = torch.rand(1, 3, 4, 4)
            y = torch.rand(1, 3, 4, 4)
            ok = run_pt2_test(SmokeModel().eval(), (x, y), shape_str, name, atol)
        else:
            ok = run_pt2_test(cls().eval(), inputs, shape_str, name, atol)
        print(f"[pt2-w] {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    print(f"==== pt2 weights numerical crosscheck: {len(CASES) - len(failures)}/{len(CASES)} PASS ====")
    if failures:
        print("FAIL:", ", ".join(failures))
    return not failures


if __name__ == "__main__":
    exit(0 if test() else 1)
