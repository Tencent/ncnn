# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# M4b 数值对拍：权重字节链路（state_dict -> zip data/N -> Pt2Value.data -> pnnx.Attribute -> .bin）
# 的首次端到端验证。结构对拍（.param diff）验不出权重字节错误，只有 pyncnn 推理
# allclose(<1e-3) 能证明 state_dict 字节原样到达 ncnn。
# 覆盖：linear / conv2d / grouped conv2d / batchnorm(running stats) / layernorm /
#       conv+bn+relu 融合链 / smoke 对照（无权重）。

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
    # groups=2：weight (8,2,3,3)，对权重布局语义要求最高
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


class MConvBnRelu(nn.Module):
    # 融合链：conv(4D 权重+bias) + bn(4 组 buffer) + relu
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


CASES = [
    ("test_pt2_w_linear", MLinear, "[1,4,8]", (torch.rand(1, 4, 8),)),
    ("test_pt2_w_conv2d", MConv2d, "[1,3,8,8]", (torch.rand(1, 3, 8, 8),)),
    ("test_pt2_w_conv2d_groups", MConv2dGroups, "[1,4,8,8]", (torch.rand(1, 4, 8, 8),)),
    ("test_pt2_w_batchnorm", MBatchNorm2d, "[1,4,8,8]", (torch.rand(1, 4, 8, 8),)),
    ("test_pt2_w_layernorm", MLayerNorm, "[1,4,8]", (torch.rand(1, 4, 8),)),
    ("test_pt2_w_conv_bn_relu", MConvBnRelu, "[1,3,8,8]", (torch.rand(1, 3, 8, 8),)),
    ("test_pt2_w_smoke", None, "[1,3,4,4],[1,3,4,4]", None),  # 对照组见 test()
]


def test():
    torch.manual_seed(0)
    failures = []
    for name, cls, shape_str, inputs in CASES:
        if cls is None:  # smoke 对照组（无权重）
            from test_pt2_smoke import Model as SmokeModel
            x = torch.rand(1, 3, 4, 4)
            y = torch.rand(1, 3, 4, 4)
            ok = run_pt2_test(SmokeModel().eval(), (x, y), shape_str, name, ATOL)
        else:
            ok = run_pt2_test(cls().eval(), inputs, shape_str, name, ATOL)
        print(f"[pt2-w] {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    print(f"==== pt2 weights numerical crosscheck: {len(CASES) - len(failures)}/{len(CASES)} PASS ====")
    if failures:
        print("FAIL:", ", ".join(failures))
    return not failures


if __name__ == "__main__":
    exit(0 if test() else 1)
