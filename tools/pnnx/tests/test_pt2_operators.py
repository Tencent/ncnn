# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pnnx_test_utils


class PointModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, y):
        if self.op == "add_tensor":
            return x + y
        if self.op == "sub_tensor":
            return x - y
        if self.op == "mul_tensor":
            return x * y
        if self.op == "div_tensor":
            return x / y
        return torch.maximum(x, y)


class UnaryPointModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        if self.op == "add_scalar":
            return x + 1.25
        if self.op == "sub_scalar":
            return x - 0.75
        if self.op == "mul_scalar":
            return x * 1.5
        if self.op == "div_scalar":
            return x / 2.5
        if self.op == "abs":
            return torch.abs(x)
        if self.op == "neg":
            return torch.neg(x)
        if self.op == "exp":
            return torch.exp(x)
        if self.op == "log":
            return torch.log(x)
        if self.op == "sqrt":
            return torch.sqrt(x)
        if self.op == "sin":
            return torch.sin(x)
        if self.op == "cos":
            return torch.cos(x)
        if self.op == "tanh":
            return torch.tanh(x)
        if self.op == "sigmoid":
            return torch.sigmoid(x)
        if self.op == "relu":
            return torch.relu(x)
        if self.op == "gelu":
            return F.gelu(x)
        if self.op == "silu":
            return F.silu(x)
        if self.op == "clamp":
            return torch.clamp(x, -0.5, 0.5)
        return x


class ModuleModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class LinearChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(8, 12)
        self.linear1 = nn.Linear(12, 5)

    def forward(self, x):
        return self.linear1(torch.relu(self.linear0(x)))


class MatmulModel(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


class ShapeModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        if self.op == "reshape":
            return x.reshape(2, 3, 20)
        if self.op == "view":
            return x.view(4, 30)
        if self.op == "flatten":
            return torch.flatten(x, 1, 2)
        if self.op == "permute":
            return x.permute(0, 2, 3, 1)
        if self.op == "transpose":
            return torch.transpose(x, 1, 3)
        if self.op == "squeeze":
            return torch.squeeze(x, 1)
        if self.op == "unsqueeze":
            return torch.unsqueeze(x, 2)
        return x


class CatModel(nn.Module):
    def forward(self, x, y):
        return torch.cat((x, y), dim=1)


class ReductionModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        if self.op == "sum_all":
            return torch.sum(x)
        if self.op == "sum_dim":
            return torch.sum(x, dim=1)
        if self.op == "sum_dims":
            return torch.sum(x, dim=(1, 3))
        if self.op == "sum_keepdim":
            return torch.sum(x, dim=2, keepdim=True)
        if self.op == "mean_all":
            return torch.mean(x)
        if self.op == "mean_dim":
            return torch.mean(x, dim=2)
        if self.op == "amax":
            return torch.amax(x, dim=(2, 3))
        return torch.amin(x, dim=1)


def make_case(case):
    family, op = case.split("_", 1)
    if family == "point":
        x = torch.rand(2, 3, 4) + 0.25
        y = torch.rand(2, 3, 4) + 0.25
        if op in ("add_tensor", "sub_tensor", "mul_tensor", "div_tensor", "maximum"):
            return PointModel(op), (x, y)
        return UnaryPointModel(op), (x,)

    if family == "linear":
        if op == "2d_bias":
            return ModuleModel(nn.Linear(8, 5)), (torch.rand(4, 8),)
        if op == "2d_nobias":
            return ModuleModel(nn.Linear(8, 5, bias=False)), (torch.rand(4, 8),)
        if op == "3d_bias":
            return ModuleModel(nn.Linear(8, 5)), (torch.rand(1, 1, 8),)
        if op == "chain":
            return LinearChain(), (torch.rand(4, 8),)
        return MatmulModel(), (torch.rand(3, 4), torch.rand(4, 5))

    if family == "conv":
        if op == "1d":
            return ModuleModel(nn.Conv1d(4, 6, 3, padding=1)), (torch.rand(1, 4, 12),)
        if op == "1d_stride":
            return ModuleModel(nn.Conv1d(4, 6, 3, stride=2, padding=1)), (torch.rand(1, 4, 13),)
        if op == "2d":
            return ModuleModel(nn.Conv2d(4, 6, 3, padding=1)), (torch.rand(1, 4, 10, 12),)
        if op == "2d_nobias":
            return ModuleModel(nn.Conv2d(4, 6, 3, padding=1, bias=False)), (torch.rand(1, 4, 10, 12),)
        if op == "2d_stride":
            return ModuleModel(nn.Conv2d(4, 6, 3, stride=2, padding=1)), (torch.rand(1, 4, 11, 13),)
        if op == "2d_dilation":
            return ModuleModel(nn.Conv2d(4, 6, 3, padding=2, dilation=2)), (torch.rand(1, 4, 12, 12),)
        if op == "2d_groups":
            return ModuleModel(nn.Conv2d(4, 8, 3, padding=1, groups=2)), (torch.rand(1, 4, 10, 12),)
        if op == "2d_depthwise":
            return ModuleModel(nn.Conv2d(4, 4, 3, padding=1, groups=4)), (torch.rand(1, 4, 10, 12),)
        return ModuleModel(nn.Conv3d(3, 5, 3, padding=1)), (torch.rand(1, 3, 6, 7, 8),)

    if family == "norm":
        if op == "layer_1d":
            return ModuleModel(nn.LayerNorm(8)), (torch.rand(3, 8),)
        if op == "layer_2d":
            return ModuleModel(nn.LayerNorm((4, 5))), (torch.rand(2, 3, 4, 5),)
        if op == "batch_1d":
            return ModuleModel(nn.BatchNorm1d(4)), (torch.rand(2, 4, 8),)
        if op == "batch_2d":
            return ModuleModel(nn.BatchNorm2d(4)), (torch.rand(2, 4, 7, 8),)
        if op == "batch_3d":
            return ModuleModel(nn.BatchNorm3d(4)), (torch.rand(2, 4, 5, 6, 7),)
        return ModuleModel(nn.GroupNorm(2, 4)), (torch.rand(2, 4, 7, 8),)

    if family == "shape":
        if op == "squeeze":
            return ShapeModel(op), (torch.rand(2, 1, 3, 4),)
        if op == "unsqueeze":
            return ShapeModel(op), (torch.rand(1, 3, 4, 5),)
        if op == "cat":
            return CatModel(), (torch.rand(2, 3, 4, 5), torch.rand(2, 2, 4, 5))
        return ShapeModel(op), (torch.rand(2, 3, 4, 5),)

    if family == "reduction":
        return ReductionModel(op), (torch.rand(2, 3, 4, 5),)

    raise ValueError("unknown case " + case)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--format", choices=("torchscript", "pt2"), required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    return pnnx_test_utils.main(make_case, args.pnnx, args.workdir, args.format, args.case)


if __name__ == "__main__":
    raise SystemExit(main())
