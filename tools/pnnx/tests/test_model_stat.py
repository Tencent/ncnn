# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(6)
        self.prelu = nn.PReLU(6)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(24, 8)
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.prelu(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.ln(x)
        x = F.gelu(x)

        x = x.reshape(1, 2, 4)
        x = torch.matmul(x, x.transpose(-1, -2))
        x = F.softmax(x, dim=-1)
        x = F.pad(x, (1, 1))
        return x


class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()

    def forward(self, x, y, weight, bias):
        x = x + y
        x = torch.matmul(x, weight)
        x = x + bias
        x = F.relu(x)
        return x


class ExpressionModel(nn.Module):
    def __init__(self):
        super(ExpressionModel, self).__init__()

    def forward(self, x):
        y = torch.sin(x) + torch.cos(x) * torch.sqrt(torch.abs(x) + 1)
        return y


class AddmmModel(nn.Module):
    def __init__(self):
        super(AddmmModel, self).__init__()

    def forward(self, x, y, z):
        return torch.addmm(x, y, z)


def _format_ops(ops):
    units = ["", "K", "M", "G", "T", "P"]
    unit_index = 0
    while ops >= 1000 and unit_index + 1 < len(units):
        ops = ops / 1000
        unit_index = unit_index + 1

    if unit_index == 0:
        return "%.0f" % ops

    return ("%.3f" % ops).rstrip("0").rstrip(".") + units[unit_index]


def _check_stat_text(text, expected_inputshape, expected_flops, expected_memops):
    inputshape = re.search(r"(?:^|\n)#? ?inputshape = (.+)", text)
    flops = re.search(r"(?:^|\n)#? ?FLOPS = ([0-9.]+[KMGTPE]?)", text)
    memops = re.search(r"(?:^|\n)#? ?memory OPS = ([0-9.]+[KMGTPE]?)", text)
    if inputshape is None or flops is None or memops is None:
        return False

    return inputshape.group(1) == expected_inputshape and \
        flops.group(1) == _format_ops(expected_flops) and \
        memops.group(1) == _format_ops(expected_memops)


def _run_case(name, net, inputs, inputshape, expected_inputshape, expected_flops, expected_memops):
    net.eval()

    torch.manual_seed(0)
    inputs = tuple(torch.rand(shape) for shape in inputs)

    a = net(*inputs)

    # export torchscript
    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    # torchscript to pnnx
    cmd = ["../src/pnnx", name + ".pt", "inputshape=" + inputshape]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return False

    if not _check_stat_text(p.stdout + p.stderr, expected_inputshape, expected_flops, expected_memops):
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return False

    with open(name + "_pnnx.py", "r") as f:
        pnnx_py = f.read()

    if "# pnnx model stat" not in pnnx_py:
        return False
    if not _check_stat_text(pnnx_py, expected_inputshape, expected_flops, expected_memops):
        return False

    # pnnx inference
    pnnx_module = __import__(name + "_pnnx")
    b = pnnx_module.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)


def test():
    if not _run_case("test_model_stat", Model(),
                     ((1, 3, 8, 8),),
                     "[1,3,8,8]", "[1,3,8,8]f32",
                     23708, 4246):
        return False

    if not _run_case("test_model_stat_multi_input", MultiInputModel(),
                     ((2, 3), (2, 3), (3, 4), (4,)),
                     "[2,3],[2,3],[3,4],[4]",
                     "[2,3]f32,[2,3]f32,[3,4]f32,[4]f32",
                     70, 80):
        return False

    if not _run_case("test_model_stat_expression", ExpressionModel(),
                     ((2, 3),),
                     "[2,3]", "[2,3]f32",
                     96, 12):
        return False

    if not _run_case("test_model_stat_addmm", AddmmModel(),
                     ((2, 4), (2, 3), (3, 4)),
                     "[2,4],[2,3],[3,4]",
                     "[2,4]f32,[2,3]f32,[3,4]f32",
                     56, 34):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
