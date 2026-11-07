# Copyright 2026 Futz12 <pchar.cn>
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Model(nn.Module):
    def __init__(self, act_type):
        super(Model, self).__init__()
        self.conv_0 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1, groups=12)
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        elif act_type == 'clip':
            self.act = nn.Hardtanh(min_val=-0.5, max_val=0.5)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'mish':
            self.act = nn.Mish()
        elif act_type == 'hardswish':
            self.act = nn.Hardswish()
        elif act_type == 'gelu':
            self.act = nn.GELU()
        elif act_type == 'silu':
            self.act = nn.SiLU()
        elif act_type == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif act_type == 'selu':
            self.act = nn.SELU()

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        return x

def test_act(act_type):
    net = Model(act_type)
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64)

    a = net(x)

    pt_name = f"test_ncnn_fuse_convolutiondepthwise1d_{act_type}.pt"
    mod = torch.jit.trace(net, x)
    mod.save(pt_name)

    os.system(f"../../src/pnnx {pt_name} inputshape=[1,12,64]")

    ncnn_module = __import__(f"test_ncnn_fuse_convolutiondepthwise1d_{act_type}_ncnn")
    b = ncnn_module.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

def test():
    act_types = ['relu', 'leakyrelu', 'clip', 'sigmoid', 'mish', 'hardswish', 'gelu', 'silu', 'elu', 'selu']
    for act_type in act_types:
        print(f"testing {act_type}...")
        if not test_act(act_type):
            print(f"{act_type} failed!")
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
