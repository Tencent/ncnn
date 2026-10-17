# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x, y):
        x = F.max_pool2d(x, 1)
        y = F.max_pool2d(y, 1)
        return torch.cat((x, y), dim=0)


def shape_string(x):
    return str(list(x.shape)).replace(" ", "")


def compare_case(net, x, y, name):
    expected = net(x, y)

    with ncnn.Net() as ncnn_net:
        if ncnn_net.load_param(name + ".ncnn.param") != 0:
            return False
        if ncnn_net.load_model(name + ".ncnn.bin") != 0:
            return False

        with ncnn_net.create_extractor() as ex:
            if ex.input("in0", ncnn.Mat(x.numpy(), batch_index=0).clone()) != 0:
                return False
            if ex.input("in1", ncnn.Mat(y.numpy(), batch_index=0).clone()) != 0:
                return False

            ret, actual_mat = ex.extract("out0")
            if ret != 0:
                return False

            actual = torch.from_numpy(actual_mat.numpy(batch_index=0).copy())

    if actual.shape != expected.shape:
        print(name, "shape mismatch", actual.shape, expected.shape)
        return False
    if not torch.allclose(actual, expected, 1e-4, 1e-4):
        print(name, "max error", (actual - expected).abs().max().item())
        return False

    return True


def test():
    torch.manual_seed(0)
    net = Model().eval()
    x = torch.rand(2, 3, 5, 7)
    y = torch.rand(3, 3, 5, 7)
    x2 = torch.rand(1, 3, 5, 7)
    y2 = torch.rand(2, 3, 5, 7)

    name = "test_torch_cat_batch"
    mod = torch.jit.trace(net, (x, y))
    mod.save(name + ".pt")

    if os.system("../../src/pnnx " + name + ".pt inputshape=" + shape_string(x) + "," + shape_string(y) + " inputshape2=" + shape_string(x2) + "," + shape_string(y2) + " fp16=0") != 0:
        return False

    with open(name + ".ncnn.param") as f:
        if not any(line.startswith("Concat ") and " 0=-233" in line for line in f):
            print("missing batch concat param")
            return False

    if not compare_case(net, x, y, name):
        return False
    if not compare_case(net, x2, y2, name):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
