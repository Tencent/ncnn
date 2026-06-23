# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


class ModelMiddleBatch(nn.Module):
    def __init__(self):
        super(ModelMiddleBatch, self).__init__()

    def forward(self, x):
        x = x.unflatten(dim=0, sizes=(3, 2))
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelReshapeMiddleBatch(nn.Module):
    def __init__(self):
        super(ModelReshapeMiddleBatch, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelMiddleBatchWithOrdinaryPermute(nn.Module):
    def __init__(self):
        super(ModelMiddleBatchWithOrdinaryPermute, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = x.permute(1, 2, 0, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelBatchToMiddleOutput(nn.Module):
    def __init__(self):
        super(ModelBatchToMiddleOutput, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        x = x.permute(1, 0, 2, 3)
        return x


class ModelBatchToMiddleOutputSameDim(nn.Module):
    def __init__(self):
        super(ModelBatchToMiddleOutputSameDim, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        x = x.permute(1, 0, 2, 3)
        return x


class ModelBatchMiddleRoundTrip(nn.Module):
    def __init__(self):
        super(ModelBatchMiddleRoundTrip, self).__init__()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelFlattenRoundTrip(nn.Module):
    def __init__(self):
        super(ModelFlattenRoundTrip, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 0, 1)
        x = x.reshape(2, 3, 5, 7)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelTwoBatchAxisReshapes(nn.Module):
    def __init__(self):
        super(ModelTwoBatchAxisReshapes, self).__init__()

    def forward(self, x, y):
        x = x.reshape(3, 2, 5, 7).permute(1, 0, 2, 3)
        y = y.reshape(4, 2, 3, 5).permute(1, 0, 2, 3)
        return F.max_pool2d(x, 3, stride=1, padding=1), F.max_pool2d(y, 3, stride=1, padding=1)


class ModelComputeBarrier(nn.Module):
    def __init__(self):
        super(ModelComputeBarrier, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        x = F.relu(x)
        x = x.permute(1, 0, 2, 3)
        x = F.max_pool2d(x, 3, stride=1, padding=1)
        return x


class ModelMultiConsumer(nn.Module):
    def __init__(self):
        super(ModelMultiConsumer, self).__init__()

    def forward(self, x):
        x = x.reshape(3, 2, 5, 7)
        y = x.permute(1, 0, 2, 3)
        z = x.reshape(3, 2, 35)
        return y, z


def compare(a, b):
    if isinstance(a, tuple):
        for a0, b0 in zip(a, b):
            if not torch.equal(a0, b0):
                return False
        return True

    return torch.equal(a, b)


def run_model(name, net, inputs, checks=None):
    net.eval()

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    a = net(*inputs)

    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    inputshape = ",".join([str(list(x.shape)).replace(" ", "") for x in inputs])
    os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape)

    if checks:
        with open(name + ".ncnn.param") as f:
            text = f.read()
            for s in checks:
                if s not in text:
                    return False

    ncnnpy = __import__(name + "_ncnn")
    b = ncnnpy.test_inference()

    return compare(a, b)


def test():
    if version.parse(torch.__version__) >= version.parse('1.13'):
        torch.manual_seed(0)
        x = torch.rand(6, 5, 7)
        if not run_model("test_ncnn_batch_layout_middle_batch", ModelMiddleBatch(), x, ["12=2", "13=1"]):
            return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_reshape_middle_batch", ModelReshapeMiddleBatch(), x, ["12=2", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_ordinary_permute", ModelMiddleBatchWithOrdinaryPermute(), x, ["12=2", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_batch_to_middle", ModelBatchToMiddleOutput(), x, ["12=1", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 2, 5, 7)
    if not run_model("test_ncnn_batch_layout_batch_to_middle_same_dim", ModelBatchToMiddleOutputSameDim(), x, ["12=1", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_roundtrip", ModelBatchMiddleRoundTrip(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 7)
    if not run_model("test_ncnn_batch_layout_flatten_roundtrip", ModelFlattenRoundTrip(), x):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    y = torch.rand(8, 3, 5)
    if not run_model("test_ncnn_batch_layout_two_reshapes", ModelTwoBatchAxisReshapes(), (x, y), ["12=2", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_compute_barrier", ModelComputeBarrier(), x, ["12=2", "13=1"]):
        return False

    torch.manual_seed(0)
    x = torch.rand(6, 5, 7)
    if not run_model("test_ncnn_batch_layout_multi_consumer", ModelMultiConsumer(), x):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
