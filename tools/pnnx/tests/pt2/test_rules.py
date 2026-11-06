# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x, w, b):
        return F.conv1d(x, w, b, padding=self.padding, groups=2)


def test_conv1d():
    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7)
    w = torch.randn(6, 2, 3)
    b = torch.randn(6)
    expected = []
    for padding in (1, "same", "valid"):
        net = Conv1d(padding).eval()
        expected.append(net(x, w, b))
        ep = torch.export.export(net, (x, w, b))
        path = "test_pt2_rules_conv1_%s.pt2" % padding
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_conv1_1_pnnx
    import test_pt2_rules_conv1_same_pnnx
    import test_pt2_rules_conv1_valid_pnnx
    actual = (
        test_pt2_rules_conv1_1_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv1_same_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv1_valid_pnnx.Model().eval()(x, w, b),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class ConvTranspose1d(nn.Module):
    def forward(self, x, w, b):
        return F.conv_transpose1d(x, w, b, stride=2, padding=1, output_padding=1, groups=2)


def test_conv_transpose1():
    net = ConvTranspose1d().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7)
    w = torch.randn(4, 3, 3)
    b = torch.randn(6)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_conv_transpose1.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv_transpose1.pt2 fp16=0")

    import test_pt2_rules_conv_transpose1_pnnx
    actual = test_pt2_rules_conv_transpose1_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MaxPool1dEmptyStride(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_max_pool1_empty_stride():
    net = MaxPool1dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool1_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool1_empty_stride.pt2 fp16=0")

    import test_pt2_rules_max_pool1_empty_stride_pnnx
    actual = test_pt2_rules_max_pool1_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class AvgPool1dEmptyStride(nn.Module):
    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_avg_pool1_empty_stride():
    net = AvgPool1dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_avg_pool1_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_avg_pool1_empty_stride.pt2 fp16=0")

    import test_pt2_rules_avg_pool1_empty_stride_pnnx
    actual = test_pt2_rules_avg_pool1_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MaxPool1dIndices(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=2, stride=None, return_indices=True)


def test_max_pool1_indices():
    net = MaxPool1dIndices().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool1_indices.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool1_indices.pt2 fp16=0")

    import test_pt2_rules_max_pool1_indices_pnnx
    actual = test_pt2_rules_max_pool1_indices_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
    return (torch.allclose(expected[0], actual[0], rtol=2e-5, atol=2e-5)
            and torch.equal(expected[1], actual[1]))


class Conv2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x, w, b):
        return F.conv2d(x, w, b, padding=self.padding, groups=2)


def test_conv2d():
    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7)
    w = torch.randn(6, 2, 3, 3)
    b = torch.randn(6)
    expected = []
    for padding in (1, "same", "valid"):
        net = Conv2d(padding).eval()
        expected.append(net(x, w, b))
        ep = torch.export.export(net, (x, w, b))
        path = "test_pt2_rules_conv2_%s.pt2" % padding
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_conv2_1_pnnx
    import test_pt2_rules_conv2_same_pnnx
    import test_pt2_rules_conv2_valid_pnnx
    actual = (
        test_pt2_rules_conv2_1_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv2_same_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv2_valid_pnnx.Model().eval()(x, w, b),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class ConvTranspose2d(nn.Module):
    def forward(self, x, w, b):
        return F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1, groups=2)


def test_conv_transpose2():
    net = ConvTranspose2d().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(6)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_conv_transpose2.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv_transpose2.pt2 fp16=0")

    import test_pt2_rules_conv_transpose2_pnnx
    actual = test_pt2_rules_conv_transpose2_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MaxPool2dEmptyStride(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_max_pool2_empty_stride():
    net = MaxPool2dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool2_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool2_empty_stride.pt2 fp16=0")

    import test_pt2_rules_max_pool2_empty_stride_pnnx
    actual = test_pt2_rules_max_pool2_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class AvgPool2dEmptyStride(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_avg_pool2_empty_stride():
    net = AvgPool2dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_avg_pool2_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_avg_pool2_empty_stride.pt2 fp16=0")

    import test_pt2_rules_avg_pool2_empty_stride_pnnx
    actual = test_pt2_rules_avg_pool2_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MaxPool2dIndices(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=None, return_indices=True)


def test_max_pool2_indices():
    net = MaxPool2dIndices().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool2_indices.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool2_indices.pt2 fp16=0")

    import test_pt2_rules_max_pool2_indices_pnnx
    actual = test_pt2_rules_max_pool2_indices_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
    return (torch.allclose(expected[0], actual[0], rtol=2e-5, atol=2e-5)
            and torch.equal(expected[1], actual[1]))


class Conv3d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x, w, b):
        return F.conv3d(x, w, b, padding=self.padding, groups=2)


def test_conv3d():
    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7, 7)
    w = torch.randn(6, 2, 3, 3, 3)
    b = torch.randn(6)
    expected = []
    for padding in (1, "same", "valid"):
        net = Conv3d(padding).eval()
        expected.append(net(x, w, b))
        ep = torch.export.export(net, (x, w, b))
        path = "test_pt2_rules_conv3_%s.pt2" % padding
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_conv3_1_pnnx
    import test_pt2_rules_conv3_same_pnnx
    import test_pt2_rules_conv3_valid_pnnx
    actual = (
        test_pt2_rules_conv3_1_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv3_same_pnnx.Model().eval()(x, w, b),
        test_pt2_rules_conv3_valid_pnnx.Model().eval()(x, w, b),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class ConvTranspose3d(nn.Module):
    def forward(self, x, w, b):
        return F.conv_transpose3d(x, w, b, stride=2, padding=1, output_padding=1, groups=2)


def test_conv_transpose3():
    net = ConvTranspose3d().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7, 7)
    w = torch.randn(4, 3, 3, 3, 3)
    b = torch.randn(6)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_conv_transpose3.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv_transpose3.pt2 fp16=0")

    import test_pt2_rules_conv_transpose3_pnnx
    actual = test_pt2_rules_conv_transpose3_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class Conv3dSingleStride(nn.Module):
    def forward(self, x, w):
        return (
            F.conv3d(x, w, stride=(2,)),
            F.conv3d(x, w, stride=(2,), padding=(1,), dilation=(2,)),
        )


def test_conv3d_single_stride():
    net = Conv3dSingleStride().eval()
    x = torch.arange(125, dtype=torch.float32).reshape(1, 1, 5, 5, 5) / 125
    w = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3) / 27
    expected = net(x, w)

    ep = torch.export.export(net, (x, w))
    torch.export.save(ep, "test_pt2_rules_conv3d_single_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv3d_single_stride.pt2 fp16=0")

    import test_pt2_rules_conv3d_single_stride_pnnx
    actual = test_pt2_rules_conv3d_single_stride_pnnx.Model().eval()(x, w)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class ConvTranspose2dSingleStride(nn.Module):
    def forward(self, x, w):
        return (
            F.conv_transpose2d(x, w, stride=(2,)),
            F.conv_transpose2d(x, w, stride=(2,), padding=(1,), output_padding=(1,), dilation=(2,)),
        )


def test_conv_transpose2d_single_stride():
    net = ConvTranspose2dSingleStride().eval()
    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) / 9
    w = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) / 9
    expected = net(x, w)

    ep = torch.export.export(net, (x, w))
    torch.export.save(ep, "test_pt2_rules_conv_transpose2d_single_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv_transpose2d_single_stride.pt2 fp16=0")

    import test_pt2_rules_conv_transpose2d_single_stride_pnnx
    actual = test_pt2_rules_conv_transpose2d_single_stride_pnnx.Model().eval()(x, w)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class ConvTranspose3dSingleStride(nn.Module):
    def forward(self, x, w):
        return (
            F.conv_transpose3d(x, w, stride=(2,)),
            F.conv_transpose3d(x, w, stride=(2,), padding=(1,), output_padding=(1,), dilation=(2,)),
        )


def test_conv_transpose3d_single_stride():
    net = ConvTranspose3dSingleStride().eval()
    x = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3) / 27
    w = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3) / 27
    expected = net(x, w)

    ep = torch.export.export(net, (x, w))
    torch.export.save(ep, "test_pt2_rules_conv_transpose3d_single_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_conv_transpose3d_single_stride.pt2 fp16=0")

    import test_pt2_rules_conv_transpose3d_single_stride_pnnx
    actual = test_pt2_rules_conv_transpose3d_single_stride_pnnx.Model().eval()(x, w)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class MaxPool3dEmptyStride(nn.Module):
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_max_pool3_empty_stride():
    net = MaxPool3dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool3_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool3_empty_stride.pt2 fp16=0")

    import test_pt2_rules_max_pool3_empty_stride_pnnx
    actual = test_pt2_rules_max_pool3_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class AvgPool3dEmptyStride(nn.Module):
    def forward(self, x):
        return F.avg_pool3d(x, kernel_size=2, stride=None, ceil_mode=True)


def test_avg_pool3_empty_stride():
    net = AvgPool3dEmptyStride().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_avg_pool3_empty_stride.pt2")
    os.system("../../src/pnnx test_pt2_rules_avg_pool3_empty_stride.pt2 fp16=0")

    import test_pt2_rules_avg_pool3_empty_stride_pnnx
    actual = test_pt2_rules_avg_pool3_empty_stride_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MaxPool3dIndices(nn.Module):
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=2, stride=None, return_indices=True)


def test_max_pool3_indices():
    net = MaxPool3dIndices().eval()

    torch.manual_seed(7767517)
    x = torch.randn(1, 4, 7, 7, 7)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_max_pool3_indices.pt2")
    os.system("../../src/pnnx test_pt2_rules_max_pool3_indices.pt2 fp16=0")

    import test_pt2_rules_max_pool3_indices_pnnx
    actual = test_pt2_rules_max_pool3_indices_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
    return (torch.allclose(expected[0], actual[0], rtol=2e-5, atol=2e-5)
            and torch.equal(expected[1], actual[1]))


class Clone(nn.Module):
    def __init__(self, memory_format):
        super().__init__()
        self.memory_format = memory_format

    def forward(self, x):
        return torch.clone(x, memory_format=self.memory_format)


def test_clone():
    torch.manual_seed(7767517)
    x = torch.randn(2, 3, 4, 5).transpose(-1, -2)
    y = torch.randn(2, 3, 4, 5, 6).transpose(-1, -2)
    expected = []
    for name, memory_format, sample in (
        ("default", None, x),
        ("contiguous", torch.contiguous_format, x),
        ("preserve", torch.preserve_format, x),
        ("channels_last", torch.channels_last, x),
        ("channels_last_3d", torch.channels_last_3d, y),
    ):
        net = Clone(memory_format).eval()
        expected.append(net(sample))
        ep = torch.export.export(net, (sample,))
        path = "test_pt2_rules_clone_" + name + ".pt2"
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_clone_default_pnnx
    import test_pt2_rules_clone_contiguous_pnnx
    import test_pt2_rules_clone_preserve_pnnx
    import test_pt2_rules_clone_channels_last_pnnx
    import test_pt2_rules_clone_channels_last_3d_pnnx
    actual = (
        test_pt2_rules_clone_default_pnnx.Model().eval()(x),
        test_pt2_rules_clone_contiguous_pnnx.Model().eval()(x),
        test_pt2_rules_clone_preserve_pnnx.Model().eval()(x),
        test_pt2_rules_clone_channels_last_pnnx.Model().eval()(x),
        test_pt2_rules_clone_channels_last_3d_pnnx.Model().eval()(y),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
            return False
        if not torch.allclose(a, b, rtol=1e-5, atol=1e-5):
            return False
    return True


class ToFloat16(nn.Module):
    def __init__(self, memory_format):
        super().__init__()
        self.memory_format = memory_format

    def forward(self, x):
        return x.to(dtype=torch.float16, copy=True, memory_format=self.memory_format)


def test_to():
    torch.manual_seed(7767517)
    x = torch.randn(2, 3, 4, 5).transpose(-1, -2)
    y = torch.randn(2, 3, 4, 5, 6).transpose(-1, -2)
    expected = []
    for name, memory_format, sample in (
        ("default", None, x),
        ("contiguous", torch.contiguous_format, x),
        ("preserve", torch.preserve_format, x),
        ("channels_last", torch.channels_last, x),
        ("channels_last_3d", torch.channels_last_3d, y),
    ):
        net = ToFloat16(memory_format).eval()
        expected.append(net(sample))
        ep = torch.export.export(net, (sample,))
        path = "test_pt2_rules_to_" + name + ".pt2"
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_to_default_pnnx
    import test_pt2_rules_to_contiguous_pnnx
    import test_pt2_rules_to_preserve_pnnx
    import test_pt2_rules_to_channels_last_pnnx
    import test_pt2_rules_to_channels_last_3d_pnnx
    actual = (
        test_pt2_rules_to_default_pnnx.Model().eval()(x),
        test_pt2_rules_to_contiguous_pnnx.Model().eval()(x),
        test_pt2_rules_to_preserve_pnnx.Model().eval()(x),
        test_pt2_rules_to_channels_last_pnnx.Model().eval()(x),
        test_pt2_rules_to_channels_last_3d_pnnx.Model().eval()(y),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
            return False
        if not torch.allclose(a, b, rtol=2e-3, atol=2e-3):
            return False
    return True


class SoftmaxDtype(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1, dtype=torch.float32)


def test_softmax_dtype():
    net = SoftmaxDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_softmax_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_softmax_dtype.pt2 fp16=0")

    import test_pt2_rules_softmax_dtype_pnnx
    actual = test_pt2_rules_softmax_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class LogSoftmaxDtype(nn.Module):
    def forward(self, x):
        return F.log_softmax(x, dim=-1, dtype=torch.float32)


def test_log_softmax_dtype():
    net = LogSoftmaxDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_log_softmax_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_log_softmax_dtype.pt2 fp16=0")

    import test_pt2_rules_log_softmax_dtype_pnnx
    actual = test_pt2_rules_log_softmax_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class SumDtype(nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=(1,), keepdim=True, dtype=torch.float32)


def test_sum_dtype():
    net = SumDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_sum_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_sum_dtype.pt2 fp16=0")

    import test_pt2_rules_sum_dtype_pnnx
    actual = test_pt2_rules_sum_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MeanDtype(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(1,), keepdim=True, dtype=torch.float32)


def test_mean_dtype():
    net = MeanDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_mean_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_mean_dtype.pt2 fp16=0")

    import test_pt2_rules_mean_dtype_pnnx
    actual = test_pt2_rules_mean_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class ProdDtype(nn.Module):
    def forward(self, x):
        return torch.prod(x, dim=1, keepdim=True, dtype=torch.float32)


def test_prod_dtype():
    net = ProdDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_prod_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_prod_dtype.pt2 fp16=0")

    import test_pt2_rules_prod_dtype_pnnx
    actual = test_pt2_rules_prod_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class SumAllDtype(nn.Module):
    def forward(self, x):
        return torch.sum(x, dtype=torch.float32)


def test_sum_all_dtype():
    net = SumAllDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_sum_all_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_sum_all_dtype.pt2 fp16=0")

    import test_pt2_rules_sum_all_dtype_pnnx
    actual = test_pt2_rules_sum_all_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class MeanAllDtype(nn.Module):
    def forward(self, x):
        return torch.mean(x, dtype=torch.float32)


def test_mean_all_dtype():
    net = MeanAllDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_mean_all_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_mean_all_dtype.pt2 fp16=0")

    import test_pt2_rules_mean_all_dtype_pnnx
    actual = test_pt2_rules_mean_all_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class ProdAllDtype(nn.Module):
    def forward(self, x):
        return torch.prod(x, dtype=torch.float32)


def test_prod_all_dtype():
    net = ProdAllDtype().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, dtype=torch.float16)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_prod_all_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_prod_all_dtype.pt2 fp16=0")

    import test_pt2_rules_prod_all_dtype_pnnx
    actual = test_pt2_rules_prod_all_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class SumIntegerDtype(nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1, dtype=torch.int64)


def test_sum_integer_dtype():
    net = SumIntegerDtype().eval()

    torch.manual_seed(7767517)
    x = torch.tensor([[2147483647, 1], [2, 3]], dtype=torch.int32)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_sum_integer_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_sum_integer_dtype.pt2 fp16=0")

    import test_pt2_rules_sum_integer_dtype_pnnx
    actual = test_pt2_rules_sum_integer_dtype_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.equal(expected, actual)


class RmsNormDefault(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (4,))


def test_rms_norm_default():
    net = RmsNormDefault().eval()
    x_fp16 = torch.full((2, 4), 1e-3, dtype=torch.float16)
    x_fp32 = torch.full((2, 4), 1e-3, dtype=torch.float32)
    x_bf16 = torch.full((2, 4), 1e-3, dtype=torch.bfloat16)
    expected = []
    for name, x in (("float16", x_fp16), ("float32", x_fp32), ("bfloat16", x_bf16)):
        expected.append(net(x))
        ep = torch.export.export(net, (x,))
        path = "test_pt2_rules_rms_norm_default_" + name + ".pt2"
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_rms_norm_default_float16_pnnx
    import test_pt2_rules_rms_norm_default_float32_pnnx
    import test_pt2_rules_rms_norm_default_bfloat16_pnnx
    actual = (
        test_pt2_rules_rms_norm_default_float16_pnnx.Model().eval()(x_fp16),
        test_pt2_rules_rms_norm_default_float32_pnnx.Model().eval()(x_fp32),
        test_pt2_rules_rms_norm_default_bfloat16_pnnx.Model().eval()(x_bf16),
    )
    for a, b, tolerance in zip(expected, actual, (3e-3, 2e-5, 2e-2)):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=tolerance, atol=tolerance):
            return False
    return True


class NewFullDefault(nn.Module):
    def forward(self, x):
        return x, x.new_full((2, 4), 0.25)


def test_new_full_default():
    net = NewFullDefault().eval()
    x_fp16 = torch.full((2, 4), 1e-3, dtype=torch.float16)
    x_fp32 = torch.full((2, 4), 1e-3, dtype=torch.float32)
    x_bf16 = torch.full((2, 4), 1e-3, dtype=torch.bfloat16)
    expected = []
    for name, x in (("float16", x_fp16), ("float32", x_fp32), ("bfloat16", x_bf16)):
        expected.append(net(x))
        ep = torch.export.export(net, (x,))
        path = "test_pt2_rules_new_full_default_" + name + ".pt2"
        torch.export.save(ep, path)
        os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_new_full_default_float16_pnnx
    import test_pt2_rules_new_full_default_float32_pnnx
    import test_pt2_rules_new_full_default_bfloat16_pnnx
    actual = (
        test_pt2_rules_new_full_default_float16_pnnx.Model().eval()(x_fp16),
        test_pt2_rules_new_full_default_float32_pnnx.Model().eval()(x_fp32),
        test_pt2_rules_new_full_default_bfloat16_pnnx.Model().eval()(x_bf16),
    )
    for a, b, tolerance in zip(expected, actual, (3e-3, 2e-5, 2e-2)):
        if len(a) != len(b):
            return False
        for a0, b0 in zip(a, b):
            if a0.shape != b0.shape or a0.dtype != b0.dtype:
                return False
            if not torch.allclose(a0, b0, rtol=tolerance, atol=tolerance):
                return False
    return True


class NewFullExplicitDtype(nn.Module):
    def forward(self, x):
        return x, x.new_full((2, 4), 3, dtype=torch.int64)


def test_new_full_explicit_dtype():
    net = NewFullExplicitDtype().eval()

    torch.manual_seed(0)
    x = torch.rand(2, 4)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_new_full_explicit_dtype.pt2")
    os.system("../../src/pnnx test_pt2_rules_new_full_explicit_dtype.pt2 fp16=0")

    import test_pt2_rules_new_full_explicit_dtype_pnnx
    actual = test_pt2_rules_new_full_explicit_dtype_pnnx.test_inference()

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
    return (torch.allclose(expected[0], actual[0], rtol=2e-5, atol=2e-5)
            and torch.equal(expected[1], actual[1]))


class InstanceNormRunning(nn.Module):
    def forward(self, x, mean, var, w, b):
        return F.instance_norm(x, mean, var, w, b, use_input_stats=False)


def test_instance_norm_running():
    net = InstanceNormRunning().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.randn(4)
    b = torch.randn(4)
    mean = torch.randn(4)
    var = torch.rand(4) + 1
    expected = net(x, mean, var, w, b)

    ep = torch.export.export(net, (x, mean, var, w, b))
    torch.export.save(ep, "test_pt2_rules_instance_norm_running.pt2")
    os.system("../../src/pnnx test_pt2_rules_instance_norm_running.pt2 fp16=0")

    import test_pt2_rules_instance_norm_running_pnnx
    actual = test_pt2_rules_instance_norm_running_pnnx.Model().eval()(x, mean, var, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class NativeGroupNorm(nn.Module):
    def forward(self, x, w, b):
        return torch.ops.aten.native_group_norm.default(x, w, b, 2, 4, 3, 2, 1e-5)[0]


def test_native_group_norm():
    net = NativeGroupNorm().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.randn(4)
    b = torch.randn(4)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_native_group_norm.pt2")
    os.system("../../src/pnnx test_pt2_rules_native_group_norm.pt2 fp16=0")

    import test_pt2_rules_native_group_norm_pnnx
    actual = test_pt2_rules_native_group_norm_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class NativeGroupNormSingleton(nn.Module):
    def forward(self, x):
        return torch.native_group_norm(x, None, None, 1, 4, 1, 4, 1e-5)[0]


def test_native_group_norm_singleton():
    net = NativeGroupNormSingleton().eval()
    x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_native_group_norm_singleton.pt2")
    os.system("../../src/pnnx test_pt2_rules_native_group_norm_singleton.pt2 fp16=0")

    import test_pt2_rules_native_group_norm_singleton_pnnx
    actual = test_pt2_rules_native_group_norm_singleton_pnnx.Model().eval()(x)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.equal(expected, actual)


class NativeGroupNormSingletonAffine(nn.Module):
    def forward(self, x, w, b):
        return torch.native_group_norm(x, w, b, 1, 4, 1, 4, 1e-5)[0]


def test_native_group_norm_singleton_affine():
    net = NativeGroupNormSingletonAffine().eval()
    x = torch.tensor([-3, -0.5, 2, 7], dtype=torch.float32).reshape(1, 4, 1)
    w = torch.tensor([0.5, 2, -1, 3], dtype=torch.float32)
    b = torch.tensor([1, -2, 3, -4], dtype=torch.float32)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_native_group_norm_singleton_affine.pt2")
    os.system("../../src/pnnx test_pt2_rules_native_group_norm_singleton_affine.pt2 fp16=0")

    import test_pt2_rules_native_group_norm_singleton_affine_pnnx
    actual = test_pt2_rules_native_group_norm_singleton_affine_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class NativeLayerNorm(nn.Module):
    def forward(self, x, w, b):
        return torch.ops.aten.native_layer_norm.default(x, (3,), w, b, 1e-5)[0]


def test_native_layer_norm():
    net = NativeLayerNorm().eval()

    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.randn(3)
    b = torch.randn(3)
    expected = net(x, w, b)

    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_native_layer_norm.pt2")
    os.system("../../src/pnnx test_pt2_rules_native_layer_norm.pt2 fp16=0")

    import test_pt2_rules_native_layer_norm_pnnx
    actual = test_pt2_rules_native_layer_norm_pnnx.Model().eval()(x, w, b)

    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    return torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


class WeightNorm(nn.Module):
    def __init__(self, dim, value):
        super().__init__()
        self.dim = dim
        self.register_buffer("v", torch.full((2, 3, 4), value, dtype=torch.float32))
        shape = [1, 1, 1]
        if dim != -1:
            shape[dim] = self.v.shape[dim]
        self.register_buffer("g", torch.ones(shape, dtype=torch.float32))

    def forward(self, x):
        return x + torch._weight_norm(self.v, self.g, self.dim)


def test_weight_norm():
    x = torch.zeros(2, 3, 4)
    expected = []
    # FP32 squared norms overflow/underflow for these finite constants.
    for dim in (-1, 0, 1, 2):
        axis = "all" if dim == -1 else str(dim)
        for name, value in (("large", 1e20), ("small", 1e-30)):
            net = WeightNorm(dim, value).eval()
            expected.append(net(x))
            ep = torch.export.export(net, (x,))
            path = "test_pt2_rules_weight_norm_%s_dim_%s.pt2" % (name, axis)
            torch.export.save(ep, path)
            os.system("../../src/pnnx " + path + " fp16=0")

    import test_pt2_rules_weight_norm_large_dim_all_pnnx
    import test_pt2_rules_weight_norm_small_dim_all_pnnx
    import test_pt2_rules_weight_norm_large_dim_0_pnnx
    import test_pt2_rules_weight_norm_small_dim_0_pnnx
    import test_pt2_rules_weight_norm_large_dim_1_pnnx
    import test_pt2_rules_weight_norm_small_dim_1_pnnx
    import test_pt2_rules_weight_norm_large_dim_2_pnnx
    import test_pt2_rules_weight_norm_small_dim_2_pnnx
    actual = (
        test_pt2_rules_weight_norm_large_dim_all_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_small_dim_all_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_large_dim_0_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_small_dim_0_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_large_dim_1_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_small_dim_1_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_large_dim_2_pnnx.Model().eval()(x),
        test_pt2_rules_weight_norm_small_dim_2_pnnx.Model().eval()(x),
    )
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class HannWindowPeriodic(nn.Module):
    def forward(self, x):
        return (
            x + torch.hann_window(8),
            x + torch.hann_window(8, periodic=True),
            x + torch.hann_window(8, periodic=False),
            x + torch.hann_window(8, periodic=False, dtype=torch.float32),
        )


def test_hann_window_periodic():
    net = HannWindowPeriodic().eval()
    x = torch.zeros(8, dtype=torch.float32)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_hann_window_periodic.pt2")
    os.system("../../src/pnnx test_pt2_rules_hann_window_periodic.pt2 fp16=0")

    import test_pt2_rules_hann_window_periodic_pnnx
    actual = test_pt2_rules_hann_window_periodic_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class HammingWindowPeriodic(nn.Module):
    def forward(self, x):
        return (
            x + torch.hamming_window(8),
            x + torch.hamming_window(8, periodic=True),
            x + torch.hamming_window(8, periodic=False),
            x + torch.hamming_window(8, periodic=False, dtype=torch.float32),
        )


def test_hamming_window_periodic():
    net = HammingWindowPeriodic().eval()
    x = torch.zeros(8, dtype=torch.float32)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_hamming_window_periodic.pt2")
    os.system("../../src/pnnx test_pt2_rules_hamming_window_periodic.pt2 fp16=0")

    import test_pt2_rules_hamming_window_periodic_pnnx
    actual = test_pt2_rules_hamming_window_periodic_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class HammingWindowAlpha(nn.Module):
    def forward(self, x):
        return (
            x + torch.hamming_window(8, periodic=True, alpha=0.6),
            x + torch.hamming_window(8, periodic=False, alpha=0.6, dtype=torch.float32),
        )


def test_hamming_window_alpha():
    net = HammingWindowAlpha().eval()
    x = torch.zeros(8, dtype=torch.float32)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_hamming_window_alpha.pt2")
    os.system("../../src/pnnx test_pt2_rules_hamming_window_alpha.pt2 fp16=0")

    import test_pt2_rules_hamming_window_alpha_pnnx
    actual = test_pt2_rules_hamming_window_alpha_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class HammingWindowAlphaBeta(nn.Module):
    def forward(self, x):
        return (
            x + torch.hamming_window(8, periodic=True, alpha=0.6, beta=0.3),
            x + torch.hamming_window(8, periodic=False, alpha=0.6, beta=0.3, dtype=torch.float32),
        )


def test_hamming_window_alpha_beta():
    net = HammingWindowAlphaBeta().eval()
    x = torch.zeros(8, dtype=torch.float32)
    expected = net(x)

    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_hamming_window_alpha_beta.pt2")
    os.system("../../src/pnnx test_pt2_rules_hamming_window_alpha_beta.pt2 fp16=0")

    import test_pt2_rules_hamming_window_alpha_beta_pnnx
    actual = test_pt2_rules_hamming_window_alpha_beta_pnnx.Model().eval()(x)

    if len(expected) != len(actual):
        return False
    for a, b in zip(expected, actual):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.allclose(a, b, rtol=2e-5, atol=2e-5):
            return False
    return True


class NativeLayerNormStats(nn.Module):
    def forward(self, x, w, b):
        return torch.ops.aten.native_layer_norm.default(x, (3,), w, b, 1e-5)


def test_native_layer_norm_stats():
    net = NativeLayerNormStats().eval()
    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.ones(3)
    b = torch.zeros(3)
    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_native_layer_norm_stats.pt2")

    result = subprocess.run(
        ["../../src/pnnx", "test_pt2_rules_native_layer_norm_stats.pt2", "fp16=0"],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode == 0 or "pt2 pnnx: unsupported operator aten::native_layer_norm" not in result.stderr:
        print("native_layer_norm_stats", result.stderr)
        return False
    return True


class NativeGroupNormStats(nn.Module):
    def forward(self, x, w, b):
        return torch.ops.aten.native_group_norm.default(x, w, b, 2, 4, 3, 2, 1e-5)


def test_native_group_norm_stats():
    net = NativeGroupNormStats().eval()
    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.ones(4)
    b = torch.zeros(4)
    ep = torch.export.export(net, (x, w, b))
    torch.export.save(ep, "test_pt2_rules_native_group_norm_stats.pt2")

    result = subprocess.run(
        ["../../src/pnnx", "test_pt2_rules_native_group_norm_stats.pt2", "fp16=0"],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode == 0 or "pt2 pnnx: unsupported operator aten::native_group_norm" not in result.stderr:
        print("native_group_norm_stats", result.stderr)
        return False
    return True


class NativeGroupNormLayout(nn.Module):
    def forward(self, x):
        return torch.ops.aten.native_group_norm.default(x, None, None, 1, 4, 6, 2, 1e-5)[0]


def test_native_group_norm_layout():
    net = NativeGroupNormLayout().eval()
    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pt2_rules_native_group_norm_layout.pt2")

    result = subprocess.run(
        ["../../src/pnnx", "test_pt2_rules_native_group_norm_layout.pt2", "fp16=0"],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode == 0 or "pt2 pnnx: unsupported operator aten::native_group_norm" not in result.stderr:
        print("native_group_norm_layout", result.stderr)
        return False
    return True


class NativeBatchNormStats(nn.Module):
    def forward(self, x, w, b, mean, var):
        return torch.ops.aten._native_batch_norm_legit_no_training.default(x, w, b, mean, var, 0.1, 1e-5)


def test_native_batch_norm_stats():
    net = NativeBatchNormStats().eval()
    torch.manual_seed(7767517)
    x = torch.randn(2, 4, 3)
    w = torch.ones(4)
    b = torch.zeros(4)
    mean = torch.zeros(4)
    var = torch.ones(4)
    ep = torch.export.export(net, (x, w, b, mean, var))
    torch.export.save(ep, "test_pt2_rules_native_batch_norm_stats.pt2")

    result = subprocess.run(
        ["../../src/pnnx", "test_pt2_rules_native_batch_norm_stats.pt2", "fp16=0"],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode == 0 or "pt2 pnnx: unsupported operator aten::_native_batch_norm_legit_no_training" not in result.stderr:
        print("native_batch_norm_stats", result.stderr)
        return False
    return True


def test():
    torch.set_num_threads(1)
    if not test_conv1d():
        print("conv1d failed")
        return False
    if not test_conv_transpose1():
        print("conv_transpose1 failed")
        return False
    if not test_max_pool1_empty_stride():
        print("max_pool1_empty_stride failed")
        return False
    if not test_avg_pool1_empty_stride():
        print("avg_pool1_empty_stride failed")
        return False
    if not test_max_pool1_indices():
        print("max_pool1_indices failed")
        return False
    if not test_conv2d():
        print("conv2d failed")
        return False
    if not test_conv_transpose2():
        print("conv_transpose2 failed")
        return False
    if not test_max_pool2_empty_stride():
        print("max_pool2_empty_stride failed")
        return False
    if not test_avg_pool2_empty_stride():
        print("avg_pool2_empty_stride failed")
        return False
    if not test_max_pool2_indices():
        print("max_pool2_indices failed")
        return False
    if not test_conv3d():
        print("conv3d failed")
        return False
    if not test_conv_transpose3():
        print("conv_transpose3 failed")
        return False
    if not test_conv3d_single_stride():
        print("conv3d_single_stride failed")
        return False
    if not test_conv_transpose2d_single_stride():
        print("conv_transpose2d_single_stride failed")
        return False
    if not test_conv_transpose3d_single_stride():
        print("conv_transpose3d_single_stride failed")
        return False
    if not test_max_pool3_empty_stride():
        print("max_pool3_empty_stride failed")
        return False
    if not test_avg_pool3_empty_stride():
        print("avg_pool3_empty_stride failed")
        return False
    if not test_max_pool3_indices():
        print("max_pool3_indices failed")
        return False
    if not test_clone():
        print("clone failed")
        return False
    if not test_to():
        print("to failed")
        return False
    if not test_softmax_dtype():
        print("softmax_dtype failed")
        return False
    if not test_log_softmax_dtype():
        print("log_softmax_dtype failed")
        return False
    if not test_sum_dtype():
        print("sum_dtype failed")
        return False
    if not test_mean_dtype():
        print("mean_dtype failed")
        return False
    if not test_prod_dtype():
        print("prod_dtype failed")
        return False
    if not test_sum_all_dtype():
        print("sum_all_dtype failed")
        return False
    if not test_mean_all_dtype():
        print("mean_all_dtype failed")
        return False
    if not test_prod_all_dtype():
        print("prod_all_dtype failed")
        return False
    if not test_sum_integer_dtype():
        print("sum_integer_dtype failed")
        return False
    if not test_rms_norm_default():
        print("rms_norm_default failed")
        return False
    if not test_new_full_default():
        print("new_full_default failed")
        return False
    if not test_new_full_explicit_dtype():
        print("new_full_explicit_dtype failed")
        return False
    if not test_instance_norm_running():
        print("instance_norm_running failed")
        return False
    if not test_native_group_norm():
        print("native_group_norm failed")
        return False
    if not test_native_group_norm_singleton():
        print("native_group_norm_singleton failed")
        return False
    if not test_native_group_norm_singleton_affine():
        print("native_group_norm_singleton_affine failed")
        return False
    if not test_native_layer_norm():
        print("native_layer_norm failed")
        return False
    if not test_weight_norm():
        print("weight_norm failed")
        return False
    if not test_hann_window_periodic():
        print("hann_window_periodic failed")
        return False
    if not test_hamming_window_periodic():
        print("hamming_window_periodic failed")
        return False
    if not test_hamming_window_alpha():
        print("hamming_window_alpha failed")
        return False
    if not test_hamming_window_alpha_beta():
        print("hamming_window_alpha_beta failed")
        return False
    if not test_native_layer_norm_stats():
        print("native_layer_norm_stats failed")
        return False
    if not test_native_group_norm_stats():
        print("native_group_norm_stats failed")
        return False
    if not test_native_group_norm_layout():
        print("native_group_norm_layout failed")
        return False
    if not test_native_batch_norm_stats():
        print("native_batch_norm_stats failed")
        return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
