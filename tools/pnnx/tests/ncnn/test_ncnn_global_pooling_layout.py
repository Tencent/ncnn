# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelDirectPool(nn.Module):
    def __init__(self):
        super(ModelDirectPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x)


class ModelDirectLinear(nn.Module):
    def __init__(self):
        super(ModelDirectLinear, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 3)

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


class ModelFlattenLinear(nn.Module):
    def __init__(self):
        super(ModelFlattenLinear, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ModelMaxPoolLinear(nn.Module):
    def __init__(self):
        super(ModelMaxPoolLinear, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ModelPool3dLinear(nn.Module):
    def __init__(self):
        super(ModelPool3dLinear, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ModelDirectPool3d(nn.Module):
    def __init__(self):
        super(ModelDirectPool3d, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        return self.pool(x)


class ModelFLinear(nn.Module):
    def __init__(self):
        super(ModelFLinear, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.rand(6, 4))
        self.bias = nn.Parameter(torch.rand(6))

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.linear(x, self.weight, self.bias)


class ModelSE(nn.Module):
    def __init__(self):
        super(ModelSE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv2d(4, 3, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv0(w)
        w = self.relu(w)
        w = self.conv1(w)
        w = self.sigmoid(w)
        return x * w, w + x


class ModelPermuteFallback(nn.Module):
    def __init__(self):
        super(ModelPermuteFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        return x.permute(0, 2, 3, 1)


class ModelSliceFallback(nn.Module):
    def __init__(self):
        super(ModelSliceFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        return x[:, :, 0, :]


class ModelReshapeFallback(nn.Module):
    def __init__(self):
        super(ModelReshapeFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        return x.reshape(x.size(0), 1, x.size(1), 1)


class ModelMultiFallback(nn.Module):
    def __init__(self):
        super(ModelMultiFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        return x.permute(0, 2, 3, 1), x[:, :, 0, :]


class ModelOutputFallback(nn.Module):
    def __init__(self):
        super(ModelOutputFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        return x, x.permute(0, 2, 3, 1)


class ModelMultiConvFallback(nn.Module):
    def __init__(self):
        super(ModelMultiConvFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        return x.permute(0, 2, 3, 1), self.conv(x)


class ModelDepthWiseFallback(nn.Module):
    def __init__(self):
        super(ModelDepthWiseFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(4, 4, 1, groups=4)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class ModelNon1x1ConvFallback(nn.Module):
    def __init__(self):
        super(ModelNon1x1ConvFallback, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


def compare(a, b):
    if isinstance(a, tuple):
        if not isinstance(b, tuple):
            return False
        if len(a) != len(b):
            return False

        for a0, b0 in zip(a, b):
            if not torch.allclose(a0, b0, 1e-3, 1e-3):
                return False
        return True

    return torch.allclose(a, b, 1e-3, 1e-3)


def run_model(name, net, x, inputshape):
    net.eval()

    a = net(x)

    mod = torch.jit.trace(net, x)
    mod.save(name + ".pt")

    if os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape) != 0:
        return False

    ncnn = __import__(name + "_ncnn")
    b = ncnn.test_inference()

    return compare(a, b)


def run_model_dynamic(name, net, x, inputshape, inputshape2):
    net.eval()

    a = net(x)

    mod = torch.jit.trace(net, x)
    mod.save(name + ".pt")

    if os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape + " inputshape2=" + inputshape2) != 0:
        return False

    ncnn = __import__(name + "_ncnn")
    b = ncnn.test_inference()

    return compare(a, b)


def run_pnnx_without_inputshape(name, net, x):
    net.eval()

    a = net(x)

    mod = torch.jit.trace(net, x)
    mod.save(name + ".pt")

    if os.system("../../src/pnnx " + name + ".pt") != 0:
        return False

    with ncnn.Net() as ncnn_net:
        ncnn_net.load_param(name + ".ncnn.param")
        ncnn_net.load_model(name + ".ncnn.bin")

        with ncnn_net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.numpy(), batch_index=0).clone())

            if isinstance(a, tuple):
                out = []
                for i in range(len(a)):
                    _, out0 = ex.extract("out%d" % i)
                    out.append(torch.from_numpy(out0.numpy(batch_index=0)))
                b = tuple(out)
            else:
                _, out0 = ex.extract("out0")
                b = torch.from_numpy(out0.numpy(batch_index=0))

    return compare(a, b)


def test():
    torch.manual_seed(0)
    x = torch.rand(2, 4, 5, 7)

    torch.manual_seed(0)
    y = torch.rand(4, 5, 7)

    torch.manual_seed(0)
    z = torch.rand(2, 4, 3, 5, 7)

    if not run_model("test_ncnn_global_pooling_layout_direct", ModelDirectPool(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_direct_linear", ModelDirectLinear(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_direct_unbatched", ModelDirectPool(), y, "[4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_flatten_linear", ModelFlattenLinear(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_maxpool_linear", ModelMaxPoolLinear(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_3d_linear", ModelPool3dLinear(), z, "[2,4,3,5,7]"):
        return False
    if not run_model_dynamic("test_ncnn_global_pooling_layout_3d_direct_dynamic", ModelDirectPool3d(), z, "[2,4,3,5,7]", "[3,6,4,6,8]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_f_linear", ModelFLinear(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_se", ModelSE(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_permute", ModelPermuteFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_slice", ModelSliceFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_reshape", ModelReshapeFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_multi_fallback", ModelMultiFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_multi_conv_fallback", ModelMultiConvFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_output_fallback", ModelOutputFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_depthwise", ModelDepthWiseFallback(), x, "[2,4,5,7]"):
        return False
    if not run_model("test_ncnn_global_pooling_layout_non1x1_conv", ModelNon1x1ConvFallback(), x, "[2,4,5,7]"):
        return False
    if not run_pnnx_without_inputshape("test_ncnn_global_pooling_layout_no_inputshape", ModelSE(), x):
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
