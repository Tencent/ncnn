# Copyright 2025 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = x.permute(2, 3, 0, 1)
        x = x.reshape(-1, x.size(0) // 8, 1, 3, 5)
        x = x.transpose(0, 1)
        x = x.reshape(-1, 1, 15)
        x = x.unsqueeze(-1)
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(1, -1)
        x = x.unsqueeze(0)
        x = x.relu()
        return x


class ModelBatch2(nn.Module):
    def __init__(self):
        super(ModelBatch2, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = x.permute(2, 3, 0, 1)
        x = x.reshape(-1, x.size(0) // 8, x.size(2), 3, 5)
        x = x.transpose(0, 1)
        x = x.reshape(-1, x.size(2), 15)
        x = x.unsqueeze(-1)
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(x.size(0), 1, -1)
        x = x.relu()
        return x


class ModelFlattenDynamicExpr(nn.Module):
    def __init__(self):
        super(ModelFlattenDynamicExpr, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = torch.flatten(x, 1, 2)
        x = x.relu()
        return x


class ModelReshapeDynamicExprCompat(nn.Module):
    def __init__(self):
        super(ModelReshapeDynamicExprCompat, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 1)
        x = x.reshape(x.size(0), x.size(2), x.size(3), x.size(1))
        x = x.relu()
        return x


def no_batch_reshape_param(name):
    with open(name + ".ncnn.param") as f:
        for line in f:
            if line.startswith("Reshape ") and (" 12=" in line or " 13=" in line):
                return False

    return True


def run_model(name, net, x0, x1):
    net.eval()

    a0 = net(x0)
    a1 = net(x1)

    # export torchscript
    mod = torch.jit.trace(net, x0)
    mod.save(name + ".pt")

    # torchscript to pnnx
    import os
    inputshape = str(list(x0.shape)).replace(" ", "")
    inputshape2 = str(list(x1.shape)).replace(" ", "")
    if os.system("../../src/pnnx " + name + ".pt inputshape=" + inputshape + " inputshape2=" + inputshape2) != 0:
        return False

    # ncnn inference
    import ncnn
    with ncnn.Net() as net:
        net.load_param(name + ".ncnn.param")
        net.load_model(name + ".ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x0.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            b0 = torch.from_numpy(out0.numpy(batch_index=0))

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x1.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            b1 = torch.from_numpy(out0.numpy(batch_index=0))

    if a0.shape != b0.shape:
        print(name)
        print(a0.shape, b0.shape)
        return False
    if a1.shape != b1.shape:
        print(name)
        print(a1.shape, b1.shape)
        return False
    if not torch.allclose(a0, b0, 1e-3, 1e-3):
        print(name)
        print((a0 - b0).abs().max())
        return False
    if not torch.allclose(a1, b1, 1e-3, 1e-3):
        print(name)
        print((a1 - b1).abs().max())
        return False

    return True


def test():
    torch.manual_seed(0)

    net = Model().half().float()

    torch.manual_seed(0)
    x0 = torch.rand(1, 3, 32, 32)

    x1 = torch.rand(1, 3, 64, 64)

    if not run_model("test_ncnn_solve_batch_index", net, x0, x1):
        return False

    torch.manual_seed(0)

    net = ModelBatch2().half().float()

    torch.manual_seed(0)
    x0 = torch.rand(2, 3, 32, 32)

    x1 = torch.rand(2, 3, 64, 64)

    if not run_model("test_ncnn_solve_batch_index_batch2", net, x0, x1):
        return False

    torch.manual_seed(0)

    net = ModelFlattenDynamicExpr()

    torch.manual_seed(0)
    x0 = torch.rand(2, 3, 5, 7)

    x1 = torch.rand(2, 3, 9, 11)

    if not run_model("test_ncnn_solve_batch_index_flatten_dynamic_expr", net, x0, x1):
        return False
    if not no_batch_reshape_param("test_ncnn_solve_batch_index_flatten_dynamic_expr"):
        return False

    torch.manual_seed(0)

    net = ModelReshapeDynamicExprCompat()

    torch.manual_seed(0)
    x0 = torch.rand(2, 3, 5, 7)

    x1 = torch.rand(2, 3, 9, 11)

    if not run_model("test_ncnn_solve_batch_index_reshape_dynamic_expr_compat", net, x0, x1):
        return False
    if not no_batch_reshape_param("test_ncnn_solve_batch_index_reshape_dynamic_expr_compat"):
        return False

    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
