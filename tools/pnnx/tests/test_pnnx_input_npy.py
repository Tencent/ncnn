# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch
import torch.nn as nn
from packaging import version


def _allclose(a, b):
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, x, y):
        return x + y, x * 0.25 - y


def _test_basic():
    net = BasicModel()
    net.eval()

    x_np = (np.arange(12, dtype=np.float32).reshape(1, 3, 4) - 5.0) / 7.0
    y_np = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)

    x_path = "test_pnnx_input_npy_basic_x.npy"
    y_path = "test_pnnx_input_npy_basic_y.npy"
    np.save(x_path, np.array(x_np, dtype=np.float32, order="F"))
    np.save(y_path, y_np)

    x = torch.from_numpy(np.ascontiguousarray(np.load(x_path)))
    y = torch.from_numpy(np.ascontiguousarray(np.load(y_path)))

    a = net(x, y)

    mod = torch.jit.trace(net, (x, y))
    mod.save("test_pnnx_input_npy_basic.pt")

    ret = os.system("../src/pnnx test_pnnx_input_npy_basic.pt input=test_pnnx_input_npy_basic_x.npy,test_pnnx_input_npy_basic_y.npy")
    if ret != 0:
        return False

    import test_pnnx_input_npy_basic_pnnx
    pnnx_net = test_pnnx_input_npy_basic_pnnx.Model()
    pnnx_net.eval()
    b = pnnx_net(x, y)

    if not (_allclose(a, b)):
        return False
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y,))
    torch.export.save(ep, "test_pnnx_input_npy_basic_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_pnnx_input_npy_basic_export.pt2 fp16=0 input=test_pnnx_input_npy_basic_x.npy,test_pnnx_input_npy_basic_y.npy")

    # pnnx inference
    import test_pnnx_input_npy_basic_export_pnnx
    c = test_pnnx_input_npy_basic_export_pnnx.Model().eval()(x, y)
    return len(a) == len(c) and _allclose(a, c)


class Input2Model(nn.Module):
    def __init__(self):
        super(Input2Model, self).__init__()

    def forward(self, x, y):
        z = x + y
        return z.reshape(z.size(0), -1), z[:, :, : z.size(2) // 2] * 2


def _test_input2():
    net = Input2Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(1, 2, 6)
    y0 = torch.rand(1, 2, 6)
    x1 = torch.rand(1, 3, 8)
    y1 = torch.rand(1, 3, 8)

    x0_path = "test_pnnx_input_npy_input2_x0.npy"
    y0_path = "test_pnnx_input_npy_input2_y0.npy"
    x1_path = "test_pnnx_input_npy_input2_x1.npy"
    y1_path = "test_pnnx_input_npy_input2_y1.npy"
    np.save(x0_path, x0.numpy())
    np.save(y0_path, y0.numpy())
    np.save(x1_path, x1.numpy())
    np.save(y1_path, y1.numpy())

    x0 = torch.from_numpy(np.ascontiguousarray(np.load(x0_path)))
    y0 = torch.from_numpy(np.ascontiguousarray(np.load(y0_path)))
    x1 = torch.from_numpy(np.ascontiguousarray(np.load(x1_path)))
    y1 = torch.from_numpy(np.ascontiguousarray(np.load(y1_path)))

    a0 = net(x0, y0)
    a1 = net(x1, y1)

    if version.parse(torch.__version__) < version.parse("2.0"):
        mod = torch.jit.trace(net, (x0, y0))
    else:
        mod = torch.jit.trace(net, (x0, y0), _store_inputs=False)
    mod.save("test_pnnx_input_npy_input2.pt")

    ret = os.system("../src/pnnx test_pnnx_input_npy_input2.pt input=test_pnnx_input_npy_input2_x0.npy,test_pnnx_input_npy_input2_y0.npy input2=test_pnnx_input_npy_input2_x1.npy,test_pnnx_input_npy_input2_y1.npy")
    if ret != 0:
        return False

    import test_pnnx_input_npy_input2_pnnx
    pnnx_net = test_pnnx_input_npy_input2_pnnx.Model()
    pnnx_net.eval()
    b0 = pnnx_net(x0, y0)
    b1 = pnnx_net(x1, y1)

    if not (_allclose(a0, b0) and _allclose(a1, b1)):
        return False
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    channels = torch.export.Dim("channels", min=2)
    width = 2 * torch.export.Dim("half_width", min=2)
    ep = torch.export.export(net, (x0, y0), dynamic_shapes=({1: channels, 2: width}, {1: channels, 2: width}))
    torch.export.save(ep, "test_pnnx_input_npy_input2_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_pnnx_input_npy_input2_export.pt2 fp16=0 input=test_pnnx_input_npy_input2_x0.npy,test_pnnx_input_npy_input2_y0.npy input2=test_pnnx_input_npy_input2_x1.npy,test_pnnx_input_npy_input2_y1.npy")

    # pnnx inference
    import test_pnnx_input_npy_input2_export_pnnx
    pnnx_export = test_pnnx_input_npy_input2_export_pnnx.Model().eval()
    c0 = pnnx_export(x0, y0)
    c1 = pnnx_export(x1, y1)
    return len(a0) == len(c0) and len(a1) == len(c1) and _allclose(a0, c0) and _allclose(a1, c1)


class Int64Model(nn.Module):
    def __init__(self):
        super(Int64Model, self).__init__()

    def forward(self, x, y):
        return torch.gather(x, 1, y)


def _test_int64():
    net = Int64Model()
    net.eval()

    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    y = torch.tensor([[[0, 1, 2, 0], [2, 1, 0, 2]], [[1, 2, 0, 1], [0, 2, 1, 0]]], dtype=torch.long)

    x_path = "test_pnnx_input_npy_int64_x.npy"
    y_path = "test_pnnx_input_npy_int64_y.npy"
    np.save(x_path, x.numpy())
    np.save(y_path, np.array(y.numpy(), dtype=np.int64, order="F"))

    x = torch.from_numpy(np.ascontiguousarray(np.load(x_path)))
    y = torch.from_numpy(np.ascontiguousarray(np.load(y_path)))

    a = net(x, y)

    mod = torch.jit.trace(net, (x, y))
    mod.save("test_pnnx_input_npy_int64.pt")

    ret = os.system("../src/pnnx test_pnnx_input_npy_int64.pt input=test_pnnx_input_npy_int64_x.npy,test_pnnx_input_npy_int64_y.npy")
    if ret != 0:
        return False

    import test_pnnx_input_npy_int64_pnnx
    pnnx_net = test_pnnx_input_npy_int64_pnnx.Model()
    pnnx_net.eval()
    b = pnnx_net(x, y)

    if not (torch.equal(a, b)):
        return False
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x, y,))
    torch.export.save(ep, "test_pnnx_input_npy_int64_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_pnnx_input_npy_int64_export.pt2 fp16=0 input=test_pnnx_input_npy_int64_x.npy,test_pnnx_input_npy_int64_y.npy")

    # pnnx inference
    import test_pnnx_input_npy_int64_export_pnnx
    c = test_pnnx_input_npy_int64_export_pnnx.Model().eval()(x, y)
    return a.dtype == c.dtype and torch.equal(a, c)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()

        self.embed = nn.Embedding(num_embeddings=11, embedding_dim=4)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(44, dtype=torch.float32).reshape(11, 4) / 10)

    def forward(self, x):
        return self.embed(x) * 0.5 + 1


def _test_embedding():
    net = EmbeddingModel()
    net.eval()

    x_path = "test_pnnx_input_npy_embedding_x.npy"
    x_np = np.array([[0, 3, 5], [10, 1, 7]], dtype=np.int64, order="F")
    np.save(x_path, x_np)

    x = torch.from_numpy(np.ascontiguousarray(np.load(x_path)))

    a = net(x)

    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_input_npy_embedding.pt")

    ret = os.system("../src/pnnx test_pnnx_input_npy_embedding.pt input=test_pnnx_input_npy_embedding_x.npy")
    if ret != 0:
        return False

    import test_pnnx_input_npy_embedding_pnnx
    pnnx_net = test_pnnx_input_npy_embedding_pnnx.Model()
    pnnx_net.eval()
    b = pnnx_net(x)

    if not (torch.equal(a, b)):
        return False
    if version.parse(torch.__version__) < version.parse('2.11'):
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_pnnx_input_npy_embedding_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_pnnx_input_npy_embedding_export.pt2 fp16=0 input=test_pnnx_input_npy_embedding_x.npy")

    # pnnx inference
    import test_pnnx_input_npy_embedding_export_pnnx
    c = test_pnnx_input_npy_embedding_export_pnnx.Model().eval()(x)
    return a.dtype == c.dtype and torch.equal(a, c)


def test():
    return _test_basic() and _test_input2() and _test_int64() and _test_embedding()


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
