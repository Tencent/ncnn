# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
from packaging import version

from pnnx_test_utils import export_model, import_model, model_formats, run_pnnx


def _allclose(a, b):
    for a0, b0 in zip(a, b):
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


def _test_formats(net, export_inputs, cases, name, arguments, compare=_allclose, store_trace_inputs=True,
                  dynamic_shapes=None):
    for model_format in model_formats():
        if model_format == "torchscript" and not store_trace_inputs:
            model_path = name + "_torchscript.pt"
            torch.jit.trace(net, export_inputs, _store_inputs=False).save(model_path)
        else:
            model_path = export_model(net, export_inputs, name, model_format, dynamic_shapes=dynamic_shapes)

        output_prefix = name + "_" + model_format
        if run_pnnx(model_path, output_prefix, arguments).returncode != 0:
            return False

        pnnx_net = import_model(output_prefix + "_pnnx.py", output_prefix + "_pnnx")
        for inputs, expected in cases:
            if not compare(expected, pnnx_net(*inputs)):
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

    arguments = ("input=" + x_path + "," + y_path,)
    return _test_formats(net, (x, y), (((x, y), a),), "test_pnnx_input_npy_basic", arguments)


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

    arguments = (
        "input=" + x0_path + "," + y0_path,
        "input2=" + x1_path + "," + y1_path,
    )
    cases = (((x0, y0), a0), ((x1, y1), a1))
    store_trace_inputs = version.parse(torch.__version__) < version.parse("2.0")
    channel = torch.export.Dim("channel", min=2, max=3)
    width = torch.export.Dim("width", min=6, max=8)
    dynamic_shapes = ({1: channel, 2: width}, {1: channel, 2: width})
    return _test_formats(net, (x0, y0), cases, "test_pnnx_input_npy_input2", arguments,
                         store_trace_inputs=store_trace_inputs, dynamic_shapes=dynamic_shapes)


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

    arguments = ("input=" + x_path + "," + y_path,)
    return _test_formats(net, (x, y), (((x, y), a),), "test_pnnx_input_npy_int64", arguments, compare=torch.equal)


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

    arguments = ("input=" + x_path,)
    return _test_formats(net, (x,), (((x,), a),), "test_pnnx_input_npy_embedding", arguments, compare=torch.equal)


def test():
    return _test_basic() and _test_input2() and _test_int64() and _test_embedding()


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
