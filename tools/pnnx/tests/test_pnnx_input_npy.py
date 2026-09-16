# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch
import torch.nn as nn
from packaging import version

from pnnx_test_utils import PT2_SKIP_RETURN_CODE
from pnnx_test_utils import convert_and_import


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

    mod = convert_and_import(
        net,
        (x, y),
        "test_pnnx_input_npy_basic",
        pnnx_args=("input=test_pnnx_input_npy_basic_x.npy,test_pnnx_input_npy_basic_y.npy",),
        output_basename="test_pnnx_input_npy_basic",
    )

    pnnx_net = mod.Model()
    pnnx_net.eval()
    b = pnnx_net(x, y)

    return _allclose(a, b)


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

    export_kwargs = {}
    if os.environ.get("PNNX_TEST_FORMAT") == "pt2":
        channels = torch.export.Dim("channels", min=2, max=3)
        width = torch.export.Dim("width", min=6, max=8)
        export_kwargs["dynamic_shapes"] = (
            {1: channels, 2: width},
            {1: channels, 2: width},
        )

    mod = convert_and_import(
        net,
        (x0, y0),
        "test_pnnx_input_npy",
        pnnx_args=(
            "input=test_pnnx_input_npy_input2_x0.npy,test_pnnx_input_npy_input2_y0.npy",
            "input2=test_pnnx_input_npy_input2_x1.npy,test_pnnx_input_npy_input2_y1.npy",
        ),
        trace_kwargs={"_store_inputs": False} if version.parse(torch.__version__) >= version.parse("2.0") else {},
        export_kwargs=export_kwargs,
        output_basename="test_pnnx_input_npy_input2",
    )

    pnnx_net = mod.Model()
    pnnx_net.eval()
    b0 = pnnx_net(x0, y0)
    try:
        b1 = pnnx_net(x1, y1)
    except RuntimeError:
        return False

    return _allclose(a0, b0) and _allclose(a1, b1)


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

    mod = convert_and_import(
        net,
        (x, y),
        "test_pnnx_input_npy_int64",
        pnnx_args=("input=test_pnnx_input_npy_int64_x.npy,test_pnnx_input_npy_int64_y.npy",),
        output_basename="test_pnnx_input_npy_int64",
    )

    pnnx_net = mod.Model()
    pnnx_net.eval()
    b = pnnx_net(x, y)

    return torch.equal(a, b)


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

    mod = convert_and_import(
        net,
        (x,),
        "test_pnnx_input_npy_embedding",
        pnnx_args=("input=test_pnnx_input_npy_embedding_x.npy",),
        output_basename="test_pnnx_input_npy_embedding",
    )

    pnnx_net = mod.Model()
    pnnx_net.eval()
    b = pnnx_net(x)

    return torch.equal(a, b)


def test():
    test_cases = (
        ("test_pnnx_input_npy_basic", _test_basic),
        ("test_pnnx_input_npy", _test_input2),
        ("test_pnnx_input_npy_int64", _test_int64),
        ("test_pnnx_input_npy_embedding", _test_embedding),
    )
    skipped = False
    for basename, test_case in test_cases:
        try:
            result = test_case()
        except SystemExit as exc:
            if exc.code != PT2_SKIP_RETURN_CODE:
                raise
            skipped = True
            continue
        if not result:
            return False

    if skipped:
        raise SystemExit(PT2_SKIP_RETURN_CODE)
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
