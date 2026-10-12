# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn


class InferDimension(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 12)


class AdjacentReference(nn.Module):
    def forward(self, x, y):
        return x.reshape(-1).reshape(y.size(0), -1)


class SharedInfer(nn.Module):
    def forward(self, x, y, z):
        shape = (y.size(0), y.size(1), -1)
        return x.reshape(shape), z.reshape(shape)


def compare(expected, actual):
    if not isinstance(expected, tuple):
        expected = (expected,)
    if not isinstance(actual, tuple):
        actual = (actual,)
    if len(expected) != len(actual):
        return False
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a.shape != b.shape or not torch.equal(a, b):
            print("output", i, "expected", tuple(a.shape), "actual", tuple(b.shape))
            return False
    return True


def run_case(name, model, shapes):
    name = "test_pnnx_reshape_semantics_" + name
    model.eval()
    inputs = []
    for sample in shapes:
        data = []
        for shape in sample:
            count = 1
            for size in shape:
                count *= size
            data.append(torch.arange(count, dtype=torch.float32).reshape(shape))
        inputs.append(tuple(data))

    torch.jit.trace(model, inputs[0]).save(name + ".pt")
    # conversion samples cover the dynamic extents used by the third sample
    shape0 = ",".join(str(list(s)).replace(" ", "") for s in shapes[0])
    shape1 = ",".join(str(list(s)).replace(" ", "") for s in shapes[1])
    pnnxcmd = "../src/pnnx " + name + ".pt inputshape=" + shape0 + " inputshape2=" + shape1
    if os.system(pnnxcmd) != 0:
        return False

    pnnxpy = __import__(name + "_pnnx")
    converted = pnnxpy.Model().eval()
    for data in inputs:
        if not compare(model(*data), converted(*data)):
            print(name, "pnnx inputs", [tuple(x.shape) for x in data])
            return False

    return True


def test():
    cases = [
        # static output metadata can replace the inferred dimension
        ("static_infer", InferDimension(), [((2, 2, 6),), ((1, 4, 6),), ((4, 1, 6),)]),
        # fusion must not turn two unknown extents into two infer dimensions
        ("adjacent_reference", AdjacentReference(), [((2, 3, 4), (2, 1)), ((3, 4, 4), (3, 1)), ((4, 3, 4), (2, 1))]),
        # identical shape expressions are shared, but each reshape infers its own size
        ("shared", SharedInfer(), [((24,), (2, 3), (48,)), ((48,), (3, 4), (96,)), ((16,), (2, 2), (32,))]),
    ]
    for name, model, shapes in cases:
        if not run_case(name, model, shapes):
            return False
        if name == "static_infer":
            with open("test_pnnx_reshape_semantics_" + name + ".pnnx.param") as f:
                reshapes = [line for line in f if line.startswith("Tensor.reshape ")]
            if len(reshapes) != 1 or "shape=(2,12)" not in reshapes[0]:
                print(name, "static output shape was not folded")
                return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
