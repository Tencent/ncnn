# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import ncnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


class ExternalReference(nn.Module):
    def forward(self, x, y):
        return x.reshape(y.size(0), -1)


class InferDimension(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 12)


class OneDimensionalReference(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0) * x.size(1))


class AdjacentInfer(nn.Module):
    def forward(self, x):
        return x.reshape(-1).reshape(-1, 12)


class AdjacentReference(nn.Module):
    def forward(self, x, y):
        return x.reshape(-1).reshape(y.size(0), -1)


class UnsqueezeReference(nn.Module):
    def forward(self, x, y):
        return x.reshape(y.size(0), -1).unsqueeze(0)


class SharedInfer(nn.Module):
    def forward(self, x, y, z):
        shape = (y.size(0), y.size(1), -1)
        return x.reshape(shape), z.reshape(shape)


class IntermediateReference(nn.Module):
    def forward(self, x, y):
        z = y.flatten(0, 1)
        return x.reshape(z.size(0), -1)


class IntermediateFlatten(nn.Module):
    def forward(self, x):
        return x.flatten(0, 1).transpose(0, 1).flatten(1, 2)


class IntermediateUnflatten(nn.Module):
    def forward(self, x):
        return x.flatten(0, 1).unflatten(0, (2, -1))


class UnflattenReference(nn.Module):
    def forward(self, x, y):
        z = y.flatten(0, 1)
        return x.unflatten(-2, (z.size(0), -1))


class IntermediateBatchLayout(nn.Module):
    def forward(self, x):
        x = F.max_pool2d(x, 1).flatten(0, 1)
        x = F.max_pool1d(x, 1)
        return x.transpose(0, 1).flatten(1, 2)


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


def run_case(name, model, shapes, input_batch_axes, output_batch_axes):
    name = "test_ncnn_reshape_semantics_" + name
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
    pnnxcmd = "../../src/pnnx " + name + ".pt inputshape=" + shape0 + " inputshape2=" + shape1
    if os.system(pnnxcmd) != 0:
        return False

    with ncnn.Net() as net:
        net.opt.num_threads = 1
        if net.load_param(name + ".ncnn.param") != 0 or net.load_model(name + ".ncnn.bin") != 0:
            return False
        input_names = net.input_names()
        for data in inputs:
            expected = model(*data)
            if not isinstance(expected, tuple):
                expected = (expected,)
            if len(data) != len(input_batch_axes) or len(expected) != len(output_batch_axes):
                print(name, "batch axis count mismatch")
                return False
            with net.create_extractor() as ex:
                for i, x in enumerate(data):
                    if ex.input(input_names[i], ncnn.Mat(x.numpy(), batch_index=input_batch_axes[i]).clone()) != 0:
                        print(name, "input", i, "failed")
                        return False
                actual = []
                for i, axis in enumerate(output_batch_axes):
                    ret, out = ex.extract("out%d" % i)
                    if ret != 0:
                        print(name, "output", i, "failed", ret)
                        return False
                    actual.append(torch.from_numpy(out.numpy(batch_index=axis).copy()))
            if not compare(expected, tuple(actual)):
                print(name, "ncnn inputs", [tuple(x.shape) for x in data])
                return False
    return True


def test():
    reference_shapes = [((24,), (2, 1)), ((36,), (3, 1)), ((48,), (4, 1))]
    infer_shapes = [((2, 2, 6),), ((1, 6, 6),), ((3, 2, 6),)]
    static_infer_shapes = [((2, 2, 6),), ((1, 4, 6),), ((4, 1, 6),)]
    intermediate_shapes = [((2, 3, 4, 5),), ((3, 4, 6, 7),), ((4, 2, 8, 9),)]
    cases = [
        # one dynamic output extent can be inferred after folding the static extent
        ("external", ExternalReference(), reference_shapes, (233, 233), (233,)),
        ("dynamic_external", ExternalReference(), [((24,), (2, 1)), ((30,), (3, 1)), ((48,), (2, 1))], (233, 233), (233,)),
        # matching input/output metadata with one dynamic extent allows noop elimination
        ("noop", ExternalReference(), [((2, 12), (2, 1)), ((3, 12), (3, 1)), ((4, 12), (4, 1))], (233, 233), (233,)),
        ("infer", InferDimension(), infer_shapes, (233,), (233,)),
        ("static_infer", InferDimension(), static_infer_shapes, (233,), (233,)),
        ("one_dimensional", OneDimensionalReference(), [((2, 3),), ((3, 4),), ((4, 5),)], (233,), (233,)),
        ("adjacent_infer", AdjacentInfer(), infer_shapes, (233,), (233,)),
        ("adjacent_reference", AdjacentReference(), [((2, 3, 4), (2, 1)), ((3, 4, 4), (3, 1)), ((4, 3, 4), (2, 1))], (233, 233), (233,)),
        ("unsqueeze_reference", UnsqueezeReference(), reference_shapes, (233, 233), (233,)),
        # identical shape expressions are shared, but each reshape infers its own size
        ("shared", SharedInfer(), [((24,), (2, 3), (48,)), ((48,), (3, 4), (96,)), ((16,), (2, 2), (32,))], (233, 233, 233), (233, 233)),
        # intermediate dimensions are dynamic only when conversion samples vary them
        ("intermediate_reference", IntermediateReference(), [((24,), (2, 3, 1)), ((36,), (3, 4, 1)), ((48,), (4, 2, 1))], (233, 233), (233,)),
        ("flatten", IntermediateFlatten(), intermediate_shapes, (233,), (233,)),
        ("batch_layout", IntermediateBatchLayout(), intermediate_shapes, (0,), (233,)),
    ]
    if version.parse(torch.__version__) >= version.parse("1.13"):
        cases.append(("unflatten", IntermediateUnflatten(), [((2, 3, 4),), ((3, 4, 4),), ((4, 2, 4),)], (233,), (233,)))
        cases.append(("unflatten_reference", UnflattenReference(), [((24, 2), (2, 3, 1)), ((36, 2), (3, 4, 1)), ((48, 2), (4, 2, 1))], (233, 233), (233,)))
    for name, model, shapes, input_batch_axes, output_batch_axes in cases:
        if not run_case(name, model, shapes, input_batch_axes, output_batch_axes):
            return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
