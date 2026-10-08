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


class SharedInterpolationShape(nn.Module):
    def forward(self, x, y, z):
        reference = F.max_pool2d(y, 1)
        shape = (reference.size(2), reference.size(3))
        return x.reshape(shape), F.interpolate(z, size=shape, mode="nearest")


class SharedCropShape(nn.Module):
    def forward(self, x, y, z):
        reference = F.max_pool2d(y, 1)
        height, width = reference.size(2), reference.size(3)
        return x.reshape(height, width), z[:, :, :height, :width]


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
    # the third sample must never participate in conversion or shape inference
    shape0 = ",".join(str(list(s)).replace(" ", "") for s in shapes[0])
    shape1 = ",".join(str(list(s)).replace(" ", "") for s in shapes[1])
    pnnxcmd = "../../src/pnnx " + name + ".pt inputshape=" + shape0 + " inputshape2=" + shape1
    if os.system(pnnxcmd) != 0:
        return False

    with ncnn.Net() as net:
        net.opt.num_threads = 1
        if net.load_param(name + ".ncnn.param") != 0 or net.load_model(name + ".ncnn.bin") != 0:
            return False
        for data in inputs:
            expected = model(*data)
            if not isinstance(expected, tuple):
                expected = (expected,)
            if len(data) != len(input_batch_axes) or len(expected) != len(output_batch_axes):
                print(name, "batch axis count mismatch")
                return False
            with net.create_extractor() as ex:
                for i, x in enumerate(data):
                    if ex.input("in%d" % i, ncnn.Mat(x.numpy(), batch_index=input_batch_axes[i]).clone()) != 0:
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
    reference_shapes = [((24,), (2, 1)), ((36,), (3, 1)), ((48,), (2, 1))]
    infer_shapes = [((2, 2, 6),), ((1, 4, 6),), ((3, 2, 6),)]
    intermediate_shapes = [((2, 3, 4, 5),), ((3, 2, 6, 7),), ((4, 2, 8, 9),)]
    cases = [
        # exactly one dynamic output extent, which is a reference, not infer
        ("external", ExternalReference(), reference_shapes, (233, 233), (233,)),
        # input/output metadata match in both conversion samples, but the reshape is not an identity on the third input
        ("noop", ExternalReference(), [((2, 12), (2, 1)), ((3, 12), (3, 1)), ((4, 12), (2, 1))], (233, 233), (233,)),
        ("infer", InferDimension(), infer_shapes, (233,), (233,)),
        ("one_dimensional", OneDimensionalReference(), [((2, 3),), ((3, 4),), ((4, 5),)], (233,), (233,)),
        ("adjacent_infer", AdjacentInfer(), infer_shapes, (233,), (233,)),
        ("adjacent_reference", AdjacentReference(), [((2, 3, 4), (2, 1)), ((3, 3, 4), (3, 1)), ((4, 3, 4), (2, 1))], (233, 233), (233,)),
        ("unsqueeze_reference", UnsqueezeReference(), reference_shapes, (233, 233), (233,)),
        # identical shape expressions are shared, but each reshape infers its own size
        ("shared", SharedInfer(), [((24,), (2, 3), (48,)), ((48,), (3, 4), (96,)), ((48,), (2, 2), (80,))], (233, 233, 233), (233, 233)),
        # two different dynamic extents have the same product while converting
        ("intermediate_reference", IntermediateReference(), [((24,), (2, 3, 1)), ((36,), (3, 2, 1)), ((48,), (4, 2, 1))], (233, 233), (233,)),
        ("flatten", IntermediateFlatten(), intermediate_shapes, (233,), (233,)),
        ("batch_layout", IntermediateBatchLayout(), intermediate_shapes, (0,), (233,)),
        # preserving reshape sizes must not introduce runtime references into other consumers
        ("shared_interp", SharedInterpolationShape(), [((48,), (3, 1, 6, 8), (2, 1, 4, 4)), ((48,), (2, 1, 6, 8), (3, 1, 4, 4)), ((48,), (4, 1, 6, 8), (1, 1, 4, 4))], (233, 0, 0), (233, 0)),
        ("shared_crop", SharedCropShape(), [((48,), (3, 1, 6, 8), (2, 1, 10, 12)), ((48,), (2, 1, 6, 8), (3, 1, 10, 12)), ((48,), (4, 1, 6, 8), (1, 1, 10, 12))], (233, 0, 233), (233, 233)),
    ]
    if version.parse(torch.__version__) >= version.parse("1.13"):
        cases.append(("unflatten", IntermediateUnflatten(), [((2, 3, 4),), ((3, 2, 4),), ((4, 2, 4),)], (233,), (233,)))
        cases.append(("unflatten_reference", UnflattenReference(), [((24, 2), (2, 3, 1)), ((36, 2), (3, 2, 1)), ((48, 2), (4, 2, 1))], (233, 233), (233,)))
    for name, model, shapes, input_batch_axes, output_batch_axes in cases:
        if not run_case(name, model, shapes, input_batch_axes, output_batch_axes):
            return False
        if name in ("shared_interp", "shared_crop"):
            # keep the reference on reshape without adding it to interp or crop
            with open("test_ncnn_reshape_semantics_" + name + ".ncnn.param") as f:
                layers = [line.split() for line in f.readlines()[2:]]
            reshapes = [layer for layer in layers if layer[0] == "Reshape"]
            consumers = [layer for layer in layers if layer[0] in ("Interp", "Crop")]
            if len(reshapes) != 1 or reshapes[0][2] != "2" or len(consumers) != 1 or consumers[0][2] != "1":
                print(name, "unexpected shape reference inputs")
                return False
    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
