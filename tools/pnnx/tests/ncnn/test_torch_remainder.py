# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import numpy as np
import torch
import ncnn


class Model(torch.nn.Module):
    def forward(self, x, y):
        return (torch.remainder(x, y), x % y, x.remainder(y),
                torch.remainder(x, 2.0), torch.remainder(x, -2.0),
                torch.remainder(5.0, y))



class BroadcastModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.remainder(x, y), torch.remainder(y, x)


class BatchBroadcastModel(torch.nn.Module):
    def forward(self, x, y):
        return (torch.nn.functional.max_pool2d(torch.remainder(x, y), 1),
                torch.nn.functional.max_pool2d(torch.remainder(y, x), 1))


def test_broadcast(model, shapes, name):
    torch.manual_seed(0)
    inputs = tuple(torch.rand(*shape) for shape in shapes)
    expected = model(*inputs)
    torch.jit.trace(model.eval(), inputs).save(name + ".pt")
    shape_arg = ",".join("[" + ",".join(str(dim) for dim in shape) + "]" for shape in shapes)
    subprocess.run(["../../src/pnnx", name + ".pt",
                    "inputshape=" + shape_arg, "fp16=0"], check=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name + "_ncnn", name + "_ncnn.py")
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    actual = generated.test_inference()
    return len(actual) == len(expected) and all(torch.equal(a, b) for a, b in zip(expected, actual))


def test():
    net = Model().eval()
    torch.manual_seed(0)
    mod = torch.jit.trace(net, (torch.rand(1, 16), torch.rand(1, 16) + 1))
    mod.save("test_torch_remainder.pt")
    subprocess.run(["../../src/pnnx", "test_torch_remainder.pt",
                    "inputshape=[1,16],[1,16]", "fp16=0"], check=True)

    x = torch.tensor([[-1., -3., 1., 3., -1e8, 1e8, 1e-7, -1e-7,
                       -0., 0., float("inf"), -float("inf"),
                       float("nan"), 1., -1., .25]])
    y = torch.tensor([[2., 2., 2., 2., 3., 3., 3., 3., 2., -2.,
                       2., -2., 0., float("inf"), float("inf"), -float("inf")]])
    expected = net(x, y)

    with ncnn.Net() as runtime:
        runtime.opt.use_vulkan_compute = False
        assert runtime.load_param("test_torch_remainder.ncnn.param") == 0
        assert runtime.load_model("test_torch_remainder.ncnn.bin") == 0
        with runtime.create_extractor() as ex:
            assert ex.input("in0", ncnn.Mat(x.numpy().reshape(-1)).clone()) == 0
            assert ex.input("in1", ncnn.Mat(y.numpy().reshape(-1)).clone()) == 0
            for i, reference in enumerate(expected):
                status, output = ex.extract("out" + str(i))
                if status != 0:
                    return False
                actual = np.array(output).reshape(-1)
                reference = reference.detach().numpy().reshape(-1)
                if not np.array_equal(reference, actual, equal_nan=True):
                    return False
                zeros = (reference == 0) & (actual == 0)
                if not np.array_equal(np.signbit(reference[zeros]), np.signbit(actual[zeros])):
                    return False
    return (test_broadcast(BroadcastModel(), ((4, 3, 4), (4,)), "test_torch_remainder_broadcast")
            and test_broadcast(BatchBroadcastModel(), ((1, 2, 3, 4), (2, 3, 4)), "test_torch_remainder_batch"))


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
