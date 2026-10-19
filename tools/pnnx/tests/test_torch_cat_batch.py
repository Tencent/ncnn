import importlib
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, y):
        return torch.cat((x, y), dim=0)


def compare_case(net, converted, inputs):
    expected = net(*inputs)
    actual = converted(*inputs)

    if actual.shape != expected.shape:
        print("shape mismatch", actual.shape, expected.shape)
        return False
    if not torch.allclose(actual, expected, 1e-4, 1e-4):
        print("max error", (actual - expected).abs().max().item())
        return False

    return True


def test():
    torch.manual_seed(0)
    net = Model().eval()
    inputs = (torch.rand(2, 3, 5, 7), torch.rand(3, 3, 5, 7))
    inputs2 = (torch.rand(1, 3, 5, 7), torch.rand(2, 3, 5, 7))

    name = "test_torch_cat_batch"
    mod = torch.jit.trace(net, inputs)
    mod.save(name + ".pt")

    if os.system("../src/pnnx " + name + ".pt inputshape=[2,3,5,7],[3,3,5,7] inputshape2=[1,3,5,7],[2,3,5,7]") != 0:
        return False

    converted = importlib.import_module(name + "_pnnx").Model().eval()
    return compare_case(net, converted, inputs) and compare_case(net, converted, inputs2)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
