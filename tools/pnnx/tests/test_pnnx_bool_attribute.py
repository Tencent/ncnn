# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from packaging import version

class Model(nn.Module):
    def __init__(self, shape):
        super(Model, self).__init__()

        count = 1
        for size in shape:
            count *= size

        if shape == ():
            values = torch.tensor(0.5)
        else:
            values = (torch.arange(count, dtype=torch.float32) % 2 - 0.5).reshape(shape)
        self.register_buffer("values", values)

    def forward(self, x):
        # fold the comparison into a bool constant
        return x + 1, self.values > 0

def test_bool_attribute(shape, fp16):
    net = Model(shape)
    net.eval()

    x = torch.ones(2)
    mask = (net.values > 0).numpy().tobytes()

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_bool_attribute.pt")

    # torchscript to pnnx
    import os
    ret = os.system("../src/pnnx test_pnnx_bool_attribute.pt inputshape=[2] fp16=%d" % fp16)
    if ret != 0:
        return False

    # check bool storage type and element count
    with open("test_pnnx_bool_attribute.ncnn.param") as f:
        constants = [line.split() for line in f if line.startswith("MemoryData ")]
    if len(constants) != 1:
        return False
    params = constants[0][5:]
    if "21=3" not in params or "0=%d" % len(mask) not in params:
        return False

    # check raw bool bytes and zero padding to 4bytes
    with open("test_pnnx_bool_attribute.ncnn.bin", "rb") as f:
        data = f.read()
    if data != mask + bytes((-len(mask)) % 4):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    name = "test_pnnx_bool_attribute_%s_%d_export" % ("scalar" if shape == () else shape[0], fp16)
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, name + ".pt2")

    # exported program to pnnx
    os.system("../src/pnnx " + name + ".pt2 fp16=0")

    # pnnx inference
    pnnx_export = __import__(name + "_pnnx")
    c0, c1 = pnnx_export.Model().eval()(x)
    a0, a1 = net(x)
    return c1.dtype == torch.bool and a1.shape == c1.shape and torch.equal(a0, c0) and torch.equal(a1, c1)

def test():
    for fp16 in (0, 1):
        for shape in ((), (1,), (2,), (3,), (4,), (5,)):
            if not test_bool_attribute(shape, fp16):
                print("test_bool_attribute failed shape=%s fp16=%d" % (shape, fp16))
                return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
