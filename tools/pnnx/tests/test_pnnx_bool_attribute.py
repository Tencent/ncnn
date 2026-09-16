# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, shape):
        super(Model, self).__init__()

        count = 1
        for size in shape:
            count *= size
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
    return data == mask + bytes((-len(mask)) % 4)

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
