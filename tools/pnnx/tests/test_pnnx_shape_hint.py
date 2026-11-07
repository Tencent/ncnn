# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        y = self.act(self.conv(x))
        y = self.pool(y).flatten(1)
        return self.fc(y)


def read_shape_hints(path):
    """return {layer type: [(dims, w, h, d, c), ...]} for the layers carrying a -23330 hint"""
    hints = {}
    with open(path) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            if not tokens[0][0].isalpha():
                # magic number, layer and blob counts, and continuation lines
                continue
            for t in tokens:
                if not t.startswith("-23330="):
                    continue
                values = [int(v) for v in t.split("=", 1)[1].split(",")]
                count = values[0]
                records = values[1:]
                if count != len(records) or count % 5 != 0:
                    raise ValueError("layer %s has %d hint values for count %d" % (tokens[1], len(records), count))
                hints[tokens[0]] = [tuple(records[i:i + 5]) for i in range(0, count, 5)]
    return hints


def test_shape_hint():
    net = Model()
    net.eval()

    x = torch.rand(1, 3, 32, 32)
    mod = torch.jit.trace(net, x)
    mod.save("test_pnnx_shape_hint.pt")

    ret = os.system("../src/pnnx test_pnnx_shape_hint.pt inputshape=[1,3,32,32]")
    if ret != 0:
        return False

    hints = read_shape_hints("test_pnnx_shape_hint.ncnn.param")

    # the torch shapes are reduced by the batch axis, the remaining axes map to Mat(w,h,d,c)
    # in reverse order, so torch [1,3,32,32] becomes dims=3 w=32 h=32 c=3
    expected = {
        "Input": [(3, 32, 32, 1, 3)],
        "Convolution": [(3, 32, 32, 1, 8)],
        "Pooling": [(1, 8, 1, 1, 1)],
        "InnerProduct": [(1, 4, 1, 1, 1)],
    }

    for layer_type, records in expected.items():
        if hints.get(layer_type) != records:
            print("unexpected shape hint for %s: %s" % (layer_type, hints.get(layer_type)))
            return False

    return True


def test():
    if not test_shape_hint():
        print("test_shape_hint failed")
        return False

    return True


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
