# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# Reproducer for fixtures/pt2_producer/*.pt2 : real ExportedProgram archives
# written by several torch releases in the 2.8+ pt2-archive era.
#
# Unlike fixtures/pt2_schema (which downgrades one 2.13 export in place), these
# were produced by the actual producer versions, so they pin that the loader
# accepts the layout / field set each release wrote without a producer gate.
# The 2.8+ container layout is stable, but the raw-payload schema minor and the
# argument-variant field set move between releases.
#
# The model uses deterministic (non-random) parameters so every fixture encodes
# the same weights and can be checked against an eager reference computed with
# whatever torch the CI happens to have. The test imports this module to build
# that reference, so the model definition below is the single source of truth.
#
# usage, with the target torch release first on PYTHONPATH, from this directory:
#   pip install --target /tmp/pp290 \
#       --index-url https://download.pytorch.org/whl/cpu "torch==2.9.0"
#   PYTHONPATH=/tmp/pp290 python generate.py 2.9.0
#   PYTHONPATH=/tmp/pp290 python generate.py            # tag from torch.__version__

import os
import sys

import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))

INPUT_SHAPE = (1, 3, 8, 8)


class M(nn.Module):
    # a conv weight + bias (state dict), a persistent buffer (state dict) and a
    # literal constant folded into the graph; enough to exercise the three
    # payload classes plus a multi-op graph, while staying version-stable.
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.register_buffer("scale", torch.ones(4, 1, 1))

    def forward(self, x):
        y = self.c(x)
        y = torch.relu(y + 1.0)
        return y * self.scale


def build_model():
    m = M().eval()
    with torch.no_grad():
        w = torch.arange(4 * 3 * 3 * 3, dtype=torch.float32) / 100.0 - 0.5
        b = torch.arange(4, dtype=torch.float32) / 8.0 - 0.2
        s = torch.arange(4, dtype=torch.float32) / 4.0 + 1.0
        m.c.weight.copy_(w.reshape(4, 3, 3, 3))
        m.c.bias.copy_(b)
        m.scale.copy_(s.reshape(4, 1, 1))
    return m


def example_input():
    n = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2] * INPUT_SHAPE[3]
    return (torch.arange(n, dtype=torch.float32) / 64.0 - 1.0).reshape(*INPUT_SHAPE)


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else torch.__version__.split("+")[0]
    tag = ver.replace(".", "_")
    path = os.path.join(OUT, "producer_%s.pt2" % tag)

    m = build_model()
    x = example_input()
    with torch.no_grad():
        ep = torch.export.export(m, (x,))
        torch.export.save(ep, path)

    print("wrote %s with torch %s" % (os.path.basename(path), torch.__version__))


if __name__ == "__main__":
    main()
