# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, cos0, sin0, y, cos1, sin1):
        # non-interleaved rope with a full-width cache and different cos/sin values for each half
        # this covers 2d / vision rope semantics that a half-width cache cannot express
        x0, x1 = torch.tensor_split(x, (8,), dim=-1)
        rx = torch.cat((-x1, x0), dim=-1)
        out0 = x * cos0 + rx * sin0

        # a second embed_dim and head count
        y0, y1 = torch.tensor_split(y, (12,), dim=-1)
        ry = torch.cat((-y1, y0), dim=-1)
        out1 = y * cos1 + ry * sin1

        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)

    # draw inputs with torch.rand in forward() parameter order to match the generated test_inference()
    # the generated wrapper reseeds with manual_seed(0) and regenerates the inputs
    # random full-width caches give the two halves different values and expose the original bug
    # input layout is (num_heads, seqlen, embed_dim), cache layout is (seqlen, embed_dim)
    x = torch.rand(4, 5, 16)
    cos0 = torch.rand(5, 16)  # full width: w == embed_dim, not embed_dim / 2
    sin0 = torch.rand(5, 16)
    y = torch.rand(3, 7, 24)
    cos1 = torch.rand(7, 24)
    sin1 = torch.rand(7, 24)

    a = net(x, cos0, sin0, y, cos1, sin1)

    # export torchscript
    mod = torch.jit.trace(net, (x, cos0, sin0, y, cos1, sin1))
    mod.save("test_ncnn_fuse_rotaryembed.pt")

    # torchscript to pnnx
    import os
    if os.system("../../src/pnnx test_ncnn_fuse_rotaryembed.pt inputshape=[4,5,16],[5,16],[5,16],[3,7,24],[7,24],[7,24] fp16=0") != 0:
        return False

    # both rope expressions must be fused for this test to exercise RotaryEmbed
    with open("test_ncnn_fuse_rotaryembed.ncnn.param") as f:
        layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
    assert layers.count("RotaryEmbed") == 2

    # ncnn inference
    import test_ncnn_fuse_rotaryembed_ncnn
    b = test_ncnn_fuse_rotaryembed_ncnn.test_inference()

    assert len(a) == len(b)
    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
