# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, cos0, sin0, y, cos1, sin1):
        # non-interleaved rope with a full-width cache: cos/sin are as wide as embed_dim, and the
        # two halves carry different values. That is what 2D / vision rope produces, and it is the
        # case a half-width cache cannot express.
        x0, x1 = torch.tensor_split(x, (8,), dim=-1)
        rx = torch.cat((-x1, x0), dim=-1)
        out0 = x * cos0 + rx * sin0

        # same shape family, different embed_dim, to cover a second pack path
        y0, y1 = torch.tensor_split(y, (12,), dim=-1)
        ry = torch.cat((-y1, y0), dim=-1)
        out1 = y * cos1 + ry * sin1

        return out0, out1

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)

    # Draw every input with torch.rand, in forward() parameter order. The generated
    # test_ncnn_fuse_rotaryembed_ncnn.py takes no arguments: it reseeds with manual_seed(0) and
    # regenerates each input the same way. An input built any other way -- a computed cos/sin table,
    # or a draw taken out of order -- would leave the two sides comparing different tensors, and the
    # test would then fail unconditionally, proving nothing.
    #
    # Random caches are also the stronger test here: the bug is that the layer reads a single cache
    # row for BOTH halves, so it only shows when the halves differ, which random values guarantee.
    x = torch.rand(1, 4, 5, 16)      # (batch, num_heads, seqlen, embed_dim)
    cos0 = torch.rand(1, 5, 16)      # full width: w == embed_dim, not embed_dim / 2
    sin0 = torch.rand(1, 5, 16)
    y = torch.rand(1, 3, 7, 24)
    cos1 = torch.rand(1, 7, 24)
    sin1 = torch.rand(1, 7, 24)

    a = net(x, cos0, sin0, y, cos1, sin1)

    # export torchscript
    mod = torch.jit.trace(net, (x, cos0, sin0, y, cos1, sin1))
    mod.save("test_ncnn_fuse_rotaryembed.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_ncnn_fuse_rotaryembed.pt inputshape=[1,4,5,16],[1,5,16],[1,5,16],[1,3,7,24],[1,7,24],[1,7,24] fp16=0")

    # ncnn inference
    import test_ncnn_fuse_rotaryembed_ncnn
    b = test_ncnn_fuse_rotaryembed_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
