# Copyright 2023 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as npy

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15):
        return (x0 - x1.expand_as(x0), x1.expand(x0.size()) - x0,
                y0 - y1.expand_as(y0), y1.expand(y0.size()) - y0,
                y0 - y2.expand_as(y0), y2.expand(y0.size()) - y0,
                y0 - y3.expand_as(y0), y3.expand(y0.size()) - y0,
                y1 - y2.expand_as(y0), y2.expand(y0.size()) - y1,
                y1 - y3.expand_as(y1), y3.expand(y1.size()) - y1,
                y2 - y3.expand_as(y2), y3.expand(y2.size()) - y2,
                z0 - z1.expand_as(z0), z1.expand(z0.size()) - z0,
                z0 - z2.expand_as(z0), z2.expand(z0.size()) - z0,
                z0 - z3.expand_as(z0), z3.expand(z0.size()) - z0,
                z0 - z4.expand_as(z0), z4.expand(z0.size()) - z0,
                z0 - z5.expand_as(z0), z5.expand(z0.size()) - z0,
                z0 - z6.expand_as(z0), z6.expand(z0.size()) - z0,
                z0 - z7.expand_as(z0), z7.expand(z0.size()) - z0,
                z1 - z2.expand_as(z0), z2.expand(z0.size()) - z1,
                z1 - z3.expand_as(z0), z3.expand(z0.size()) - z1,
                z1 - z4.expand_as(z1), z4.expand(z1.size()) - z1,
                z1 - z5.expand_as(z1), z5.expand(z1.size()) - z1,
                z1 - z6.expand_as(z3), z6.expand(z3.size()) - z1,
                z1 - z7.expand_as(z1), z7.expand(z1.size()) - z1,
                z2 - z3.expand_as(z0), z3.expand(z0.size()) - z2,
                z2 - z4.expand_as(z2), z4.expand(z2.size()) - z2,
                z2 - z5.expand_as(z3), z5.expand(z3.size()) - z2,
                z2 - z6.expand_as(z2), z6.expand(z2.size()) - z2,
                z2 - z7.expand_as(z2), z7.expand(z2.size()) - z2,
                z3 - z4.expand_as(z1), z4.expand(z1.size()) - z3,
                z3 - z5.expand_as(z3), z5.expand(z3.size()) - z3,
                z3 - z6.expand_as(z3), z6.expand(z3.size()) - z3,
                z3 - z7.expand_as(z3), z7.expand(z3.size()) - z3,
                z4 - z5.expand_as(z1), z5.expand(z1.size()) - z4,
                z4 - z6.expand_as(z2), z6.expand(z2.size()) - z4,
                z4 - z7.expand_as(z4), z7.expand(z4.size()) - z4,
                z5 - z6.expand_as(z3), z6.expand(z3.size()) - z5,
                z5 - z7.expand_as(z5), z7.expand(z5.size()) - z5,
                z6 - z7.expand_as(z6), z7.expand(z6.size()) - z6,
                w0 - w1.expand_as(w0), w1.expand(w0.size()) - w0,
                w0 - w2.expand_as(w0), w2.expand(w0.size()) - w0,
                w0 - w3.expand_as(w0), w3.expand(w0.size()) - w0,
                w0 - w4.expand_as(w0), w4.expand(w0.size()) - w0,
                w0 - w5.expand_as(w0), w5.expand(w0.size()) - w0,
                w0 - w6.expand_as(w0), w6.expand(w0.size()) - w0,
                w0 - w7.expand_as(w0), w7.expand(w0.size()) - w0,
                w0 - w8.expand_as(w0), w8.expand(w0.size()) - w0,
                w0 - w9.expand_as(w0), w9.expand(w0.size()) - w0,
                w0 - w10.expand_as(w0), w10.expand(w0.size()) - w0,
                w0 - w11.expand_as(w0), w11.expand(w0.size()) - w0,
                w0 - w12.expand_as(w0), w12.expand(w0.size()) - w0,
                w0 - w13.expand_as(w0), w13.expand(w0.size()) - w0,
                w0 - w14.expand_as(w0), w14.expand(w0.size()) - w0,
                w0 - w15.expand_as(w0), w15.expand(w0.size()) - w0,
                w1 - w5.expand_as(w1), w5.expand(w1.size()) - w1,
                w1 - w6.expand_as(w1), w6.expand(w1.size()) - w1,
                w1 - w7.expand_as(w1), w7.expand(w1.size()) - w1,
                w1 - w11.expand_as(w1), w11.expand(w1.size()) - w1,
                w1 - w12.expand_as(w1), w12.expand(w1.size()) - w1,
                w1 - w13.expand_as(w1), w13.expand(w1.size()) - w1,
                w1 - w15.expand_as(w1), w15.expand(w1.size()) - w1,
                w2 - w5.expand_as(w2), w5.expand(w2.size()) - w2,
                w2 - w8.expand_as(w2), w8.expand(w2.size()) - w2,
                w2 - w9.expand_as(w2), w9.expand(w2.size()) - w2,
                w2 - w11.expand_as(w2), w11.expand(w2.size()) - w2,
                w2 - w12.expand_as(w2), w12.expand(w2.size()) - w2,
                w2 - w14.expand_as(w2), w14.expand(w2.size()) - w2,
                w2 - w15.expand_as(w2), w15.expand(w2.size()) - w2,
                w3 - w6.expand_as(w3), w6.expand(w3.size()) - w3,
                w3 - w8.expand_as(w3), w8.expand(w3.size()) - w3,
                w3 - w10.expand_as(w3), w10.expand(w3.size()) - w3,
                w3 - w11.expand_as(w3), w11.expand(w3.size()) - w3,
                w3 - w13.expand_as(w3), w13.expand(w3.size()) - w3,
                w3 - w14.expand_as(w3), w14.expand(w3.size()) - w3,
                w3 - w15.expand_as(w3), w15.expand(w3.size()) - w3,
                w4 - w7.expand_as(w4), w7.expand(w4.size()) - w4,
                w4 - w9.expand_as(w4), w9.expand(w4.size()) - w4,
                w4 - w10.expand_as(w4), w10.expand(w4.size()) - w4,
                w4 - w12.expand_as(w4), w12.expand(w4.size()) - w4,
                w4 - w13.expand_as(w4), w13.expand(w4.size()) - w4,
                w4 - w14.expand_as(w4), w14.expand(w4.size()) - w4,
                w4 - w15.expand_as(w4), w15.expand(w4.size()) - w4,
                w5 - w11.expand_as(w5), w11.expand(w5.size()) - w5,
                w5 - w12.expand_as(w5), w12.expand(w5.size()) - w5,
                w5 - w15.expand_as(w5), w15.expand(w5.size()) - w5,
                w6 - w11.expand_as(w6), w11.expand(w6.size()) - w6,
                w6 - w13.expand_as(w6), w13.expand(w6.size()) - w6,
                w6 - w15.expand_as(w6), w15.expand(w6.size()) - w6,
                w7 - w12.expand_as(w7), w12.expand(w7.size()) - w7,
                w7 - w13.expand_as(w7), w13.expand(w7.size()) - w7,
                w7 - w15.expand_as(w7), w15.expand(w7.size()) - w7,
                w8 - w11.expand_as(w8), w11.expand(w8.size()) - w8,
                w8 - w14.expand_as(w8), w14.expand(w8.size()) - w8,
                w8 - w15.expand_as(w8), w15.expand(w8.size()) - w8,
                w9 - w12.expand_as(w9), w12.expand(w9.size()) - w9,
                w9 - w14.expand_as(w9), w14.expand(w9.size()) - w9,
                w9 - w15.expand_as(w9), w15.expand(w9.size()) - w9,
                w10 - w13.expand_as(w10), w13.expand(w10.size()) - w10,
                w10 - w14.expand_as(w10), w14.expand(w10.size()) - w10,
                w10 - w15.expand_as(w10), w15.expand(w10.size()) - w10,
                w11 - w15.expand_as(w11), w15.expand(w11.size()) - w11,
                w12 - w15.expand_as(w12), w15.expand(w12.size()) - w12,
                w13 - w15.expand_as(w13), w15.expand(w13.size()) - w13,
                w14 - w15.expand_as(w14), w15.expand(w14.size()) - w14,

                # some negative cases
                w11.expand_as(w5) - w14.expand_as(w10),
                w5.expand(w1.size()) - w11,
                w15.expand(6, 7, 8, 9) - w14
                )

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x0 = torch.rand(5)
    x1 = torch.rand(1)
    y0 = torch.rand(7, 5)
    y1 = torch.rand(1, 5)
    y2 = torch.rand(7, 1)
    y3 = torch.rand(1, 1)
    z0 = torch.rand(4, 7, 5)
    z1 = torch.rand(1, 7, 5)
    z2 = torch.rand(4, 1, 5)
    z3 = torch.rand(4, 7, 1)
    z4 = torch.rand(1, 1, 5)
    z5 = torch.rand(1, 7, 1)
    z6 = torch.rand(4, 1, 1)
    z7 = torch.rand(1, 1, 1)
    w0 = torch.rand(6, 4, 7, 5)
    w1 = torch.rand(1, 4, 7, 5)
    w2 = torch.rand(6, 1, 7, 5)
    w3 = torch.rand(6, 4, 1, 5)
    w4 = torch.rand(6, 4, 7, 1)
    w5 = torch.rand(1, 1, 7, 5)
    w6 = torch.rand(1, 4, 1, 5)
    w7 = torch.rand(1, 4, 7, 1)
    w8 = torch.rand(6, 1, 1, 5)
    w9 = torch.rand(6, 1, 7, 1)
    w10 = torch.rand(6, 4, 1, 1)
    w11 = torch.rand(1, 1, 1, 5)
    w12 = torch.rand(1, 1, 7, 1)
    w13 = torch.rand(1, 4, 1, 1)
    w14 = torch.rand(6, 1, 1, 1)
    w15 = torch.rand(1, 1, 1, 1)

    nx0 = x0.numpy()
    nx1 = x1.numpy()
    ny0 = y0.numpy()
    ny1 = y1.numpy()
    ny2 = y2.numpy()
    ny3 = y3.numpy()
    nz0 = z0.numpy()
    nz1 = z1.numpy()
    nz2 = z2.numpy()
    nz3 = z3.numpy()
    nz4 = z4.numpy()
    nz5 = z5.numpy()
    nz6 = z6.numpy()
    nz7 = z7.numpy()
    nw0 = w0.numpy()
    nw1 = w1.numpy()
    nw2 = w2.numpy()
    nw3 = w3.numpy()
    nw4 = w4.numpy()
    nw5 = w5.numpy()
    nw6 = w6.numpy()
    nw7 = w7.numpy()
    nw8 = w8.numpy()
    nw9 = w9.numpy()
    nw10 = w10.numpy()
    nw11 = w11.numpy()
    nw12 = w12.numpy()
    nw13 = w13.numpy()
    nw14 = w14.numpy()
    nw15 = w15.numpy()

    npy.save("x0.npy", nx0);
    npy.save("x1.npy", nx1);
    npy.save("y0.npy", ny0);
    npy.save("y1.npy", ny1);
    npy.save("y2.npy", ny2);
    npy.save("y3.npy", ny3);
    npy.save("z0.npy", nz0);
    npy.save("z1.npy", nz1);
    npy.save("z2.npy", nz2);
    npy.save("z3.npy", nz3);
    npy.save("z4.npy", nz4);
    npy.save("z5.npy", nz5);
    npy.save("z6.npy", nz6);
    npy.save("z7.npy", nz7);
    npy.save("w0.npy", nw0);
    npy.save("w1.npy", nw1);
    npy.save("w2.npy", nw2);
    npy.save("w3.npy", nw3);
    npy.save("w4.npy", nw4);
    npy.save("w5.npy", nw5);
    npy.save("w6.npy", nw6);
    npy.save("w7.npy", nw7);
    npy.save("w8.npy", nw8);
    npy.save("w9.npy", nw9);
    npy.save("w10.npy", nw10);
    npy.save("w11.npy", nw11);
    npy.save("w12.npy", nw12);
    npy.save("w13.npy", nw13);
    npy.save("w14.npy", nw14);
    npy.save("w15.npy", nw15);


    a = net(x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15)

    # export torchscript
    mod = torch.jit.trace(net, (x0, x1, y0, y1, y2, y3, z0, z1, z2, z3, z4, z5, z6, z7, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15))
    mod.save("test_pnnx_eliminate_noop_expand.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_pnnx_eliminate_noop_expand.pt input=x0.npy,x1.npy,y0.npy,y1.npy,y2.npy,y3.npy,z0.npy,z1.npy,z2.npy,z3.npy,z4.npy,z5.npy,z6.npy,z7.npy,w0.npy,w1.npy,w2.npy,w3.npy,w4.npy,w5.npy,w6.npy,w7.npy,w8.npy,w9.npy,w10.npy,w11.npy,w12.npy,w13.npy,w14.npy,w15.npy")

    # pnnx inference
    import test_pnnx_eliminate_noop_expand_pnnx
    b = test_pnnx_eliminate_noop_expand_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        # allclose may auto broadcast compare
        if a0.shape != b0.shape:
            return False
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
