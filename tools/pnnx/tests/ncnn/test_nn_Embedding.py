# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.embed_0 = nn.Embedding(embedding_dim=128, num_embeddings=10)

    def forward(self, x):
        x = self.embed_0(x)
        return x

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (13,), dtype=torch.int)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_Embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Embedding.pt inputshape=[13]i32")

    # ncnn inference
    import test_nn_Embedding_ncnn
    b = test_nn_Embedding_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
