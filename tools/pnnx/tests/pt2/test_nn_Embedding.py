# Copyright 2026 Tencent
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
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (1, 13), dtype=torch.int)

    a = net(x)

    # export pt2
    mod = torch.export.export(net, (x,))
    torch.export.save(mod, "test_nn_Embedding.pt2")

    # pt2 to pnnx
    import os
    os.system("../../src/pnnx test_nn_Embedding.pt2 inputshape=[1,13]i32")

    # pnnx inference
    import test_nn_Embedding_pnnx
    b = test_nn_Embedding_pnnx.test_inference()

    return torch.equal(a, b)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
