# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

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

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_Embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_Embedding.pt inputshape=[1,13]i32")

    # pnnx inference
    import test_nn_Embedding_pnnx
    b = test_nn_Embedding_pnnx.test_inference()

    if not (torch.equal(a, b)):
        return False

    if version.parse(torch.__version__) < version.parse('2.11') or os.environ.get('PNNX_BUILD_PT2', '1') == '0':
        return True

    # export exported program
    ep = torch.export.export(net, (x,))
    torch.export.save(ep, "test_nn_Embedding_export.pt2")

    # exported program to pnnx
    os.system("../src/pnnx test_nn_Embedding_export.pt2 fp16=0")

    # pnnx inference
    import test_nn_Embedding_export_pnnx
    c = test_nn_Embedding_export_pnnx.test_inference()

    return torch.equal(a, c)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
