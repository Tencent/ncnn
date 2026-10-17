# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnnx_test_utils import convert_and_import_ncnn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.embed_0 = nn.Embedding(embedding_dim=128, num_embeddings=10)

    def forward(self, x, q):
        x = self.embed_0(x)
        q = self.embed_0(q)
        return x, q

def test():
    net = Model().half().float()
    net.eval()

    torch.manual_seed(0)
    x = torch.randint(10, (13,), dtype=torch.int)
    q = torch.randint(10, (2, 13), dtype=torch.int)

    a = net(x, q)

    module = convert_and_import_ncnn(net, (x, q), "test_nn_Embedding", pnnx_args=("inputshape=[13]i32,[2,13]i32",))

    # ncnn inference
    b = module.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
