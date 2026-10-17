# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # export torchscript
    mod = torch.jit.trace(net, (x, q))
    mod.save("test_nn_Embedding.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_Embedding.pt inputshape=[13]i32,[2,13]i32")

    # ncnn inference
    import test_nn_Embedding_ncnn
    b = test_nn_Embedding_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True


class RenormModel(nn.Module):
    def __init__(self, norm_type):
        super(RenormModel, self).__init__()
        self.embed = nn.Embedding(6, 3, max_norm=1.0, norm_type=norm_type)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(1, 19, dtype=torch.float32).reshape(6, 3))

    def forward(self, x):
        return self.embed(x)

def test_renorm():
    import os
    x = torch.tensor([0, 0], dtype=torch.int)
    probe = torch.tensor([4, 5], dtype=torch.int)
    for i, norm_type in enumerate([1.0, 2.0, 0.5, float("inf"), float("-inf"), float("nan")]):
        net = RenormModel(norm_type).eval()
        mod = torch.jit.trace(net, x)
        name = "test_nn_Embedding_renorm_" + str(i)
        mod.save(name + ".pt")
        if os.system("../../src/pnnx " + name + ".pt inputshape=[2]i32 fp16=0") != 0:
            return False
        import ncnn
        with open(name + ".ncnn.param") as f:
            layers = [line.split()[0] for line in f.readlines()[2:] if line.strip()]
        if "nn.Embedding" not in layers or "Embed" in layers:
            return False
        with ncnn.Net() as runtime:
            if runtime.load_param(name + ".ncnn.param") == 0:
                return False
    return True

if __name__ == "__main__":
    if test() and test_renorm():
        exit(0)
    else:
        exit(1)
