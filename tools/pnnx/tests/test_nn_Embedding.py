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

    return torch.equal(a, b)


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
    for i, norm_type in enumerate([1.0, 2.0, 0.5]):
        net = RenormModel(norm_type).eval()
        mod = torch.jit.trace(net, x)
        name = "test_nn_Embedding_renorm_" + str(i)
        mod.save(name + ".pt")
        if os.system("../src/pnnx " + name + ".pt inputshape=[2]i32 fp16=0") != 0:
            return False
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, name + "_pnnx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = net(probe)
        actual = module.Model().eval()(probe)
        if not torch.allclose(expected, actual, 1e-6, 1e-6):
            return False
    return True

if __name__ == "__main__":
    if test() and test_renorm():
        exit(0)
    else:
        exit(1)
