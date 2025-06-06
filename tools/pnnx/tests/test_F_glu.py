# Copyright (c) 2022 Xiaomi Corp.        (author: Fangjun Kuang)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x0 = F.glu(x, dim=0)

        y0 = F.glu(y, dim=0)
        y1 = F.glu(y, dim=1)

        z0 = F.glu(z, dim=0)
        z1 = F.glu(z, dim=1)
        z2 = F.glu(z, dim=2)
        return x0, y0, y1, z0, z1, z2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(18)
    y = torch.rand(12, 16)
    z = torch.rand(24, 28, 34)

    x0, y0, y1, z0, z1, z2 = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_F_glu.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_F_glu.pt inputshape=[18],[12,16],[24,28,34]")

    # pnnx inference
    import test_F_glu_pnnx
    x0p, y0p, y1p, z0p, z1p, z2p = test_F_glu_pnnx.test_inference()

    return torch.equal(x0, x0p) and torch.equal(y0, y0p) and torch.equal(y1, y1p) \
            and torch.equal(z0, z0p) and torch.equal(z1, z1p) and torch.equal(z2, z2p)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
