# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# pt2 (torch.export) dict / nested-pytree OUTPUT support.
# torchscript cannot preserve a dict return value, so this test only exercises
# the pt2 channel: torch.export flattens the dict output into flat graph
# outputs in pytree leaf order and pnnx returns them in the same order; the
# helper compares the flattened leaves against the original dict reference
# (see test_pnnx).

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        y = F.relu(x)
        return {'logits': y,
                'aux': {'sum': y.sum(),
                        'mean': (y.mean(), y.max())},
                'flag': x > 0}

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 4)

    # pt2 path (dict return not traceable by torchscript; skip automatically on
    # torch without export support)
    from pnnx_test_helper import test_pnnx
    pt2_ok = test_pnnx(net, (x,), ["[2,4]"], "test_pt2_dict_output")

    return pt2_ok is not False

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
