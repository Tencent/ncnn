# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# pt2 (torch.export) dict / nested-pytree INPUT support.
# torchscript cannot trace a dict module argument, so this test only exercises
# the pt2 channel: torch.export flattens the dict argument into user_inputs in
# pytree leaf order and pnnx converts the flattened graph (see test_pnnx).

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, cfg):
        a0, a1 = cfg['a']            # tuple value
        b0, b1 = cfg['b']            # list value
        d = cfg['c']['d']            # nested dict value
        y0 = x + a0 + b0 + d
        y1 = x * a1 - b1
        y2 = x + d
        return y0, (y1, y2)          # nested tuple output

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 4)
    cfg = {'a': (torch.rand(2, 4), torch.rand(2, 4)),
           'b': [torch.rand(2, 4), torch.rand(2, 4)],
           'c': {'d': torch.rand(2, 4)}}

    # pt2 path (dict arg not traceable by torchscript; skip automatically on
    # torch without export support)
    from pnnx_test_helper import test_pnnx
    # pytree leaf order of (x, cfg): x, a0, a1, b0, b1, d
    pt2_ok = test_pnnx(net, (x, cfg), ["[2,4]", "[2,4]", "[2,4]", "[2,4]", "[2,4]", "[2,4]"], "test_pt2_dict_input")

    return pt2_ok is not False

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
