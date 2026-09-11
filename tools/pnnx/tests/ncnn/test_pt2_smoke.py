# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause


import torch
import torch.nn as nn
import torch.nn.functional as F

from testutil_pt2 import run_pt2_test


class Model(nn.Module):
    def forward(self, x, y):
        a = torch.flatten(x, start_dim=1)   # [1,3,4,4] -> [1,48]
        b = torch.flatten(y, start_dim=1)   # [1,3,4,4] -> [1,48]
        c = torch.cat((a, b), dim=1)        # dim=0 is reserved for the batch axis.
        return F.relu(c)


def test():
    net = Model().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 4)
    y = torch.rand(1, 3, 4, 4)
    return run_pt2_test(
        net,
        inputs=(x, y),
        inputshape_str="[1,3,4,4],[1,3,4,4]",
        base_name="test_pt2_smoke",
        atol=1e-4,
    )


if __name__ == "__main__":
    exit(0 if test() else 1)
