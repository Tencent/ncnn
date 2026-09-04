# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# pt2 路径 smoke 测试：flatten + cat(dim=1 非默认) + relu
# 覆盖：M3 已验证的 flatten/relu，以及 M4 新增的 cat 张量列表输入（kind=8）。
# 注：cat(dim=0) 会撞 ncnn "cat along batch axis" 限制（batch_axis 硬编码为 0），
#     故只用 dim=1（非 batch 轴）验证 cat 路径；dim 兜底默认值由代码审查保证。

import torch
import torch.nn as nn
import torch.nn.functional as F

from testutil_pt2 import run_pt2_test


class Model(nn.Module):
    def forward(self, x, y):
        a = torch.flatten(x, start_dim=1)   # [1,3,4,4] -> [1,48]
        b = torch.flatten(y, start_dim=1)   # [1,3,4,4] -> [1,48]
        c = torch.cat((a, b), dim=1)        # [1,96]   dim=1 非默认，会显式进图
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
