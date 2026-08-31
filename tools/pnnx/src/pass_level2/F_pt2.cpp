// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// pt2 路径的形态归一分支(住这里,渐进铺开)。
//
// 默认值缺省形态已由离线静态表解决:torch.export 会省略等于默认值的实参,
// builder(load_pt2)按 src/aten_defaults_table.h 把省略实参补全为完整
// schema 形态,使 pt2 图与 torchscript 图同构,torch_cat / torch_flatten /
// torch_stack / F_linear / F_conv2d_1 等既有 ts 形态分支零改动直接消费。
// (2026-08-31 N3:此前内联补缺省值的 flatten 2-op、conv2d 5-op/6-op/7-op
// 四条 PT2 分支已随之删除。)
//
// 本文件留给"非默认值类"的 pt2 形态差异:torch.export 与 torchscript 对同一
// 算子形态不同的场景(分解形态、overload 变体、多输出结构等),按
// GraphRewriterPass 惯例每算子一条 match_pattern_graph() 匹配 aten 原形态。
// priority 取值参考:需先于消费 ts 形态的 pass(torch_* 60 / F_* 110-140)。

#include "pass_level2.h"

namespace pnnx {

} // namespace pnnx
