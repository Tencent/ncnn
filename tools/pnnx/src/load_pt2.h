// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_H
#define PNNX_LOAD_PT2_H

#include "ir.h"
#include "load_pt2_parse.h"

#include <set>
#include <string>
#include <vector>

namespace pnnx {

// 把 Pt2Graph（规范 pnnx IR，方案2：Python 预处理已做 aten→pnnx 映射+形状推理+参数补全）
// 转换为 pnnx::Graph。emit 仅做纯建图（建 op/operand/param/attr），不做映射/形状/参数逻辑。
// 形状来自 Python 提取的 tensor_meta（已写入 JSON 的 output_shapes），不再需要 input_shapes。
int emit_pt2_graph(const Pt2Graph& g, Graph& pnnx_graph);

// 公开入口：直接把 .pt2 载入 pnnx::Graph（供 main.cpp 调用）。
int load_pt2(const std::string& pt2path, Graph& g,
             const std::string& device,
             const std::vector<std::vector<int64_t> >& input_shapes,
             const std::vector<std::string>& input_types,
             const std::vector<std::vector<char> >& input_contents,
             const std::vector<std::vector<int64_t> >& input_shapes2,
             const std::vector<std::string>& input_types2,
             const std::vector<std::vector<char> >& input_contents2,
             const std::vector<std::string>& customop_modules,
             const std::vector<std::string>& module_operators,
             const std::string& foldable_constants_zippath,
             std::set<std::string>& foldable_constants);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_H
