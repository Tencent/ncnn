// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "pt2_schema.h"

#include <stdio.h>

namespace pnnx {

int load_pt2(const std::string& ptpath, Graph& g,
             const std::vector<std::vector<int64_t> >& /*input_shapes*/,
             const std::vector<std::string>& /*input_types*/)
{
    Pt2Program program;
    int ret = load_pt2_schema(ptpath, program);
    if (ret != 0)
        return ret;

    fprintf(stderr, "load_pt2: schema_version=%lld.%lld torch_version=%s nodes=%zu params=%zu\n",
            program.schema_version_major, program.schema_version_minor, program.torch_version.c_str(),
            program.nodes.size(), program.weights.size());

    // 忠实转写 builder:op 保持 aten 原名、节点参数全部 operand 化、
    // 权重 → pnnx.Attribute operand、零归一化(归一化在 pass_level2 PT2 形态分支)。
    // TODO(N2): 遍历 program.nodes 构建 pnnx_graph
    fprintf(stderr, "load_pt2: pt2 graph building not implemented yet\n");

    (void)g;
    return -1;
}

} // namespace pnnx
