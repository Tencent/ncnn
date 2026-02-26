// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_silu.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_silu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.sigmoid               op_0        1 1 input a
pnnx.Expression         op_1        2 1 input a out expr=mul(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.silu";
    }

    const char* name_str() const
    {
        return "silu";
    }
};

class fuse_silu_pass_1 : public fuse_silu_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
nn.Sigmoid              op_0        1 1 input a
pnnx.Expression         op_1        2 1 input a out expr=mul(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_silu(Graph& graph)
{
    fuse_silu_pass a;
    fuse_silu_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
