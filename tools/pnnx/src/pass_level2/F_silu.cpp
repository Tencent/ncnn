// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_silu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::silu              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.silu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_silu, 101)

class F_silu_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::sigmoid           op_0        1 1 input 166
aten::mul               op_1        2 1 input 166 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.silu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_silu_1, 100)

} // namespace pnnx
