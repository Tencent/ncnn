// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_rrelu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 lower
pnnx.Input              input_2     0 1 upper
prim::Constant          op_0        0 1 training value=False
prim::Constant          op_1        0 1 generator value=None
aten::rrelu             op_2        5 1 input lower upper training generator out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.rrelu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_rrelu, 100)

} // namespace pnnx
