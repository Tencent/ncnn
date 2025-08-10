// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_dropout : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 p
pnnx.Input              input_2     0 1 training
aten::dropout           op_0        3 1 input p training out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.dropout";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_dropout, 100)

} // namespace pnnx
