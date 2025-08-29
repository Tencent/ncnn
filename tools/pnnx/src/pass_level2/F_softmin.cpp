// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_softmin : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::neg               op_0        1 1 input 6
F.softmax               op_1        1 1 6 out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmin";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmin, 102)

} // namespace pnnx
