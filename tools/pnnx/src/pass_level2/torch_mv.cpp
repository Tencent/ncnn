// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_mv : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 vec
aten::mv                op_1        2 1 input vec output
pnnx.Output             output      1 0 output
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mv";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mv, 90)

} // namespace pnnx
