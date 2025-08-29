// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_masked_select : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask
aten::masked_select     op_0        2 1 input mask out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.masked_select";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_masked_select, 70)

} // namespace pnnx
