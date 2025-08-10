// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_where : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 condition
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 other
aten::where             op_0        3 1 condition input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.where";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_where, 70)

} // namespace pnnx
