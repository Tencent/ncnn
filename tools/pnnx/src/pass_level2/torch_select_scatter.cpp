// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_select_scatter : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 src
pnnx.Input              input_2     0 1 dim
pnnx.Input              input_3     0 1 index
aten::select_scatter    op_0        4 1 input src dim index out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.select_scatter";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_select_scatter, 70)

} // namespace pnnx
