// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_tril : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 diagonal value=%diagonal
aten::tril              op_1        2 1 input diagonal out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tril";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tril, 70)

} // namespace pnnx
