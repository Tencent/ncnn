// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

// dynamo keeps the aten::_weight_norm composite op (torchscript expands the
// parametrization). Expand it into weight = v * (g / ||v||_2), i.e.
// linalg_vector_norm + div + mul.
class torch_weight_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              v           0 1 v
pnnx.Input              g           0 1 g
prim::Constant          op_dim      0 1 dim value=%dim
aten::_weight_norm      op_0        3 1 v g dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              v           0 1 v
pnnx.Input              g           0 1 g
prim::Constant          op_dim      0 1 dim value=%dim
prim::Constant          op_ord      0 1 ord value=2
prim::Constant          op_kd       0 1 keepdim value=True
aten::linalg_vector_norm norm       4 1 v ord dim keepdim norm_out
aten::div               scale       2 1 g norm_out scale_out
aten::mul               mul_out     2 1 v scale_out out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_weight_norm, 140)

} // namespace pnnx
