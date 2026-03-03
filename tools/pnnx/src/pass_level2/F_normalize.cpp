// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_normalize : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
torch.norm              op_0        1 1 input 9 p=%p dim=(%dim) keepdim=True
prim::Constant          op_1        0 1 eps value=%eps
aten::clamp_min         op_2        2 1 9 eps 11
Tensor.expand_as        op_3        2 1 11 input denorm
aten::div               op_4        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.normalize";
    }
};

class F_normalize_2 : public F_normalize
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
torch.norm              op_0        1 1 input 9 p=%p dim=(%dim) keepdim=True
torch.clamp             op_1        1 1 9 11 max=None min=%eps
Tensor.expand           op_2        1 1 11 denorm sizes=*
aten::div               op_3        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_normalize_3 : public F_normalize
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
torch.norm              op_0        1 1 input 3 p=%p dim=%dim keepdim=True
torch.clamp             op_1        1 1 3 4 max=None min=%eps
Tensor.expand           op_2        1 1 4 denorm sizes=*
aten::div               op_3        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_normalize_dims : public F_normalize
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
torch.norm              op_0        1 1 input 9 p=%p dim=%dim keepdim=True
prim::Constant          op_1        0 1 eps value=%eps
aten::clamp_min         op_2        2 1 9 eps 11
Tensor.expand_as        op_3        2 1 11 input denorm
aten::div               op_4        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize, 130)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize_2, 130)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize_dims, 131)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize_3, 131)

} // namespace pnnx
