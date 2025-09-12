// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_cat : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 tensors
pnnx.Input              input_1     0 1 dim
aten::cat               op_0        2 1 tensors dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.cat";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_cat, 60)

class torch_cat_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 tensors
aten::cat               op_0        1 1 tensors out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.cat";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_cat_1, 60)

} // namespace pnnx
