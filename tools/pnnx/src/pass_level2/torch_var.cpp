// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_var : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 unbiased
pnnx.Input              input_3     0 1 keepdim
aten::var               op_0        4 1 input dim unbiased keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.var";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_var, 50)

} // namespace pnnx
