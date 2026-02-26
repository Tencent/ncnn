// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_repeat_interleave : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 repeats
pnnx.Input              input_2     0 1 dim
prim::Constant          op_0        0 1 output_size value=*
aten::repeat_interleave op_1        4 1 input repeats dim output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.repeat_interleave";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_repeat_interleave, 60)

class torch_repeat_interleave_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 repeats
pnnx.Input              input_2     0 1 dim
aten::repeat_interleave op_0        3 1 input repeats dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.repeat_interleave";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_repeat_interleave_1, 60)

} // namespace pnnx
