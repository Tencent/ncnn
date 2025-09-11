// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_roll : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shifts
pnnx.Input              input_2     0 1 dims
aten::roll              op_0        3 1 input shifts dims out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.roll";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_roll, 60)

class torch_roll_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shifts
aten::roll_shift_and_dim_onnx op_0  2 1 input shifts out dim=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.roll";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_roll_1, 60)

} // namespace pnnx
