// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class nn_quantized_FloatFunctional_cat : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 tensors
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 scale
pnnx.Input              input_3     0 1 zero_point
quantized::cat          op_0        4 1 tensors dim scale zero_point out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.quantized.cat";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_quantized_FloatFunctional_cat, 60)

} // namespace pnnx
