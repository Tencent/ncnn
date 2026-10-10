// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 output_size
pnnx.Input              input_2     0 1 kernel_size
pnnx.Input              input_3     0 1 dilation
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 stride
aten::col2im            op_0        6 1 input output_size kernel_size dilation padding stride out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.fold";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_fold, 110)

} // namespace pnnx
