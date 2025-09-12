// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_as_strided : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
pnnx.Input              input_2     0 1 stride
prim::Constant          op_0        0 1 storage_offset value=*
aten::as_strided        op_1        4 1 input size stride storage_offset out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.as_strided";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_as_strided, 60)

} // namespace pnnx
