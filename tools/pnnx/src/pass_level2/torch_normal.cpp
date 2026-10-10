// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_normal : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mean
pnnx.Input              input_2     0 1 std
pnnx.Input              input_3     0 1 generator
aten::normal            op_0        4 1 input mean std generator out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.normal";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_normal, 20)

} // namespace pnnx
