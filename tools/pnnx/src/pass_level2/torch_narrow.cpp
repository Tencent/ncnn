// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_narrow : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 start
pnnx.Input              input_3     0 1 length
aten::narrow            op_0        4 1 input dim start length out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.narrow";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_narrow, 60)

} // namespace pnnx
