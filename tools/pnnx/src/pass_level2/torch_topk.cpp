// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_topk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 k
pnnx.Input              input_2     0 1 dim
pnnx.Input              input_3     0 1 largest
pnnx.Input              input_4     0 1 sorted
aten::topk              op_0        5 2 input k dim largest sorted values indices
pnnx.Output             output      2 0 values indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.topk";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_topk, 50)

} // namespace pnnx
