// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_gather : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 index
prim::Constant          op_0        0 1 sparse_grad value=*
aten::gather            op_1        4 1 input dim index sparse_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.gather";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_gather, 70)

} // namespace pnnx
