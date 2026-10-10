// Copyright 2021 Tencent
// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_cumsum : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_1        0 1 dtype value=*
aten::cumsum            op_2        3 1 input dim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.cumsum";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_cumsum, 90)

} // namespace pnnx
