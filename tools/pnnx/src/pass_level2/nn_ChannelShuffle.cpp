// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class nn_ChannelShuffle : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0             0 1 input
prim::Constant          op_0                0 1 groups value=%groups
aten::channel_shuffle   op_1                2 1 input groups out
pnnx.Output             output              1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.ChannelShuffle";
    }

    const char* name_str() const
    {
        return "channelshuffle";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_ChannelShuffle, 20)

} // namespace pnnx
