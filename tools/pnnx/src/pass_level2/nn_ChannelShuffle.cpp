// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class nn_ChannelShuffle_caffe : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.ShuffleChannel    op_0        1 1 input out groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.ChannelShuffle";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_ChannelShuffle_caffe, 120)

} // namespace pnnx
