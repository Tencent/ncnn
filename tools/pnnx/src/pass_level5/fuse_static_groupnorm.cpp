// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_groupnorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fgroupnorm_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%num_channels)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%num_channels)f32
F.group_norm            op_0        3 1 input weight bias out num_groups=%num_groups eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.GroupNorm            gn          1 1 input out num_channels=%num_channels num_groups=%num_groups eps=%eps affine=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_static_groupnorm(Graph& graph)
{
    fuse_static_Fgroupnorm_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
