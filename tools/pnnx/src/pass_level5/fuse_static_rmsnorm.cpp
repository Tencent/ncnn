// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_rmsnorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Frmsnorm_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data
F.rms_norm              op_0        2 1 input weight out normalized_shape=%normalized_shape eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.RMSNorm              rmsn        1 1 input out normalized_shape=%normalized_shape eps=%eps elementwise_affine=True @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_static_rmsnorm(Graph& graph)
{
    fuse_static_Frmsnorm_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
