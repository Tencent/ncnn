// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_rmsnorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_rmsnorm_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 weight @data #weight=(%c)f32
pnnx.Expression         op_1        1 1 input sq expr=pow(@0,2)
torch.mean              op_2        1 1 sq sqmean dim=(-1) keepdim=True
pnnx.Expression         op_3        3 1 weight input sqmean out expr=mul(@0,mul(@1,rsqrt(add(@2,%eps))))
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.RMSNorm              rmsnorm     1 1 input out elementwise_affine=True eps=%eps normalized_shape=(%c) @weight=%op_0.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_rmsnorm_pass_1 : public fuse_rmsnorm_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 weight @data #weight=(%c)f32
pnnx.Expression         op_1        1 1 input sq expr=pow(@0,2.000000e+00)
torch.mean              op_2        1 1 sq sqmean dim=(-1) keepdim=True
pnnx.Expression         op_3        3 1 weight input sqmean out expr=mul(@0,mul(@1,reciprocal(sqrt(add(@2,%eps)))))
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_rmsnorm_pass_onnx : public fuse_rmsnorm_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 weight @data #weight=(%c)f32
pnnx.Expression         op_1        1 1 input sq expr=pow(@0,2.000000e+00)
torch.mean              op_2        1 1 sq sqmean dim=(-1) keepdim=True
pnnx.Expression         op_3        3 1 weight input sqmean out expr=mul(@0,mul(@1,div(1.000000e+00,sqrt(add(@2,%eps)))))
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_rmsnorm(Graph& graph)
{
    fuse_rmsnorm_pass a;
    fuse_rmsnorm_pass_1 a1;
    fuse_rmsnorm_pass_onnx b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
