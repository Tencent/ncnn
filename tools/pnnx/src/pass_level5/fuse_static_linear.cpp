// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "fuse_static_linear.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Flinear_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_features,%in_features)f32
F.linear                op_0        2 1 input weight out bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Linear               linear      1 1 input out in_features=%in_features out_features=%out_features bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Flinear_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_features,%in_features)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_features)f32
F.linear                op_0        3 1 input weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Linear               linear      1 1 input out in_features=%in_features out_features=%out_features bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Flinear_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_features,%in_features)f32
pnnx.Attribute          op_bias     0 1 bias @data=(1,%out_features,1)f32
F.linear                op_0        2 1 input weight a
pnnx.Expression         op_1        2 1 a bias out expr=add(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Linear               linear      1 1 input out in_features=%in_features out_features=%out_features bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        // fix bias shape
        const int out_features = captured_params.at("out_features").i;
        ops.at("linear")->attrs.at("bias").shape = {out_features};
    }
};

void fuse_static_linear(Graph& graph)
{
    fuse_static_Flinear_pass_3 a3;

    fuse_static_Flinear_pass a;
    fuse_static_Flinear_pass_2 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a3, opindex);

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
