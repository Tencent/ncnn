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

#include "fuse_static_batchnorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fbatchnorm_pass_1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm1d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        size_t input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 2 || input_rank == 3;
    }
};

class fuse_static_Fbatchnorm_pass_1d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm1d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        size_t input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 2 || input_rank == 3;
    }
};

class fuse_static_Fbatchnorm_pass_2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps #input=(?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm2d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_2d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps #input=(?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm2d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
F.batchnorm             op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps #input=(?,?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm3d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=False @running_mean=%op_mean.data @running_var=%op_var.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Fbatchnorm_pass_3d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data=(%num_features)f32
pnnx.Attribute          op_var      0 1 running_var @data=(%num_features)f32
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps #input=(?,?,?,?,?)f32
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm3d          batchnorm   1 1 input out num_features=%num_features eps=%eps affine=True @running_mean=%op_mean.data @running_var=%op_var.data @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_static_batchnorm(Graph& graph)
{
    fuse_static_Fbatchnorm_pass_1d a;
    fuse_static_Fbatchnorm_pass_2d b;
    fuse_static_Fbatchnorm_pass_3d c;
    fuse_static_Fbatchnorm_pass_1d_1 a1;
    fuse_static_Fbatchnorm_pass_2d_1 b1;
    fuse_static_Fbatchnorm_pass_3d_1 c1;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &b1, opindex);
    pnnx_graph_rewrite(graph, &c1, opindex);
}

} // namespace pnnx
