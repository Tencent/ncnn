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

#include "fuse_static_instancenorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Finstancenorm_pass_1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.instance_norm         op_0        3 1 input weight bias out running_mean=None running_var=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.InstanceNorm1d";
    }

    const char* name_str() const
    {
        return "instance_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators) const
    {
        int input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 2 || input_rank == 3;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        op->params["num_features"] = weight.shape[0];
        op->params["eps"] = captured_params.at("eps");
        op->params["affine"] = true;
        op->params["track_running_stats"] = false;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

class fuse_static_Finstancenorm_pass_2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.instance_norm         op_0        3 1 input weight bias out running_mean=None running_var=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.InstanceNorm1d";
    }

    const char* name_str() const
    {
        return "instance_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators) const
    {
        int input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 4;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        op->params["num_features"] = weight.shape[0];
        op->params["eps"] = captured_params.at("eps");
        op->params["affine"] = true;
        op->params["track_running_stats"] = false;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

class fuse_static_Finstancenorm_pass_3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.instance_norm         op_0        3 1 input weight bias out running_mean=None running_var=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.InstanceNorm1d";
    }

    const char* name_str() const
    {
        return "instance_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators) const
    {
        int input_rank = matched_operators.at("op_0")->inputs[0]->shape.size();
        return input_rank == 5;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        op->params["num_features"] = weight.shape[0];
        op->params["eps"] = captured_params.at("eps");
        op->params["affine"] = true;
        op->params["track_running_stats"] = false;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

void fuse_static_instancenorm(Graph& graph)
{
    fuse_static_Finstancenorm_pass_1d a;
    fuse_static_Finstancenorm_pass_2d b;
    fuse_static_Finstancenorm_pass_3d c;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
}

} // namespace pnnx
