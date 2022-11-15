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
pnnx.Attribute          op_weight   0 1 weight @qwq
F.linear                op_0        2 1 input weight out bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    const char* name_str() const
    {
        return "linear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
        }

        op->params["in_features"] = weight.shape[1];
        op->params["out_features"] = weight.shape[0];
        op->params["bias"] = false;

        op->attrs["weight"] = weight;
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
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.linear                op_0        3 1 input weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    const char* name_str() const
    {
        return "linear";
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

        op->params["in_features"] = weight.shape[1];
        op->params["out_features"] = weight.shape[0];
        op->params["bias"] = true;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
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
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.linear                op_0        2 1 input weight a bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Expression         op_1        2 1 a bias out expr=%expr
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    const char* name_str() const
    {
        return "linear";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::string& expr = captured_params.at("expr").s;
        if (expr != "add(@0,@1)")
            return false;

        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        int out_channels = weight.shape[0];
        if (bias.shape != std::vector<int>{1, out_channels, 1})
            return false;

        return true;
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

        op->params["in_features"] = weight.shape[1];
        op->params["out_features"] = weight.shape[0];
        op->params["bias"] = true;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
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
