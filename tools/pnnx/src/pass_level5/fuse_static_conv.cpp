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

#include "fuse_static_conv.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fconv1d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
F.conv1d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv1d";
    }

    const char* name_str() const
    {
        return "conv1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
        }

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = false;

        op->attrs["weight"] = weight;
    }
};

class fuse_static_Fconv1d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.conv1d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv1d";
    }

    const char* name_str() const
    {
        return "conv1d";
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

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

class fuse_static_Fconv2d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
F.conv2d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv2d";
    }

    const char* name_str() const
    {
        return "conv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
        }

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2], weight.shape[3]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = false;

        op->attrs["weight"] = weight;
    }
};

class fuse_static_Fconv2d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.conv2d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv2d";
    }

    const char* name_str() const
    {
        return "conv2d";
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

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2], weight.shape[3]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

class fuse_static_Fconv3d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
F.conv3d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv3d";
    }

    const char* name_str() const
    {
        return "conv3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
        }

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2], weight.shape[3], weight.shape[4]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = false;

        op->attrs["weight"] = weight;
    }
};

class fuse_static_Fconv3d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.conv3d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv3d";
    }

    const char* name_str() const
    {
        return "conv3d";
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

        op->params["in_channels"] = weight.shape[1] * captured_params.at("groups").i;
        op->params["out_channels"] = weight.shape[0];
        op->params["kernel_size"] = std::vector<int>{weight.shape[2], weight.shape[3], weight.shape[4]};
        op->params["padding_mode"] = std::string("zeros");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;
    }
};

void fuse_static_conv(Graph& graph)
{
    fuse_static_Fconv1d_pass a;
    fuse_static_Fconv1d_pass_2 b;
    fuse_static_Fconv2d_pass c;
    fuse_static_Fconv2d_pass_2 d;
    fuse_static_Fconv3d_pass e;
    fuse_static_Fconv3d_pass_2 f;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
}

} // namespace pnnx
