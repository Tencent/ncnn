// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_conv3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
F.conv3d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Convolution3D";
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

        op->params["0"] = weight.shape[0];
        op->params["1"] = weight.shape[4];
        op->params["11"] = weight.shape[3];
        op->params["21"] = weight.shape[2];
        op->params["2"] = captured_params.at("dilation").ai[2];
        op->params["12"] = captured_params.at("dilation").ai[1];
        op->params["22"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[2];
        op->params["13"] = captured_params.at("stride").ai[1];
        op->params["23"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[2];
            op->params["14"] = captured_params.at("padding").ai[1];
            op->params["24"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 0;
        op->params["6"] = (int)(weight.data.size() / sizeof(float));

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = weight;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv3d, 20)

class F_conv3d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.conv3d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Convolution3D";
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

        op->params["0"] = weight.shape[0];
        op->params["1"] = weight.shape[4];
        op->params["11"] = weight.shape[3];
        op->params["21"] = weight.shape[2];
        op->params["2"] = captured_params.at("dilation").ai[2];
        op->params["12"] = captured_params.at("dilation").ai[1];
        op->params["22"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[2];
        op->params["13"] = captured_params.at("stride").ai[1];
        op->params["23"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[2];
            op->params["14"] = captured_params.at("padding").ai[1];
            op->params["24"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 1;
        op->params["6"] = (int)(weight.data.size() / sizeof(float));

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = weight;
        op->attrs["2"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv3d_1, 20)

class F_conv3d_2 : public GraphRewriterPass
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
        return "ConvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "convdw3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
        }

        op->params["0"] = weight.shape[0];
        op->params["1"] = weight.shape[4];
        op->params["11"] = weight.shape[3];
        op->params["21"] = weight.shape[2];
        op->params["2"] = captured_params.at("dilation").ai[2];
        op->params["12"] = captured_params.at("dilation").ai[1];
        op->params["22"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[2];
        op->params["13"] = captured_params.at("stride").ai[1];
        op->params["23"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[2];
            op->params["14"] = captured_params.at("padding").ai[1];
            op->params["24"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 0;
        op->params["6"] = (int)(weight.data.size() / sizeof(float));
        op->params["7"] = captured_params.at("groups");

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = weight;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv3d_2, 21)

class F_conv3d_3 : public GraphRewriterPass
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
        return "ConvolutionDepthWise3D";
    }

    const char* name_str() const
    {
        return "convdw3d";
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

        op->params["0"] = weight.shape[0];
        op->params["1"] = weight.shape[4];
        op->params["11"] = weight.shape[3];
        op->params["21"] = weight.shape[2];
        op->params["2"] = captured_params.at("dilation").ai[2];
        op->params["12"] = captured_params.at("dilation").ai[1];
        op->params["22"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[2];
        op->params["13"] = captured_params.at("stride").ai[1];
        op->params["23"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[2];
            op->params["14"] = captured_params.at("padding").ai[1];
            op->params["24"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 1;
        op->params["6"] = (int)(weight.data.size() / sizeof(float));
        op->params["7"] = captured_params.at("groups");

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = weight;
        op->attrs["2"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv3d_3, 21)

} // namespace ncnn

} // namespace pnnx
