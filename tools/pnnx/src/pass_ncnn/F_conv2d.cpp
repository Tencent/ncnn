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

class F_conv2d_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv2d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Convolution";
    }

    const char* name_str() const
    {
        return "conv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0, 0};
        }

        op->params["0"] = weight_shape[0];
        op->params["1"] = weight_shape[3];
        op->params["11"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[1];
            op->params["14"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 0;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]);
        op->params["19"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv2d_4, 22)

class F_conv2d_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Input              bias        0 1 bias
F.conv2d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Convolution";
    }

    const char* name_str() const
    {
        return "conv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0, 0};
        }

        op->params["0"] = weight_shape[0];
        op->params["1"] = weight_shape[3];
        op->params["11"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[1];
            op->params["14"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 1;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]);
        op->params["19"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv2d_5, 22)

class F_conv2d_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv2d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdw2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0, 0};
        }

        op->params["0"] = weight_shape[0];
        op->params["1"] = weight_shape[3];
        op->params["11"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[1];
            op->params["14"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 0;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]);
        op->params["7"] = captured_params.at("groups");
        op->params["19"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv2d_6, 23)

class F_conv2d_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Input              bias        0 1 bias
F.conv2d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdw2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0, 0};
        }

        op->params["0"] = weight_shape[0];
        op->params["1"] = weight_shape[3];
        op->params["11"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        if (captured_params.at("padding").type == 4)
        {
            if (captured_params.at("padding").s == "same")
                op->params["4"] = -233;
            else if (captured_params.at("padding").s == "valid")
                op->params["4"] = 0;
        }
        else
        {
            op->params["4"] = captured_params.at("padding").ai[1];
            op->params["14"] = captured_params.at("padding").ai[0];
        }
        op->params["5"] = 1;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]);
        op->params["7"] = captured_params.at("groups");
        op->params["19"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv2d_7, 23)

} // namespace ncnn

} // namespace pnnx
