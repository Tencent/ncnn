// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_conv_transpose1d_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv_transpose1d      op_0        2 1 input weight out bias=None stride=%stride output_padding=%output_padding padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Deconvolution1D";
    }

    const char* name_str() const
    {
        return "deconv1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0};
        }

        op->params["0"] = weight_shape[1];
        op->params["1"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[0];
        op->params["18"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = 0;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2]);
        op->params["28"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv_transpose1d_4, 22)

class F_conv_transpose1d_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Input              bias        0 1 bias
F.conv_transpose1d      op_0        3 1 input weight bias out stride=%stride output_padding=%output_padding padding=%padding dilation=%dilation groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Deconvolution1D";
    }

    const char* name_str() const
    {
        return "deconv1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0};
        }

        op->params["0"] = weight_shape[1];
        op->params["1"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[0];
        op->params["18"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = 1;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2]);
        op->params["28"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv_transpose1d_5, 22)

class F_conv_transpose1d_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv_transpose1d      op_0        2 1 input weight out bias=None stride=%stride output_padding=%output_padding padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise1D";
    }

    const char* name_str() const
    {
        return "deconvdw1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0};
        }

        op->params["0"] = weight_shape[1] * captured_params.at("groups").i;
        op->params["1"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[0];
        op->params["18"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = 0;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2]);
        op->params["7"] = captured_params.at("groups");
        op->params["28"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv_transpose1d_6, 23)

class F_conv_transpose1d_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Input              bias        0 1 bias
F.conv_transpose1d      op_0        3 1 input weight bias out stride=%stride output_padding=%output_padding padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise1D";
    }

    const char* name_str() const
    {
        return "deconvdw1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> weight_shape = op->inputs[1]->shape;
        if (weight_shape.empty())
        {
            weight_shape = {0, 0, 0};
        }

        op->params["0"] = weight_shape[1] * captured_params.at("groups").i;
        op->params["1"] = weight_shape[2];
        op->params["2"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[0];
        op->params["18"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = 1;
        op->params["6"] = (int)(weight_shape[0] * weight_shape[1] * weight_shape[2]);
        op->params["7"] = captured_params.at("groups");
        op->params["28"] = 1; // dynamic weight
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_conv_transpose1d_7, 23)

} // namespace ncnn

} // namespace pnnx
