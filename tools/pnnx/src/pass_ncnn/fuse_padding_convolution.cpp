// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_padding_convolution.h"

#include "pass_level2.h"

namespace pnnx {

namespace ncnn {

class fuse_padding_convolution_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Padding                 op_0        1 1 input a %*=%*
Convolution             op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Convolution";
    }

    const char* name_str() const
    {
        return "padconv";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // check type = CONSTANT
        if (captured_params.find("op_0.4") != captured_params.end())
        {
            if (captured_params.at("op_0.4").i != 0)
                return false;
        }

        // check per_channel_pad_data_size = 0
        if (captured_params.find("op_0.6") != captured_params.end())
        {
            if (captured_params.at("op_0.6").i != 0)
                return false;
        }

        // check front = 0
        if (captured_params.find("op_0.7") != captured_params.end())
        {
            if (captured_params.at("op_0.7").i != 0)
                return false;
        }

        // check behind = 0
        if (captured_params.find("op_0.8") != captured_params.end())
        {
            if (captured_params.at("op_0.8").i != 0)
                return false;
        }

        const int conv_pad_left = captured_params.find("op_1.4") != captured_params.end() ? captured_params.at("op_1.4").i : 0;
        const int conv_pad_top = captured_params.find("op_1.14") != captured_params.end() ? captured_params.at("op_1.14").i : conv_pad_left;
        const int conv_pad_right = captured_params.find("op_1.15") != captured_params.end() ? captured_params.at("op_1.15").i : conv_pad_left;
        const int conv_pad_bottom = captured_params.find("op_1.16") != captured_params.end() ? captured_params.at("op_1.16").i : conv_pad_top;
        if (conv_pad_left == 0 && conv_pad_top == 0 && conv_pad_right == 0 && conv_pad_bottom == 0)
            return true;

        // check padding value == convolution pad_value
        float padding_value = captured_params.find("op_0.5") != captured_params.end() ? captured_params.at("op_0.5").f : 0.f;
        float conv_pad_value = captured_params.find("op_1.18") != captured_params.end() ? captured_params.at("op_1.18").f : 0.f;
        if (padding_value != conv_pad_value)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_1.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_1.")
                op->attrs[akey.substr(5)] = ap;
        }

        // adjust pad params
        const int conv_pad_left = op->params.find("4") != op->params.end() ? op->params.at("4").i : 0;
        const int conv_pad_top = op->params.find("14") != op->params.end() ? op->params.at("14").i : conv_pad_left;
        const int conv_pad_right = op->params.find("15") != op->params.end() ? op->params.at("15").i : conv_pad_left;
        const int conv_pad_bottom = op->params.find("16") != op->params.end() ? op->params.at("16").i : conv_pad_top;

        const int pad_top = captured_params.find("op_0.0") != captured_params.end() ? captured_params.at("op_0.0").i : 0;
        const int pad_bottom = captured_params.find("op_0.1") != captured_params.end() ? captured_params.at("op_0.1").i : 0;
        const int pad_left = captured_params.find("op_0.2") != captured_params.end() ? captured_params.at("op_0.2").i : 0;
        const int pad_right = captured_params.find("op_0.3") != captured_params.end() ? captured_params.at("op_0.3").i : 0;

        const int new_conv_pad_left = conv_pad_left + pad_left;
        const int new_conv_pad_top = conv_pad_top + pad_top;
        const int new_conv_pad_right = conv_pad_right + pad_right;
        const int new_conv_pad_bottom = conv_pad_bottom + pad_bottom;

        if (new_conv_pad_left != conv_pad_left)
            op->params["4"] = new_conv_pad_left;
        if (new_conv_pad_top != new_conv_pad_left)
            op->params["14"] = new_conv_pad_top;
        if (new_conv_pad_right != new_conv_pad_left)
            op->params["15"] = new_conv_pad_right;
        if (new_conv_pad_bottom != new_conv_pad_top)
            op->params["16"] = new_conv_pad_bottom;

        float padding_value = captured_params.find("op_0.5") != captured_params.end() ? captured_params.at("op_0.5").f : 0.f;
        if (padding_value != 0.f)
            op->params["18"] = padding_value;
    }
};

void fuse_padding_convolution(Graph& graph)
{
    fuse_padding_convolution_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace ncnn

} // namespace pnnx
