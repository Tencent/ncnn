// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_ConstantPad3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConstantPad3d        op_0        1 1 input out padding=%padding value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Padding";
    }

    const char* name_str() const
    {
        return "constpad3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float pad_value = 0.f;
        if (captured_params.at("value").type == 2)
            pad_value = captured_params.at("value").i;
        if (captured_params.at("value").type == 3)
            pad_value = captured_params.at("value").f;

        op->params["0"] = captured_params.at("padding").ai[2];
        op->params["1"] = captured_params.at("padding").ai[3];
        op->params["2"] = captured_params.at("padding").ai[0];
        op->params["3"] = captured_params.at("padding").ai[1];
        op->params["4"] = 0;
        op->params["5"] = pad_value;
        op->params["6"] = 0; // per_channel_pad_data_size
        op->params["7"] = captured_params.at("padding").ai[4];
        op->params["8"] = captured_params.at("padding").ai[5];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_ConstantPad3d, 20)

} // namespace ncnn

} // namespace pnnx
