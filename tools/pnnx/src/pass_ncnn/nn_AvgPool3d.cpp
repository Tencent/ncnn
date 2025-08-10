// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_AvgPool3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AvgPool3d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling3D";
    }

    const char* name_str() const
    {
        return "avgpool3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("divisor_override").type != 0)
        {
            fprintf(stderr, "unsupported avgpool3d divisor_override\n");
            return;
        }

        op->params["0"] = 1;
        op->params["1"] = captured_params.at("kernel_size").ai[2];
        op->params["11"] = captured_params.at("kernel_size").ai[1];
        op->params["21"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = captured_params.at("stride").ai[2];
        op->params["12"] = captured_params.at("stride").ai[1];
        op->params["22"] = captured_params.at("stride").ai[0];
        op->params["3"] = captured_params.at("padding").ai[2];
        op->params["13"] = captured_params.at("padding").ai[1];
        op->params["23"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("ceil_mode").b ? 0 : 1;
        op->params["6"] = captured_params.at("count_include_pad").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AvgPool3d, 20)

} // namespace ncnn

} // namespace pnnx
