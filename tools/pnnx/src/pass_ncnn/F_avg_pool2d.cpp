// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_avg_pool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.avg_pool2d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling";
    }

    const char* name_str() const
    {
        return "avgpool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("divisor_override").type != 0)
        {
            fprintf(stderr, "unsupported avgpool2d divisor_override\n");
            return;
        }

        std::vector<int> stride;
        if (captured_params.at("stride").type == 0)
        {
            stride = captured_params.at("kernel_size").ai;
        }
        else
        {
            stride = captured_params.at("stride").ai;
        }

        op->params["0"] = 1;
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = stride[1];
        op->params["12"] = stride[0];
        op->params["3"] = captured_params.at("padding").ai[1];
        op->params["13"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("ceil_mode").b ? 0 : 1;
        op->params["6"] = captured_params.at("count_include_pad").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_avg_pool2d, 20)

} // namespace ncnn

} // namespace pnnx
