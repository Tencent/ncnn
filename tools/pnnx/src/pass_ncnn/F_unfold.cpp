// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_unfold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.unfold                op_0        1 1 input out kernel_size=%kernel_size dilation=%dilation stride=%stride padding=%padding
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Unfold";
    }

    const char* name_str() const
    {
        return "unfold";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[1];
        op->params["14"] = captured_params.at("padding").ai[0];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_unfold, 20)

} // namespace ncnn

} // namespace pnnx
