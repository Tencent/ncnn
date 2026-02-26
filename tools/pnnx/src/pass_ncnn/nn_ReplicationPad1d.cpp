// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_ReplicationPad1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ReplicationPad1d     op_0        1 1 input out padding=%padding
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Padding";
    }

    const char* name_str() const
    {
        return "replicatepad1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& padding = captured_params.at("padding").ai;
        op->params["0"] = 0;
        op->params["1"] = 0;
        op->params["2"] = padding[0];
        op->params["3"] = padding[1];
        op->params["4"] = 1; // type
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_ReplicationPad1d, 20)

} // namespace ncnn

} // namespace pnnx
