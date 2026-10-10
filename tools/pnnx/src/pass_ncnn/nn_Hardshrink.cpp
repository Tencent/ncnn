// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_Hardshrink : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Hardshrink           op_0        1 1 input out lambd=%lambd
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Shrink";
    }

    const char* name_str() const
    {
        return "hshrink";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float lambd = 0.5f;
        if (captured_params.at("lambd").type == 2)
        {
            lambd = captured_params.at("lambd").i;
        }
        if (captured_params.at("lambd").type == 3)
        {
            lambd = captured_params.at("lambd").f;
        }

        op->params["0"] = 0.f;
        op->params["1"] = lambd;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Hardshrink, 20)

} // namespace ncnn

} // namespace pnnx
