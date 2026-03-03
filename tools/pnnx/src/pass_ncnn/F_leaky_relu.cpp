// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_leaky_relu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.leaky_relu            op_0        1 1 input out negative_slope=%negative_slope
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ReLU";
    }

    const char* name_str() const
    {
        return "leakyrelu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float negative_slope = 0.f;

        if (captured_params.at("negative_slope").type == 2)
        {
            negative_slope = captured_params.at("negative_slope").i;
        }
        if (captured_params.at("negative_slope").type == 3)
        {
            negative_slope = captured_params.at("negative_slope").f;
        }

        op->params["0"] = negative_slope;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_leaky_relu, 20)

} // namespace ncnn

} // namespace pnnx
