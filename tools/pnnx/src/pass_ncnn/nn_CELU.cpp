// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_CELU : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.CELU                 op_0        1 1 input out alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CELU";
    }

    const char* name_str() const
    {
        return "celu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 1.f;

        if (captured_params.at("alpha").type == 2)
        {
            alpha = (float)captured_params.at("alpha").i;
        }
        if (captured_params.at("alpha").type == 3)
        {
            alpha = captured_params.at("alpha").f;
        }

        op->params["0"] = alpha;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_CELU, 20)

} // namespace ncnn

} // namespace pnnx
