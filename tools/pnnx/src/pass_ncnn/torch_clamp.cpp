// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class torch_clamp : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.clamp             op_0        1 1 input out min=%min max=%max
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Clip";
    }

    const char* name_str() const
    {
        return "clamp";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float min = -FLT_MAX;
        float max = FLT_MAX;

        if (captured_params.at("min").type == 2)
        {
            min = (float)captured_params.at("min").i;
        }
        if (captured_params.at("min").type == 3)
        {
            min = captured_params.at("min").f;
        }

        if (captured_params.at("max").type == 2)
        {
            max = (float)captured_params.at("max").i;
        }
        if (captured_params.at("max").type == 3)
        {
            max = captured_params.at("max").f;
        }

        op->params["0"] = min;
        op->params["1"] = max;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_clamp, 20)

} // namespace ncnn

} // namespace pnnx
