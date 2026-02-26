// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class F_hardtanh : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.hardtanh              op_0        1 1 input out min_val=%min_val max_val=%max_val
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Clip";
    }

    const char* name_str() const
    {
        return "htanh";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float min = -FLT_MAX;
        float max = FLT_MAX;

        if (captured_params.at("min_val").type == 2)
        {
            min = captured_params.at("min_val").i;
        }
        if (captured_params.at("min_val").type == 3)
        {
            min = captured_params.at("min_val").f;
        }

        if (captured_params.at("max_val").type == 2)
        {
            max = captured_params.at("max_val").i;
        }
        if (captured_params.at("max_val").type == 3)
        {
            max = captured_params.at("max_val").f;
        }

        op->params["0"] = min;
        op->params["1"] = max;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_hardtanh, 20)

} // namespace ncnn

} // namespace pnnx
