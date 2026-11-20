// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class F_softplus : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.softplus              op_0        1 1 input out beta=1 threshold=%threshold
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const Parameter& threshold = captured_params.at("threshold");
        if (threshold.type == 2) return true;
        if (threshold.type == 3) return !(threshold.f < -FLT_MAX);
        return threshold.type == 4 && (threshold.s == "inf" || threshold.s == "nan" || threshold.s == "nan.0" || threshold.s == "-nan" || threshold.s == "-nan.0");
    }

    const char* type_str() const
    {
        return "Softplus";
    }

    const char* name_str() const
    {
        return "softplus";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const Parameter& threshold = captured_params.at("threshold");
        float value = FLT_MAX;
        if (threshold.type == 2) value = (float)threshold.i;
        if (threshold.type == 3 && threshold.f <= FLT_MAX) value = threshold.f;

        // Positive infinity and NaN thresholds never select the linear branch for finite inputs.
        op->params["0"] = value;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_softplus, 20)

} // namespace ncnn

} // namespace pnnx
