// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_rms_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.rms_norm              op_0        1 1 input out weight=None normalized_shape=%normalized_shape eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "RMSNorm";
    }

    const char* name_str() const
    {
        return "rmsn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& normalized_shape = captured_params.at("normalized_shape").ai;
        int affine_size = normalized_shape[0];
        for (size_t i = 1; i < normalized_shape.size(); i++)
        {
            affine_size *= normalized_shape[i];
        }

        const float eps = captured_params.at("eps").type == 0 ? 0.f : captured_params.at("eps").f;

        op->params["0"] = affine_size;
        op->params["1"] = eps;
        op->params["2"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_rms_norm, 20)

} // namespace ncnn

} // namespace pnnx
