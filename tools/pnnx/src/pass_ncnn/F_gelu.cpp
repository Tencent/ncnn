// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_gelu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input           input          0 1 input
F.gelu               op_0           1 1 input out
pnnx.Output          output         1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GELU";
    }

    const char* name_str() const
    {
        return "gelu";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_gelu, 20)

class F_gelu_1 : public F_gelu
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input           input          0 1 input
F.gelu               op_0           1 1 input out approximate=%approximate
pnnx.Output          output         1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("approximate").s == "tanh")
            op->params["0"] = 1; // fast_gelu
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_gelu_1, 20)

} // namespace ncnn

} // namespace pnnx
