// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_masked_fill : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask
Tensor.masked_fill      op_0        2 1 input mask out value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 mask
pnnx.Input              input_1     0 1 input
Where                    op_0        2 1 mask input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "masked_fill";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float value = captured_params.at("value").type == 3 ? captured_params.at("value").f : (float)captured_params.at("value").i;
        op->params["0"] = 1;
        op->params["1"] = value;
        op->params["2"] = 0;
        op->params["3"] = 0.f;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_masked_fill, 20)

} // namespace ncnn

} // namespace pnnx
