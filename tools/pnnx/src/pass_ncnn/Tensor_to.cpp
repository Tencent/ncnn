// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_to : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.to               op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");
        return op->inputs[0]->type > 0 && op->inputs[0]->type == op->outputs[0]->type;
    }

    const char* type_str() const
    {
        // Mutations have been functionalized. ncnn owns tensor storage and
        // layout, so a same-dtype copy does not change its inference values.
        return "Noop";
    }

    const char* name_str() const
    {
        return "to";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_to, 20)

} // namespace ncnn

} // namespace pnnx
