// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

class Tensor_reshape_as : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
Tensor.reshape_as       op_0        2 1 input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "reshape_as";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");
        std::map<std::string, Parameter> params;
        return resolve_reshape_params(op, get_logical_shape_expr(op->inputs[1], 1), params);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        std::map<std::string, Parameter> params;
        if (resolve_reshape_params(op, get_logical_shape_expr(op->inputs[1], 1), params))
            write_reshape_params(op, params);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_reshape_as, 20)

} // namespace ncnn

} // namespace pnnx
