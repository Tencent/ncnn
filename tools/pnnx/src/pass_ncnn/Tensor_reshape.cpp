// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

class Tensor_reshape : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.reshape          op_0        1 1 input out shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "reshape";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");
        std::map<std::string, Parameter> params;
        return resolve_reshape_params(op, get_shape(op, captured_params), params);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::map<std::string, Parameter> params;
        if (resolve_reshape_params(op, get_shape(op, captured_params), params))
            write_reshape_params(op, params);
    }

    std::vector<std::string> get_shape(const Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const auto& dimensions = captured_params.at("shape").ai;
        std::vector<std::string> shape;
        for (int i = 0; i < (int)dimensions.size(); i++)
        {
            // layout conversions use zero to copy the corresponding logical dimension
            shape.push_back(dimensions[i] == 0 ? get_logical_dim_expr(op->inputs[0], 0, i) : std::to_string(dimensions[i]));
        }
        return shape;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_reshape, 20)

} // namespace ncnn

} // namespace pnnx
