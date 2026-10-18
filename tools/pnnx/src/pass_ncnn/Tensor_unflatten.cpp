// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

class Tensor_unflatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.unflatten         op_0        1 1 input out dim=%dim sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "unflatten";
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
        int dim = captured_params.at("dim").i;
        const auto& sizes = captured_params.at("sizes").ai;
        auto shape = get_logical_shape_expr(op->inputs[0], 0);
        if (shape.empty())
        {
            fprintf(stderr, "reshape %s: unflatten input rank is unknown\n", op->name.c_str());
            return std::vector<std::string>();
        }

        if (dim < 0)
            dim += (int)shape.size();
        if (dim < 0 || dim >= (int)shape.size())
        {
            fprintf(stderr, "reshape %s: unflatten dim %d is out of range for input rank %d\n", op->name.c_str(), captured_params.at("dim").i, (int)shape.size());
            return std::vector<std::string>();
        }

        if (sizes.empty())
        {
            fprintf(stderr, "reshape %s: unflatten sizes must not be empty\n", op->name.c_str());
            return std::vector<std::string>();
        }

        std::vector<std::string> expanded;
        for (int size : sizes)
        {
            if (size != -1 && size <= 0)
            {
                fprintf(stderr, "reshape %s: unflatten sizes must be positive or inferred\n", op->name.c_str());
                return std::vector<std::string>();
            }
            expanded.push_back(std::to_string(size));
        }
        shape.erase(shape.begin() + dim);
        shape.insert(shape.begin() + dim, expanded.begin(), expanded.end());
        return shape;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_unflatten, 20)

} // namespace ncnn

} // namespace pnnx
