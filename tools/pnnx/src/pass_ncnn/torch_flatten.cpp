// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"
#include "reshape_shape.h"

namespace pnnx {

namespace ncnn {

class torch_flatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.flatten           op_0        1 1 input out start_dim=%start_dim end_dim=%end_dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Flatten";
    }

    const char* name_str() const
    {
        return "flatten";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");
        const int input_rank = op->inputs[0]->shape.size();
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        if (op->outputs[0]->params["__ncnn_batch_axis"].i != ncnn_batch_axis)
            return false;

        const int start_dim = captured_params.at("start_dim").i;
        if ((start_dim == 0 && ncnn_batch_axis == 233) || (start_dim == 1 && ncnn_batch_axis == 0))
        {
            const int end_dim = captured_params.at("end_dim").i;
            if (end_dim == -1)
                return true;

            if (end_dim == input_rank - 1)
                return true;
        }

        return false;
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten, 20)

class torch_flatten_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.flatten           op_0        1 1 input out start_dim=%start_dim end_dim=%end_dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "flatten";
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
        int start = captured_params.at("start_dim").i;
        int end = captured_params.at("end_dim").i;
        auto shape = get_logical_shape_expr(op->inputs[0], 0);
        if (start < 0)
            start += (int)shape.size();
        if (end < 0)
            end += (int)shape.size();
        if (start < 0 || end < start || end >= (int)shape.size())
        {
            fprintf(stderr, "reshape %s: flatten input rank is unknown\n", op->name.c_str());
            return std::vector<std::string>();
        }

        const std::string merged = make_shape_product_expr(std::vector<std::string>(shape.begin() + start, shape.begin() + end + 1));
        shape.erase(shape.begin() + start, shape.begin() + end + 1);
        shape.insert(shape.begin() + start, merged);

        const auto& output_shape = op->outputs[0]->shape;
        if (output_shape.size() == shape.size())
        {
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (output_shape[i] > 0)
                    shape[i] = std::to_string(output_shape[i]);
            }
        }
        return shape;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten_2, 21)

} // namespace ncnn

} // namespace pnnx
