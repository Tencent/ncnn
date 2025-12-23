// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

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
        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        const int start_dim = captured_params.at("start_dim").i;
        if (start_dim == 0 || (start_dim == 1 && batch_index == 0))
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

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int start_dim = captured_params.at("start_dim").i;
        int end_dim = captured_params.at("end_dim").i;

        const int input_rank = op->inputs[0]->shape.size();

        if (start_dim < 0)
            start_dim += input_rank;

        if (end_dim < 0)
            end_dim += input_rank;

        if (input_rank <= start_dim || input_rank <= end_dim)
        {
            fprintf(stderr, "flatten %d to %d not possible for %d-rank tensor\n", start_dim, end_dim, input_rank);
            return;
        }

        std::vector<int> shape_flattened;
        for (int i = 0; i < start_dim; i++)
        {
            shape_flattened.push_back(op->inputs[0]->shape[i]);
        }
        shape_flattened.push_back(-1);
        for (int i = end_dim + 1; i < input_rank; i++)
        {
            shape_flattened.push_back(op->inputs[0]->shape[i]);
        }

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        std::vector<int> new_shape;
        for (int i = 0; i < (int)shape_flattened.size(); i++)
        {
            if (i == batch_index && shape_flattened[i] == 1)
                continue;

            new_shape.push_back(shape_flattened[i]);
        }

        if (new_shape.size() == 5 && batch_index == 233)
        {
            if (new_shape[0] == 1)
            {
                fprintf(stderr, "assume flatten 5-rank tensor has batch_index 0\n");
                new_shape.erase(new_shape.begin());
            }
        }

        const int shape_rank = (int)new_shape.size();

        if (shape_rank > 5)
        {
            fprintf(stderr, "reshape to %d-rank tensor is not supported yet!\n", shape_rank);
            return;
        }

        if (shape_rank == 1)
        {
            op->params["0"] = new_shape[0];
        }
        if (shape_rank == 2)
        {
            op->params["0"] = new_shape[1];
            op->params["1"] = new_shape[0];
        }
        if (shape_rank == 3)
        {
            op->params["0"] = new_shape[2];
            op->params["1"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
        if (shape_rank == 4)
        {
            op->params["0"] = new_shape[3];
            op->params["1"] = new_shape[2];
            op->params["11"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flatten_2, 21)

} // namespace ncnn

} // namespace pnnx
