// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_permute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Permute";
    }

    const char* name_str() const
    {
        return "permute";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = 0;

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        const std::vector<int>& dims = captured_params.at("dims").ai;

        int input_rank = (int)op->inputs[0]->shape.size();

        if (input_rank == 0)
        {
            // assume input is fine
            input_rank = (int)dims.size();
        }

        if (batch_index >= 0 && batch_index < input_rank)
            input_rank -= 1;

        if (input_rank > 4)
        {
            fprintf(stderr, "permute %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        // drop permute batch index
        std::vector<int> new_dims;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] == batch_index)
                continue;

            int new_dim = dims[i] > batch_index ? dims[i] - 1 : dims[i];
            new_dims.push_back(new_dim);
        }

        if (input_rank != (int)new_dims.size())
        {
            fprintf(stderr, "permute %d-rank tensor with %d-rank dims is not possible\n", input_rank, (int)new_dims.size());
            return;
        }

        if (input_rank == 1)
        {
            // noop
            op->type = "Noop";
        }
        if (input_rank == 2)
        {
            if (new_dims == std::vector<int>{0, 1})
                op->type = "Noop";
            else if (new_dims == std::vector<int>{1, 0})
                op->params["0"] = 1;
        }
        if (input_rank == 3)
        {
            if (new_dims == std::vector<int>{0, 1, 2})
                op->type = "Noop";
            else if (new_dims == std::vector<int>{0, 2, 1})
                op->params["0"] = 1;
            else if (new_dims == std::vector<int>{1, 0, 2})
                op->params["0"] = 2;
            else if (new_dims == std::vector<int>{1, 2, 0})
                op->params["0"] = 3;
            else if (new_dims == std::vector<int>{2, 0, 1})
                op->params["0"] = 4;
            else if (new_dims == std::vector<int>{2, 1, 0})
                op->params["0"] = 5;
        }
        if (input_rank == 4)
        {
            if (new_dims == std::vector<int>{0, 1, 2, 3})
                op->type = "Noop";
            else if (new_dims == std::vector<int>{0, 1, 3, 2})
                op->params["0"] = 1;
            else if (new_dims == std::vector<int>{0, 2, 1, 3})
                op->params["0"] = 2;
            else if (new_dims == std::vector<int>{0, 2, 3, 1})
                op->params["0"] = 3;
            else if (new_dims == std::vector<int>{0, 3, 1, 2})
                op->params["0"] = 4;
            else if (new_dims == std::vector<int>{0, 3, 2, 1})
                op->params["0"] = 5;
            else if (new_dims == std::vector<int>{1, 0, 2, 3})
                op->params["0"] = 6;
            else if (new_dims == std::vector<int>{1, 0, 3, 2})
                op->params["0"] = 7;
            else if (new_dims == std::vector<int>{1, 2, 0, 3})
                op->params["0"] = 8;
            else if (new_dims == std::vector<int>{1, 2, 3, 0})
                op->params["0"] = 9;
            else if (new_dims == std::vector<int>{1, 3, 0, 2})
                op->params["0"] = 10;
            else if (new_dims == std::vector<int>{1, 3, 2, 0})
                op->params["0"] = 11;
            else if (new_dims == std::vector<int>{2, 0, 1, 3})
                op->params["0"] = 12;
            else if (new_dims == std::vector<int>{2, 0, 3, 1})
                op->params["0"] = 13;
            else if (new_dims == std::vector<int>{2, 1, 0, 3})
                op->params["0"] = 14;
            else if (new_dims == std::vector<int>{2, 1, 3, 0})
                op->params["0"] = 15;
            else if (new_dims == std::vector<int>{2, 3, 0, 1})
                op->params["0"] = 16;
            else if (new_dims == std::vector<int>{2, 3, 1, 0})
                op->params["0"] = 17;
            else if (new_dims == std::vector<int>{3, 0, 1, 2})
                op->params["0"] = 18;
            else if (new_dims == std::vector<int>{3, 0, 2, 1})
                op->params["0"] = 19;
            else if (new_dims == std::vector<int>{3, 1, 0, 2})
                op->params["0"] = 20;
            else if (new_dims == std::vector<int>{3, 1, 2, 0})
                op->params["0"] = 21;
            else if (new_dims == std::vector<int>{3, 2, 0, 1})
                op->params["0"] = 22;
            else if (new_dims == std::vector<int>{3, 2, 1, 0})
                op->params["0"] = 23;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_permute, 20)

} // namespace ncnn

} // namespace pnnx
