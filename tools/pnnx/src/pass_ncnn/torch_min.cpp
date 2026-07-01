// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_min : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input       0 1 input
torch.min               op_0        1 2 input out indices dim=%dim keepdim=%keepdim
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "min";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim = captured_params.at("dim").i;

        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        int input_rank = op->inputs[0]->shape.size();
        if (input_rank == 0)
        {
            input_rank = op->outputs[0]->shape.size();
            if (!captured_params.at("keepdim").b && input_rank > 0)
                input_rank += 1;
        }
        if (dim < 0 && input_rank > 0)
            dim += input_rank;

        std::vector<int> dims;
        if (ncnn_batch_axis != 233 && dim == ncnn_batch_axis)
        {
            fprintf(stderr, "min along batch axis is not supported yet\n");
        }
        else
        {
            int new_dim = ncnn_batch_axis != 233 && dim > ncnn_batch_axis ? dim - 1 : dim;
            dims = std::vector<int>{new_dim};
        }

        op->params["0"] = 5;
        op->params["1"] = 0;
        op->params["3"] = dims;
        op->params["4"] = captured_params.at("keepdim").b ? 1 : 0;
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_min, 20)

class torch_min_0 : public torch_min
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.min               op_0        1 1 input out dim=%dim keepdim=%keepdim
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_min_0, 20)

class torch_min_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.min               op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "min";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
        if (ncnn_batch_axis != 233)
        {
            fprintf(stderr, "min along batch axis is not supported yet\n");
        }

        op->params["0"] = 5;
        op->params["1"] = 1;
        op->params["4"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_min_1, 20)

} // namespace ncnn

} // namespace pnnx
