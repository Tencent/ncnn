// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.norm              op_0        1 1 input out dim=%dim keepdim=%keepdim p=%p
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float p = 0.f;
        if (captured_params.at("p").type == 2)
            p = (float)captured_params.at("p").i;
        if (captured_params.at("p").type == 3)
            p = captured_params.at("p").f;
        if (captured_params.at("p").type == 4 && captured_params.at("p").s == "fro")
            p = 2.f;

        if (p != 1.f && p != 2.f)
        {
            fprintf(stderr, "unsupported norm p=%f\n", p);
            return;
        }

        op->params["0"] = (p == 1.f) ? 7 : 8;

        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        if (captured_params.at("dim").type == 0)
        {
            if (ncnn_batch_axis != 233)
            {
                fprintf(stderr, "norm along batch axis is not supported yet\n");
            }

            op->params["1"] = 1;
        }
        else
        {
            const std::vector<int>& dims = captured_params.at("dim").ai;

            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                input_rank = op->outputs[0]->shape.size();

            // drop batch index
            std::vector<int> new_dims;
            for (int i = 0; i < (int)dims.size(); i++)
            {
                int dim = dims[i];
                if (dim < 0 && input_rank > 0)
                    dim += input_rank;

                if (ncnn_batch_axis != 233 && dim == ncnn_batch_axis)
                {
                    fprintf(stderr, "norm along batch axis is not supported yet\n");
                    continue;
                }

                int new_dim = ncnn_batch_axis != 233 && dim > ncnn_batch_axis ? dim - 1 : dim;
                new_dims.push_back(new_dim);
            }

            op->params["1"] = 0;
            op->params["3"] = new_dims;
        }

        op->params["4"] = captured_params.at("keepdim").b ? 1 : 0;
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_norm, 20)

} // namespace ncnn

} // namespace pnnx
