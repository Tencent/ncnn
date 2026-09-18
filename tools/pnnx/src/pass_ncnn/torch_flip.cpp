// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_flip : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.flip              op_0        1 1 input out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Flip";
    }

    const char* name_str() const
    {
        return "flip";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;

        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;
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
                fprintf(stderr, "flip along batch axis is not supported yet\n");
                continue;
            }

            int new_dim = ncnn_batch_axis != 233 && dim > ncnn_batch_axis ? dim - 1 : dim;
            new_dims.push_back(new_dim);
        }

        op->params["0"] = new_dims;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_flip, 20)

} // namespace ncnn

} // namespace pnnx
