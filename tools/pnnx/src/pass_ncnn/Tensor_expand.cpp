// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_expand : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.expand           op_0        1 1 input out sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tile";
    }

    const char* name_str() const
    {
        return "expand";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& sizes = captured_params.at("sizes").ai;

        const int ncnn_batch_axis = op->outputs[0]->params["__ncnn_batch_axis"].i;

        std::vector<int> shape = op->inputs[0]->shape;
        if (shape.empty())
        {
            fprintf(stderr, "expand tensor with unknown input shape is not supported yet, fallback to repeat 1\n");
            shape = sizes;
        }

        const int input_rank = (int)shape.size();
        const int output_rank = (int)sizes.size();
        if (input_rank > output_rank)
        {
            fprintf(stderr, "expand %d-rank tensor to %d-rank tensor is not supported yet, fallback to repeat 1\n", input_rank, output_rank);
        }

        const int rank_offset = output_rank - input_rank;

        // drop sizes batch index
        std::vector<int> repeats;
        for (int i = 0; i < output_rank; i++)
        {
            const int shape_index = i - rank_offset;
            const int shape_dim = shape_index >= 0 && shape_index < input_rank ? shape[shape_index] : 1;

            if (i == ncnn_batch_axis)
            {
                if (sizes[i] != -1 && sizes[i] != shape_dim)
                {
                    fprintf(stderr, "expand tensor along batch index %d is not supported yet!\n", ncnn_batch_axis);
                }
                continue;
            }

            int repeat = 1;
            if (sizes[i] != -1 && shape_dim == 1)
            {
                repeat = sizes[i];
            }

            repeats.push_back(repeat);
        }

        if (repeats.size() == 5 && ncnn_batch_axis == 233)
        {
            if (repeats[0] == 1)
            {
                fprintf(stderr, "assume expand 5-rank tensor has batch_index 0\n");
                repeats.erase(repeats.begin());
            }
        }

        const int repeats_rank = (int)repeats.size();

        if (repeats_rank > 5)
        {
            fprintf(stderr, "expand to %d-rank tensor is not supported yet!\n", repeats_rank);
        }

        op->params["2"] = repeats;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_expand, 20)

} // namespace ncnn

} // namespace pnnx
