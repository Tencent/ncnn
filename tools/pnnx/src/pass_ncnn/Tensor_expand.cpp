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

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        const std::vector<int> shape = op->inputs[0]->shape;
        if (shape.empty())
        {
            fprintf(stderr, "expand tensor with unknown input shape is not supported yet!\n");
            return;
        }

        // drop sizes batch index
        std::vector<int> repeats;
        for (int i = 0; i < (int)sizes.size(); i++)
        {
            if (i == batch_index)
            {
                if (sizes[i] == 1)
                    continue;

                fprintf(stderr, "expand tensor along batch index %d is not supported yet!\n", batch_index);
            }

            int repeat = 1;
            if (sizes[i] != -1 && shape[i] == 1)
            {
                repeat = sizes[i];
            }

            repeats.push_back(repeat);
        }

        if (repeats.size() == 5 && batch_index == 233)
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
            return;
        }

        op->params["2"] = repeats;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_expand, 20)

} // namespace ncnn

} // namespace pnnx
