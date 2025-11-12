// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_repeat : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.repeat           op_0        1 1 input out sizes=%sizes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tile";
    }

    const char* name_str() const
    {
        return "repeat";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& sizes = captured_params.at("sizes").ai;

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        // drop sizes batch index
        std::vector<int> new_sizes;
        for (int i = 0; i < (int)sizes.size(); i++)
        {
            if (i == batch_index)
            {
                if (sizes[i] == 1)
                    continue;

                fprintf(stderr, "repeat tensor along batch index %d is not supported yet!\n", batch_index);
            }

            new_sizes.push_back(sizes[i]);
        }

        if (new_sizes.size() == 5 && batch_index == 233)
        {
            if (new_sizes[0] == 1)
            {
                fprintf(stderr, "assume repeat 5-rank tensor has batch_index 0\n");
                new_sizes.erase(new_sizes.begin());
            }
        }

        const int sizes_rank = (int)new_sizes.size();

        if (sizes_rank > 5)
        {
            fprintf(stderr, "repeat to %d-rank tensor is not supported yet!\n", sizes_rank);
            return;
        }

        op->params["2"] = new_sizes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_repeat, 20)

} // namespace ncnn

} // namespace pnnx
