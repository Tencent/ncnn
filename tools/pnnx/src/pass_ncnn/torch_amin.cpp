// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_amin : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.amin              op_0        1 1 input out dim=%dim keepdim=%keepdim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "amin";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dim").ai;

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        // drop batch index
        std::vector<int> new_dims;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] == batch_index)
                continue;

            int new_dim = dims[i] > batch_index ? dims[i] - 1 : dims[i];
            new_dims.push_back(new_dim);
        }

        op->params["0"] = 5;
        op->params["1"] = 0;
        op->params["3"] = new_dims;
        op->params["4"] = captured_params.at("keepdim").b ? 1 : 0;
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_amin, 20)

} // namespace ncnn

} // namespace pnnx
