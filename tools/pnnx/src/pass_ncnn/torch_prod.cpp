// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_prod : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.prod              op_0        1 1 input out dim=%dim keepdim=%keepdim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "prod";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> new_dims;
        if (captured_params.at("dim").type == 2)
        {
            int dim = captured_params.at("dim").i;

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            if (dim == batch_index)
            {
                fprintf(stderr, "prod along batch axis is not supported\n");
                return;
            }

            int new_dim = dim > batch_index ? dim - 1 : dim;
            new_dims = std::vector<int>{new_dim};
        }
        else
        {
            const std::vector<int>& dims = captured_params.at("dim").ai;

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            // drop batch index
            for (int i = 0; i < (int)dims.size(); i++)
            {
                if (dims[i] == batch_index)
                    continue;

                int new_dim = dims[i] > batch_index ? dims[i] - 1 : dims[i];
                new_dims.push_back(new_dim);
            }
        }

        op->params["0"] = 6;
        op->params["1"] = 0;
        op->params["3"] = new_dims;
        op->params["4"] = captured_params.at("keepdim").b ? 1 : 0;
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_prod, 20)

class torch_prod_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.prod              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "prod";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 6;
        op->params["1"] = 1;
        op->params["4"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_prod_1, 20)

} // namespace ncnn

} // namespace pnnx
