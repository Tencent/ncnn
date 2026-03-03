// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_max_pool1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.max_pool1d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride dilation=(1) padding=%padding ceil_mode=%ceil_mode return_indices=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling1D";
    }

    const char* name_str() const
    {
        return "maxpool1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> stride;
        if (captured_params.at("stride").type == 0)
        {
            stride = captured_params.at("kernel_size").ai;
        }
        else
        {
            stride = captured_params.at("stride").ai;
        }

        op->params["0"] = 0;
        op->params["1"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = stride[0];
        op->params["3"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("ceil_mode").b ? 0 : 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_max_pool1d, 20)

class F_max_pool1d_1 : public F_max_pool1d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.max_pool1d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode return_indices=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_max_pool1d_1, 20)

} // namespace ncnn

} // namespace pnnx
