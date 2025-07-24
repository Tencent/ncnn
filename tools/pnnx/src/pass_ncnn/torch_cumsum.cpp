// Copyright 2021 Tencent
// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_cumsum : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.cumsum            op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CumulativeSum";
    }

    const char* name_str() const
    {
        return "cumsum";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int dim = captured_params.at("dim").i;

        op->params["0"] = dim;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_cumsum, 20)

} // namespace ncnn

} // namespace pnnx
