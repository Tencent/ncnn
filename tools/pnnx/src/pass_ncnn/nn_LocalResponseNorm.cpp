// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_LocalResponseNorm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LocalResponseNorm    op_0        1 1 input out size=%size alpha=%alpha beta=%beta k=%k
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "LRN";
    }

    const char* name_str() const
    {
        return "lrn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = 0; // region_type ACROSS_CHANNELS
        op->params["1"] = captured_params.at("size");
        op->params["2"] = captured_params.at("alpha");
        op->params["3"] = captured_params.at("beta");
        op->params["4"] = captured_params.at("k");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LocalResponseNorm, 20)

} // namespace ncnn

} // namespace pnnx
