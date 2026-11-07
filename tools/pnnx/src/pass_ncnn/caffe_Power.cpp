// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_Power : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.Power             op_0        1 1 input out power=%power scale=%scale shift=%shift
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Power";
    }

    const char* name_str() const
    {
        return "power";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("power");
        op->params["1"] = captured_params.at("scale");
        op->params["2"] = captured_params.at("shift");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_Power, 20)

} // namespace ncnn

} // namespace pnnx
