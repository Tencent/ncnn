// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_PriorBox : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 feature
pnnx.Input              input_1     0 1 image
caffe.PriorBox          op_0        2 1 feature image out min_sizes=%min_sizes max_sizes=%max_sizes aspect_ratios=%aspect_ratios variance=%variance clip=%clip image_h=%image_h image_w=%image_w step_h=%step_h step_w=%step_w offset=%offset
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "PriorBox";
    }

    const char* name_str() const
    {
        return "priorbox";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("min_sizes");
        op->params["1"] = captured_params.at("max_sizes");
        op->params["2"] = captured_params.at("aspect_ratios");
        const std::vector<float>& variance = captured_params.at("variance").af;
        for (int i = 0; i < 4; i++)
            op->params[std::to_string(3 + i)] = variance[i];
        op->params["7"] = 0;
        op->params["8"] = captured_params.at("clip").b ? 1 : 0;
        op->params["9"] = captured_params.at("image_w");
        op->params["10"] = captured_params.at("image_h");
        op->params["11"] = captured_params.at("step_w");
        op->params["12"] = captured_params.at("step_h");
        op->params["13"] = captured_params.at("offset");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_PriorBox, 20)

} // namespace ncnn

} // namespace pnnx
