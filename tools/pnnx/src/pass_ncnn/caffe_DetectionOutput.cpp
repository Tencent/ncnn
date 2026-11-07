// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_DetectionOutput : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 location
pnnx.Input              input_1     0 1 confidence
pnnx.Input              input_2     0 1 priorbox
caffe.DetectionOutput   op_0        3 1 location confidence priorbox out num_classes=%num_classes nms_threshold=%nms_threshold top_k=%top_k keep_top_k=%keep_top_k confidence_threshold=%confidence_threshold
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DetectionOutput";
    }

    const char* name_str() const
    {
        return "detectionoutput";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("num_classes");
        op->params["1"] = captured_params.at("nms_threshold");
        const int top_k = captured_params.at("top_k").i;
        const int keep_top_k = captured_params.at("keep_top_k").i;
        op->params["2"] = top_k < 0 ? INT_MAX : top_k;
        op->params["3"] = keep_top_k < 0 ? INT_MAX : keep_top_k;
        op->params["4"] = captured_params.at("confidence_threshold");
        // ncnn exposes label, score and coordinates; Caffe also stores the image id
        op->outputs[0]->shape = {-1, 6};
        op->outputs[0]->params["__batch_index"] = 233;
        op->outputs[0]->params["__ncnn_batch_axis"] = 233;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_DetectionOutput, 20)

} // namespace ncnn

} // namespace pnnx
