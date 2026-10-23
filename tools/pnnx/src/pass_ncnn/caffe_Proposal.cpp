// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_Proposal : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 scores
pnnx.Input              input_1     0 1 deltas
pnnx.Input              input_2     0 1 info
caffe.Proposal          op_0        3 1 scores deltas info out feat_stride=%feat_stride return_scores=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Proposal";
    }

    const char* name_str() const
    {
        return "proposal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("feat_stride");
        op->params["1"] = 16;
        op->params["2"] = 6000;
        op->params["3"] = 300;
        op->params["4"] = 0.7f;
        op->params["5"] = 16;
        op->outputs[0]->shape = {-1, 1, 4};
        op->outputs[0]->params["__batch_index"] = 233;
        if (op->outputs.size() == 2)
        {
            op->outputs[1]->shape = {-1, 1, 1};
            op->outputs[1]->params["__batch_index"] = 233;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_Proposal, 20)

class caffe_Proposal_1 : public caffe_Proposal
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 scores
pnnx.Input              input_1     0 1 deltas
pnnx.Input              input_2     0 1 info
caffe.Proposal          op_0        3 2 scores deltas info out out_scores feat_stride=%feat_stride return_scores=True
pnnx.Output             output_0    1 0 out
pnnx.Output             output_1    1 0 out_scores
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_Proposal_1, 20)

} // namespace ncnn

} // namespace pnnx
