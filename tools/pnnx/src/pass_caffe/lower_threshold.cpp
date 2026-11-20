// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_threshold.h"

#include "parameter.h"
#include "pass_level2.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_threshold(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input and one output, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    if (!has_known_shape(op->inputs[0]))
    {
        fprintf(stderr, "caffe layer %s (%s): input shape is unknown; provide inputshape\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    const float threshold = get_float(op, "threshold", 0.f);
    op->outputs[0]->shape = op->inputs[0]->shape;
    op->params.clear();
    op->params["threshold"] = threshold;
    op->outputs[0]->params = op->inputs[0]->params;
    return 0;
}

class lower_threshold_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Threshold         op_0            1 1 input out threshold=%threshold
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input           0 1 input
prim::Constant          threshold       0 1 threshold value=%threshold
prim::Constant          zero            0 1 zero value=0.0
aten::sub               sub             2 1 input threshold shifted
aten::sign              sign            1 1 shifted signed
aten::max               max             2 1 signed zero out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"threshold", "zero"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("sub")->inputs[0];
        for (const char* name : {"sub", "sign"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

void lower_threshold(Graph& graph)
{
    int opindex = 0;
    lower_threshold_pass a;
    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace caffe2pnnx

} // namespace pnnx
