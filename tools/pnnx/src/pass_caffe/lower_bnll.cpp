// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_bnll.h"

#include "parameter.h"
#include "pass_level2.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_bnll(Operator* op)
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
    op->outputs[0]->shape = op->inputs[0]->shape;
    op->outputs[0]->params = op->inputs[0]->params;
    op->params.clear();
    return 0;
}

class lower_bnll_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.BNLL              op_0            1 1 input out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input           0 1 input
prim::Constant          zero            0 1 zero value=0.0
prim::Constant          one             0 1 one value=1.0
aten::max               max             2 1 input zero positive
aten::abs               abs             1 1 input magnitude
aten::neg               neg             1 1 magnitude negative
aten::exp               exp             1 1 negative exponential
aten::add               add_one         2 1 one exponential sum
aten::log               log             1 1 sum logarithm
aten::add               add             2 1 positive logarithm out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"zero", "one"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("max")->inputs[0];
        for (const char* name : {"max", "abs", "neg", "exp", "add_one", "log"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

void lower_bnll(Graph& graph)
{
    int opindex = 0;
    lower_bnll_pass a;
    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace caffe2pnnx

} // namespace pnnx
