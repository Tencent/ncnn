// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_LOG_SOFTMAX_H
#define PNNX_PASS_NCNN_LOG_SOFTMAX_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class LogSoftmaxPass : public GraphRewriterPass
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.logsumexp         logsumexp   1 1 input logsumexp dim=%dim keepdim=True
BinaryOp                sub         2 1 input logsumexp out 0=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* logsumexp = ops.at("logsumexp");
        logsumexp->params["dim"] = std::vector<int>(1, captured_params.at("dim").i);

        const Operand* input = logsumexp->inputs[0];
        Operand* reduced = logsumexp->outputs[0];
        reduced->type = input->type;
        reduced->shape = input->shape;
        reduced->params = input->params;

        int dim = captured_params.at("dim").i;
        if (dim < 0)
            dim += (int)reduced->shape.size();
        if (dim >= 0 && dim < (int)reduced->shape.size())
            reduced->shape[dim] = 1;
    }
};

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_LOG_SOFTMAX_H
