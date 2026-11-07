// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_eltwise.h"

#include "parameter.h"
#include "pass_level2.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static Operand* scalar(Graph& graph, Operator* before, const std::string& suffix, float value)
{
    Operator* op = graph.new_operator_before("prim::Constant", before->name + suffix, before);
    op->params["value"] = value;
    Operand* output = graph.new_operand(op->name);
    output->producer = op;
    output->type = 1;
    op->outputs.push_back(output);
    return output;
}

static Operand* arithmetic(Graph& graph, Operator* before, const char* type, const std::string& suffix, const std::vector<Operand*>& inputs, const Operand* tensor)
{
    Operator* op = graph.new_operator_before(type, before->name + suffix, before);
    op->inputs = inputs;
    for (Operand* input : inputs)
        input->consumers.push_back(op);
    Operand* output = graph.new_operand(op->name);
    output->producer = op;
    output->type = 1;
    if (tensor)
    {
        output->shape = tensor->shape;
        output->type = tensor->type;
        output->params = tensor->params;
    }
    op->outputs.push_back(output);
    return output;
}

static void replace_arithmetic(Operator* op, const char* type, const std::vector<Operand*>& inputs)
{
    op->outputs[0]->shape = op->inputs[0]->shape;
    op->outputs[0]->type = op->inputs[0]->type;
    op->outputs[0]->params = op->inputs[0]->params;
    for (Operand* input : op->inputs)
        input->remove_consumer(op);
    op->type = type;
    op->params.clear();
    op->attrs.clear();
    op->inputs = inputs;
    for (Operand* input : inputs)
        input->consumers.push_back(op);
}

int prepare_eltwise(Graph& graph, Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1 || !has_known_shape(op->inputs[0]))
    {
        fprintf(stderr, "caffe layer %s (%s): Eltwise requires nonempty inputs, one output and known shape; inputs=%zu outputs=%zu input_rank=%zu (provide inputshape if unknown)\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    const int operation = get_int(op, "operation", 1);
    const std::vector<float> coeff = get_parameter(op, "coeff", std::vector<float>()).af;
    if (operation < 0 || operation > 2 || (!coeff.empty() && (operation != 1 || coeff.size() != op->inputs.size())))
    {
        fprintf(stderr, "caffe layer %s (%s): Eltwise operation must be 0/1/2; coefficients require SUM and one per input; operation=%d coefficients=%zu inputs=%zu\n", op->name.c_str(), op->type.c_str(), operation, coeff.size(), op->inputs.size());
        return -1;
    }
    for (Operand* input : op->inputs)
    {
        if (!has_known_shape(input))
        {
            fprintf(stderr, "caffe layer %s (%s): input %s shape is unknown; provide inputshape\n", op->name.c_str(), op->type.c_str(), input->name.c_str());
            return -1;
        }
        if (input->shape.size() != shape.size())
        {
            fprintf(stderr, "caffe layer %s (%s): input %s rank %zu differs from first input rank %zu; broadcasting is unsupported\n", op->name.c_str(), op->type.c_str(), input->name.c_str(), input->shape.size(), shape.size());
            return -1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (input->shape[i] != shape[i])
            {
                fprintf(stderr, "caffe layer %s (%s): input %s dimension %zu is %d, expected %d; broadcasting is unsupported\n", op->name.c_str(), op->type.c_str(), input->name.c_str(), i, input->shape[i], shape[i]);
                return -1;
            }
        }
    }
    if (op->inputs.size() == 2)
    {
        op->params.clear();
        op->params["operation"] = operation;
        op->params["coeff"] = coeff;
        op->outputs[0]->params = op->inputs[0]->params;
        return 0;
    }

    const char* type = "aten::add";
    if (operation == 0)
        type = "aten::mul";
    else if (operation == 2)
        type = "aten::maximum";

    const std::vector<Operand*> inputs = op->inputs;
    if (inputs.size() == 1)
    {
        Operand* coefficient = scalar(graph, op, "_coeff", coeff.empty() ? 1.f : coeff[0]);
        replace_arithmetic(op, "aten::mul", {inputs[0], coefficient});
        return 0;
    }

    // preserve the source input order and left-associated reduction
    Operand* accumulated = inputs[0];
    if (!coeff.empty())
    {
        Operand* coefficient = scalar(graph, op, "_coeff0", coeff[0]);
        accumulated = arithmetic(graph, op, "aten::mul", "_scaled0", {inputs[0], coefficient}, inputs[0]);
    }
    for (size_t i = 1; i < inputs.size(); i++)
    {
        const std::string suffix = std::to_string(i);
        Operand* other = inputs[i];
        if (!coeff.empty())
        {
            Operand* coefficient = scalar(graph, op, "_coeff" + suffix, coeff[i]);
            other = arithmetic(graph, op, "aten::mul", "_scaled" + suffix, {other, coefficient}, inputs[0]);
        }
        if (i + 1 == inputs.size())
            replace_arithmetic(op, type, {accumulated, other});
        else
            accumulated = arithmetic(graph, op, type, "_reduce" + suffix, {accumulated, other}, inputs[0]);
    }
    return 0;
}

class lower_eltwise_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input           0 1 input
pnnx.Input              other_input     0 1 other
caffe.Eltwise           op_0            2 1 input other out operation=%operation coeff=%coeff
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input           0 1 input
pnnx.Input              other_input     0 1 other
aten::add               binary          2 1 input other out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int operation = captured_params.at("operation").i;
        return operation >= 0 && operation <= 2 && captured_params.at("coeff").af.empty();
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        const int operation = captured_params.at("operation").i;
        if (operation == 0)
            ops.at("binary")->type = "aten::mul";
        else if (operation == 2)
            ops.at("binary")->type = "aten::maximum";
    }
};

class lower_eltwise_sum_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input           0 1 input
pnnx.Input              other_input     0 1 other
caffe.Eltwise           op_0            2 1 input other out operation=%operation coeff=%coeff
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input           0 1 input
pnnx.Input              other_input     0 1 other
prim::Constant          coeff0          0 1 coeff0 value=1.0
prim::Constant          coeff1          0 1 coeff1 value=1.0
aten::mul               mul0            2 1 input coeff0 scaled0
aten::mul               mul1            2 1 other coeff1 scaled1
aten::add               add             2 1 scaled0 scaled1 out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("operation").i == 1 && captured_params.at("coeff").af.size() == 2;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        const std::vector<float>& coeff = captured_params.at("coeff").af;
        ops.at("coeff0")->params["value"] = coeff[0];
        ops.at("coeff1")->params["value"] = coeff[1];
        for (const char* name : {"coeff0", "coeff1"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("mul0")->inputs[0];
        for (const char* name : {"mul0", "mul1"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

void lower_eltwise(Graph& graph)
{
    int opindex = 0;
    lower_eltwise_pass a;
    pnnx_graph_rewrite(graph, &a, opindex);
    lower_eltwise_sum_pass b;
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace caffe2pnnx

} // namespace pnnx
