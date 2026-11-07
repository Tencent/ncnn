// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_power.h"

#include "parameter.h"
#include "pass_level2.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_power(Operator* op)
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
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    const float power = get_float(op, "power", 1.f);
    const float scale = get_float(op, "scale", 1.f);
    const float shift = get_float(op, "shift", 0.f);
    op->params.clear();
    op->params["power"] = power;
    op->params["scale"] = scale;
    op->params["shift"] = shift;
    // fixed integer powers are rewritten after source shape propagation
    // other exponents retain the native Power path for negative-base powf
    if (power == 1.f || power == 2.f || power == 3.f)
        op->outputs[0]->params = op->inputs[0]->params;
    return 0;
}

class lower_power_1_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Power             op_0            1 1 input out power=%power scale=%scale shift=%shift
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input           0 1 input
prim::Constant          scale           0 1 scale value=%scale
prim::Constant          shift           0 1 shift value=%shift
aten::mul               mul             2 1 input scale scaled
aten::add               affine          2 1 scaled shift out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("power").f == 1.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"scale", "shift"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("mul")->inputs[0];
        for (const char* name : {"mul"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

class lower_power_2_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Power             op_0            1 1 input out power=%power scale=%scale shift=%shift
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input           0 1 input
prim::Constant          scale           0 1 scale value=%scale
prim::Constant          shift           0 1 shift value=%shift
aten::mul               mul             2 1 input scale scaled
aten::add               affine          2 1 scaled shift affine
aten::mul               square          2 1 affine affine out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("power").f == 2.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"scale", "shift"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("mul")->inputs[0];
        for (const char* name : {"mul", "affine"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

class lower_power_3_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Power             op_0            1 1 input out power=%power scale=%scale shift=%shift
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input           0 1 input
prim::Constant          scale           0 1 scale value=%scale
prim::Constant          shift           0 1 shift value=%shift
aten::mul               mul             2 1 input scale scaled
aten::add               affine          2 1 scaled shift affine
aten::mul               square          2 1 affine affine square
aten::mul               cube            2 1 square affine out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("power").f == 3.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"scale", "shift"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("mul")->inputs[0];
        for (const char* name : {"mul", "affine", "square"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

void lower_power(Graph& graph)
{
    int opindex = 0;
    lower_power_1_pass a;
    pnnx_graph_rewrite(graph, &a, opindex);
    lower_power_2_pass b;
    pnnx_graph_rewrite(graph, &b, opindex);
    lower_power_3_pass c;
    pnnx_graph_rewrite(graph, &c, opindex);
}

} // namespace caffe2pnnx

} // namespace pnnx
