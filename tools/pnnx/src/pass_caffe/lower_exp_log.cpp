// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_exp_log.h"

#include "parameter.h"
#include "pass_level2.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_exp_log(Operator* op)
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
    const bool exp = op->type == "caffe.Exp";
    const float base = get_float(op, "base", -1.f);
    const float scale = get_float(op, "scale", 1.f);
    const float shift = get_float(op, "shift", 0.f);
    if (base != -1.f && (base <= 0.f || (!exp && base == 1.f)))
    {
        fprintf(stderr, "caffe layer %s (%s): base must be -1 or positive, and Log base cannot be 1; got %g\n", op->name.c_str(), op->type.c_str(), base);
        return -1;
    }
    op->outputs[0]->shape = op->inputs[0]->shape;
    op->params.clear();
    op->params["base"] = base;
    op->params["scale"] = scale;
    op->params["shift"] = shift;
    op->outputs[0]->params = op->inputs[0]->params;
    return 0;
}

class lower_exp_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Exp               op_0            1 1 input out base=%base scale=%scale shift=%shift
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
aten::exp               exp             1 1 affine out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("base").f == -1.f;
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

class lower_exp_base_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Exp               op_0            1 1 input out base=%base scale=%scale shift=%shift
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
prim::Constant          base            0 1 base value=%base
aten::pow               pow             2 1 base affine out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("base").f != -1.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"scale", "shift", "base"})
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

class lower_log_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Log               op_0            1 1 input out base=%base scale=%scale shift=%shift
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
aten::log               log             1 1 affine out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("base").f == -1.f;
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

class lower_log_base_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input           0 1 input
caffe.Log               op_0            1 1 input out base=%base scale=%scale shift=%shift
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input           0 1 input
prim::Constant          scale           0 1 scale value=%scale
prim::Constant          shift           0 1 shift value=%shift
aten::mul               mul             2 1 input scale scaled
aten::add               affine          2 1 scaled shift affine
prim::Constant          base            0 1 base value=%base
aten::log               log_base        1 1 base log_base
aten::log               log             1 1 affine log
aten::div               div             2 1 log log_base out
pnnx.Output             output          1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("base").f != -1.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"scale", "shift", "base", "log_base"})
            ops.at(name)->outputs[0]->type = 1;

        // preserve the prepared external output and annotate only internal tensors
        const Operand* input = ops.at("mul")->inputs[0];
        for (const char* name : {"mul", "affine", "log"})
        {
            Operand* output = ops.at(name)->outputs[0];
            output->shape = input->shape;
            output->type = input->type;
            output->params = input->params;
        }
    }
};

void lower_exp_log(Graph& graph)
{
    int opindex = 0;
    lower_exp_pass a;
    pnnx_graph_rewrite(graph, &a, opindex);
    lower_exp_base_pass b;
    pnnx_graph_rewrite(graph, &b, opindex);
    lower_log_pass c;
    pnnx_graph_rewrite(graph, &c, opindex);
    lower_log_base_pass d;
    pnnx_graph_rewrite(graph, &d, opindex);
}

} // namespace caffe2pnnx

} // namespace pnnx
