// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_embedding : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
prim::Constant          op_0        0 1 padding_idx value=*
prim::Constant          op_1        0 1 scale_grad_by_freq value=%scale_grad_by_freq
prim::Constant          op_2        0 1 sparse value=%sparse
aten::embedding         op_3        5 1 weight input padding_idx scale_grad_by_freq sparse out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding, 140)

class F_embedding_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
Gather                  op_0        2 1 weight input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["scale_grad_by_freq"] = false;
        op->params["sparse"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_onnx, 140)

class F_embedding_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Gather              op_0        2 1 input weight out arg0=0 arg1=1 arg2=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["scale_grad_by_freq"] = false;
        op->params["sparse"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_tnn, 140)

class F_embedding_caffe : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.Embed             op_0        1 1 input out bias_term=False @0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data=%op_0.0
F.embedding             embed       2 1 input weight out scale_grad_by_freq=False sparse=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);
        for (const char* name : {"weight", "bias"})
        {
            if (ops.find(name) == ops.end())
                continue;
            Operator* attr = ops.at(name);
            attr->outputs[0]->shape = attr->attrs.at("data").shape;
            attr->outputs[0]->type = attr->attrs.at("data").type;
            attr->outputs[0]->params["__batch_index"] = 233;
        }
        Operator* embed = ops.at("embed");
        embed->inputnames = {"input", "weight"};
        embed->outputs[0]->shape = embed->inputs[0]->shape;
        embed->outputs[0]->shape.push_back(ops.at("weight")->attrs.at("data").shape[1]);
        embed->outputs[0]->type = ops.at("weight")->attrs.at("data").type;
        embed->outputs[0]->params["__batch_index"] = embed->inputs[0]->params.at("__batch_index");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_caffe, 140)

class F_embedding_caffe_bias : public F_embedding_caffe
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.Embed             op_0        1 1 input out bias_term=True @0 @1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data=%op_0.0
pnnx.Attribute          bias        0 1 bias @data=%op_0.1
F.embedding             embed       2 1 input weight embedded scale_grad_by_freq=False sparse=False
pnnx.Expression         add         2 1 embedded bias out expr=add(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_caffe_bias, 140)

} // namespace pnnx
