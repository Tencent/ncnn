// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_batch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 running_mean
pnnx.Input              input_2     0 1 running_var
pnnx.Input              input_3     0 1 weight
pnnx.Input              input_4     0 1 bias
prim::Constant          op_0        0 1 training value=*
prim::Constant          op_1        0 1 momentum value=*
prim::Constant          op_2        0 1 eps value=%eps
prim::Constant          op_3        0 1 cudnn_enabled value=*
aten::batch_norm        op_4        9 1 input weight bias running_mean running_var training momentum eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.batch_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm, 130)

class F_batch_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 10
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 running_mean
pnnx.Input              input_2     0 1 running_var
pnnx.Input              input_3     0 1 weight
pnnx.Input              input_4     0 1 bias
prim::Constant          op_0        0 1 momentum value=*
prim::Constant          op_1        0 1 eps value=%eps
aten::_native_batch_norm_legit_no_training op_2 7 3 input weight bias running_mean running_var momentum eps out save_mean save_invstd
pnnx.Output             output      3 0 out save_mean save_invstd
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.batch_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(op, captured_params, captured_attrs);

        op->outputs.resize(1);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm_1, 130)

class F_batch_norm_onnx : public GraphRewriterPass
{
public:
    // https://github.com/pytorch/pytorch/pull/29458
    // unsqueeze - batchnorm - squeeze
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 running_mean
pnnx.Input              input_4     0 1 running_var
torch.unsqueeze         op_0        1 1 input a dim=2
BatchNormalization      op_1        5 1 a weight bias running_mean running_var b %*=%*
torch.squeeze           op_2        1 1 b out dim=2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.batch_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.epsilon") != captured_params.end())
        {
            op->params["eps"] = captured_params.at("op_0.epsilon");
        }
        else
        {
            op->params["eps"] = 1e-05;
        }

        std::swap(op->inputs[1], op->inputs[3]);
        std::swap(op->inputs[2], op->inputs[4]);
        std::swap(op->inputnames[1], op->inputnames[3]);
        std::swap(op->inputnames[2], op->inputnames[4]);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm_onnx, 130)

class F_batch_norm_onnx_1 : public F_batch_norm_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 running_mean
pnnx.Input              input_4     0 1 running_var
BatchNormalization      op_0        5 1 input weight bias running_mean running_var out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm_onnx_1, 131)

class F_batch_norm_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 weight @data=(%num_features)f32
pnnx.Attribute          op_1        0 1 bias @data=(%num_features)f32
tnn.BatchNormCxx        op_2        3 1 input weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          mean        0 1 running_mean
pnnx.Attribute          var         0 1 running_var
pnnx.Attribute          weight      0 1 weight @data=%op_0.data
pnnx.Attribute          bias        0 1 bias @data=%op_1.data
F.batch_norm            bn          5 1 input running_mean running_var weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        const int num_features = captured_params.at("num_features").i;

        Operator* op_mean = ops.at("mean");
        op_mean->attrs["data"] = Attribute({num_features}, std::vector<float>(num_features, 0.f));

        Operator* op_var = ops.at("var");
        op_var->attrs["data"] = Attribute({num_features}, std::vector<float>(num_features, 1.f));

        Operator* op_bn = ops.at("bn");
        op_bn->params["eps"] = 0.f;
        op_bn->inputnames = {"input", "running_mean", "running_var", "weight", "bias"};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm_tnn, 130)

} // namespace pnnx
