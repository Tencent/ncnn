// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_batch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data
pnnx.Attribute          op_var      0 1 running_var @data
F.batch_norm            op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "BatchNorm";
    }

    const char* name_str() const
    {
        return "bn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute running_mean = captured_attrs.at("op_mean.data");
        Attribute running_var = captured_attrs.at("op_var.data");

        op->params["0"] = running_mean.shape[0];
        op->params["1"] = captured_params.at("eps");

        const int channels = running_mean.shape[0];

        op->attrs["0"] = Attribute({channels}, std::vector<float>(channels, 1.f));
        op->attrs["1"] = running_mean;
        op->attrs["2"] = running_var;
        op->attrs["3"] = Attribute({channels}, std::vector<float>(channels, 0.f));
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_batch_norm, 20)

class F_batch_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @data
pnnx.Attribute          op_var      0 1 running_var @data
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "BatchNorm";
    }

    const char* name_str() const
    {
        return "bn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute running_mean = captured_attrs.at("op_mean.data");
        Attribute running_var = captured_attrs.at("op_var.data");
        Attribute weight = captured_attrs.at("op_weight.data");
        Attribute bias = captured_attrs.at("op_bias.data");

        op->params["0"] = running_mean.shape[0];
        op->params["1"] = captured_params.at("eps");

        op->attrs["0"] = weight;
        op->attrs["1"] = running_mean;
        op->attrs["2"] = running_var;
        op->attrs["3"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_batch_norm_1, 20)

} // namespace ncnn

} // namespace pnnx
