// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_binaryop_eltwise.h"

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class fuse_binaryop_eltwise_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
BinaryOp                op_0        1 1 a a2 0=2 1=1 2=%c0
BinaryOp                op_1        1 1 b b2 0=2 1=1 2=%c1
BinaryOp                op_2        2 1 a2 b2 out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Eltwise";
    }

    const char* name_str() const
    {
        return "weighted_sum";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        auto a_shape = matched_operators.at("op_0")->inputs[0]->shape;
        auto b_shape = matched_operators.at("op_1")->inputs[0]->shape;
        return !a_shape.empty() && a_shape == b_shape;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        float c0 = 1.f;
        float c1 = 1.f;

        if (captured_params.at("c0").type == 2)
            c0 = (float)captured_params.at("c0").i;
        if (captured_params.at("c0").type == 3)
            c0 = captured_params.at("c0").f;

        if (captured_params.at("c1").type == 2)
            c1 = (float)captured_params.at("c1").i;
        if (captured_params.at("c1").type == 3)
            c1 = captured_params.at("c1").f;

        op->params["0"] = 1;
        op->params["1"] = std::vector<float>{c0, c1};
    }
};

class fuse_binaryop_eltwise_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
BinaryOp                op_0        1 1 a a2 0=2 1=1 2=%c0
BinaryOp                op_1        2 1 a2 b out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Eltwise";
    }

    const char* name_str() const
    {
        return "weighted_sum";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        auto a_shape = matched_operators.at("op_0")->inputs[0]->shape;
        auto b_shape = matched_operators.at("op_1")->inputs[1]->shape;
        return !a_shape.empty() && a_shape == b_shape;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        float c0 = 1.f;
        float c1 = 1.f;

        if (captured_params.at("c0").type == 2)
            c0 = (float)captured_params.at("c0").i;
        if (captured_params.at("c0").type == 3)
            c0 = captured_params.at("c0").f;

        op->params["0"] = 1;
        op->params["1"] = std::vector<float>{c0, c1};
    }
};

class fuse_binaryop_eltwise_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
BinaryOp                op_0        1 1 b b2 0=2 1=1 2=%c1
BinaryOp                op_1        2 1 a b2 out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Eltwise";
    }

    const char* name_str() const
    {
        return "weighted_sum";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        auto a_shape = matched_operators.at("op_1")->inputs[0]->shape;
        auto b_shape = matched_operators.at("op_0")->inputs[0]->shape;
        return !a_shape.empty() && a_shape == b_shape;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        float c0 = 1.f;
        float c1 = 1.f;

        if (captured_params.at("c1").type == 2)
            c1 = (float)captured_params.at("c1").i;
        if (captured_params.at("c1").type == 3)
            c1 = captured_params.at("c1").f;

        op->params["0"] = 1;
        op->params["1"] = std::vector<float>{c0, c1};
    }
};

void fuse_binaryop_eltwise(Graph& graph)
{
    fuse_binaryop_eltwise_pass a;
    fuse_binaryop_eltwise_pass_1 b;
    fuse_binaryop_eltwise_pass_2 c;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
}

} // namespace ncnn

} // namespace pnnx
