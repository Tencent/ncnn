// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_dynamic_adaptive_pool.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_dynamic_adaptive_pool_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
prim::Constant          op_1        0 1 outw value=%outw
prim::ListConstruct     op_2        2 1 outh outw output_size
F.adaptive_avg_pool2d   op_3        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_3")->inputs[0]->shape.size();
        return (input_rank == 3 && h_axis == 1) || (input_rank == 4 && h_axis == 2);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, outw};
    }
};

class fuse_dynamic_adaptive_pool_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outw dim=%w_axis
prim::Constant          op_1        0 1 outh value=%outh
prim::ListConstruct     op_2        2 1 outh outw output_size
F.adaptive_avg_pool2d   op_3        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_3")->inputs[0]->shape.size();
        return (input_rank == 3 && w_axis == 2) || (input_rank == 4 && w_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{outh, 0};
    }
};

class fuse_dynamic_adaptive_pool_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
prim::Constant          op_1        0 1 outw value=%outw
prim::ListConstruct     op_2        2 1 outh outw output_size
F.adaptive_max_pool2d   op_3        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_3")->inputs[0]->shape.size();
        return (input_rank == 3 && h_axis == 1) || (input_rank == 4 && h_axis == 2);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, outw};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outw dim=%w_axis
prim::Constant          op_1        0 1 outh value=%outh
prim::ListConstruct     op_2        2 1 outh outw output_size
F.adaptive_max_pool2d   op_3        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_3")->inputs[0]->shape.size();
        return (input_rank == 3 && w_axis == 2) || (input_rank == 4 && w_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{outh, 0};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
prim::Constant          op_1        0 1 outh value=%outh
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1) || (input_rank == 5 && d_axis == 2);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, outh, outw};
    }
};

class fuse_dynamic_adaptive_pool_pass_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
prim::Constant          op_1        0 1 outd value=%outd
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && h_axis == 2) || (input_rank == 5 && h_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{outd, 0, outw};
    }
};

class fuse_dynamic_adaptive_pool_pass_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outw dim=%w_axis
prim::Constant          op_1        0 1 outd value=%outd
prim::Constant          op_2        0 1 outh value=%outh
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && w_axis == 3) || (input_rank == 5 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{outd, outh, 0};
    }
};

class fuse_dynamic_adaptive_pool_pass_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
Tensor.size             op_1        1 1 input outh dim=%h_axis
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1 && h_axis == 2) || (input_rank == 5 && d_axis == 2 && h_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, 0, outw};
    }
};

class fuse_dynamic_adaptive_pool_pass_8 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
Tensor.size             op_1        1 1 input outw dim=%w_axis
prim::Constant          op_2        0 1 outh value=%outh
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1 && w_axis == 3) || (input_rank == 5 && d_axis == 2 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{0, outh, 0};
    }
};

class fuse_dynamic_adaptive_pool_pass_9 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
Tensor.size             op_1        1 1 input outw dim=%w_axis
prim::Constant          op_2        0 1 outd value=%outd
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_avg_pool3d   op_4        2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && h_axis == 2 && w_axis == 3) || (input_rank == 5 && h_axis == 3 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        op->params["output_size"] = std::vector<int>{outd, 0, 0};
    }
};

class fuse_dynamic_adaptive_pool_pass_a : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
prim::Constant          op_1        0 1 outh value=%outh
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1) || (input_rank == 5 && d_axis == 2);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, outh, outw};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_b : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
prim::Constant          op_1        0 1 outd value=%outd
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && h_axis == 2) || (input_rank == 5 && h_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{outd, 0, outw};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_c : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outw dim=%w_axis
prim::Constant          op_1        0 1 outd value=%outd
prim::Constant          op_2        0 1 outh value=%outh
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && w_axis == 3) || (input_rank == 5 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{outd, outh, 0};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
Tensor.size             op_1        1 1 input outh dim=%h_axis
prim::Constant          op_2        0 1 outw value=%outw
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        int h_axis = captured_params.at("h_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1 && h_axis == 2) || (input_rank == 5 && d_axis == 2 && h_axis == 3);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outw = captured_params.at("outw").i;
        op->params["output_size"] = std::vector<int>{0, 0, outw};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_e : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outd dim=%d_axis
Tensor.size             op_1        1 1 input outw dim=%w_axis
prim::Constant          op_2        0 1 outh value=%outh
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int d_axis = captured_params.at("d_axis").i;
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && d_axis == 1 && w_axis == 3) || (input_rank == 5 && d_axis == 2 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outh = captured_params.at("outh").i;
        op->params["output_size"] = std::vector<int>{0, outh, 0};
        op->params["return_indices"] = true;
    }
};

class fuse_dynamic_adaptive_pool_pass_f : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input outh dim=%h_axis
Tensor.size             op_1        1 1 input outw dim=%w_axis
prim::Constant          op_2        0 1 outd value=%outd
prim::ListConstruct     op_3        3 1 outd outh outw output_size
F.adaptive_max_pool3d   op_4        2 2 input output_size out indices return_indices=True
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        int h_axis = captured_params.at("h_axis").i;
        int w_axis = captured_params.at("w_axis").i;
        size_t input_rank = matched_operators.at("op_4")->inputs[0]->shape.size();
        return (input_rank == 4 && h_axis == 2 && w_axis == 3) || (input_rank == 5 && h_axis == 3 && w_axis == 4);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int outd = captured_params.at("outd").i;
        op->params["output_size"] = std::vector<int>{outd, 0, 0};
        op->params["return_indices"] = true;
    }
};

void fuse_dynamic_adaptive_pool(Graph& graph)
{
    fuse_dynamic_adaptive_pool_pass a;
    fuse_dynamic_adaptive_pool_pass_1 b;
    fuse_dynamic_adaptive_pool_pass_2 c;
    fuse_dynamic_adaptive_pool_pass_3 d;
    fuse_dynamic_adaptive_pool_pass_4 e;
    fuse_dynamic_adaptive_pool_pass_5 f;
    fuse_dynamic_adaptive_pool_pass_6 g;
    fuse_dynamic_adaptive_pool_pass_7 h;
    fuse_dynamic_adaptive_pool_pass_8 i;
    fuse_dynamic_adaptive_pool_pass_9 j;
    fuse_dynamic_adaptive_pool_pass_a k;
    fuse_dynamic_adaptive_pool_pass_b l;
    fuse_dynamic_adaptive_pool_pass_c m;
    fuse_dynamic_adaptive_pool_pass_d n;
    fuse_dynamic_adaptive_pool_pass_e o;
    fuse_dynamic_adaptive_pool_pass_f p;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
    pnnx_graph_rewrite(graph, &g, opindex);
    pnnx_graph_rewrite(graph, &h, opindex);
    pnnx_graph_rewrite(graph, &i, opindex);
    pnnx_graph_rewrite(graph, &j, opindex);
    pnnx_graph_rewrite(graph, &k, opindex);
    pnnx_graph_rewrite(graph, &l, opindex);
    pnnx_graph_rewrite(graph, &m, opindex);
    pnnx_graph_rewrite(graph, &n, opindex);
    pnnx_graph_rewrite(graph, &o, opindex);
    pnnx_graph_rewrite(graph, &p, opindex);
}

} // namespace pnnx
