// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_where : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 condition
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 other
torch.where             op_0        3 1 condition input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "where";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_where, 20)

class torch_where_scalar_a : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 condition
pnnx.Input              input_1     0 1 other
torch.where             op_0        2 1 condition other out input=%input
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "where";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float input_val = captured_params.at("input").type == 3 ? captured_params.at("input").f : (float)captured_params.at("input").i;
        op->params["0"] = 1;
        op->params["1"] = input_val;
        op->params["2"] = 0;
        op->params["3"] = 0.f;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_where_scalar_a, 21)

class torch_where_scalar_b : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 condition
pnnx.Input              input_1     0 1 input
torch.where             op_0        2 1 condition input out other=%other
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "where";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float other_val = captured_params.at("other").type == 3 ? captured_params.at("other").f : (float)captured_params.at("other").i;
        op->params["0"] = 0;
        op->params["1"] = 0.f;
        op->params["2"] = 1;
        op->params["3"] = other_val;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_where_scalar_b, 22)

class torch_where_scalar_ab : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 condition
torch.where             op_0        1 1 condition out input=%input other=%other
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Where";
    }

    const char* name_str() const
    {
        return "where";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float input_val = captured_params.at("input").type == 3 ? captured_params.at("input").f : (float)captured_params.at("input").i;
        float other_val = captured_params.at("other").type == 3 ? captured_params.at("other").f : (float)captured_params.at("other").i;
        op->params["0"] = 1;
        op->params["1"] = input_val;
        op->params["2"] = 1;
        op->params["3"] = other_val;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_where_scalar_ab, 23)

} // namespace ncnn

} // namespace pnnx
