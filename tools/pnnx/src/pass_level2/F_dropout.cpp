// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <iomanip>
#include <sstream>

namespace pnnx {

class F_dropout : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 p
pnnx.Input              input_2     0 1 training
aten::dropout           op_0        3 1 input p training out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.dropout";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_dropout, 100)

class F_dropout_caffe : public F_dropout
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.Dropout           op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        auto it = captured_params.find("op_0.scale_train");
        return it == captured_params.end() || it->second.b;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["p"] = 0.f;
        op->params["training"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_dropout_caffe, 100)

class F_dropout_caffe_1 : public F_dropout_caffe
{
public:
    const char* type_str() const
    {
        return "pnnx.Expression";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return !F_dropout_caffe::match(captured_params);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        auto it = captured_params.find("op_0.dropout_ratio");
        const float ratio = it == captured_params.end() ? .5f : it->second.f;
        std::ostringstream ss;
        ss << std::setprecision(9) << "mul(@0," << 1.f - ratio << ")";
        op->params["expr"] = ss.str();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_dropout_caffe_1, 100)

} // namespace pnnx
