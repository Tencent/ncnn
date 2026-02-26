// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_softmax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 dtype value=*
aten::softmax           op_2        3 1 input dim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax, 100)

class F_softmax_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::softmax_no_dtype  op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_1, 100)

class F_softmax_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Softmax                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axis") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axis");
        }
        else
        {
            op->params["dim"] = -1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx, 101)

class F_softmax_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
Tensor.permute          op_0        1 1 input a dims=%dims
Softmax                 op_1        1 1 a b axis=%axis
Tensor.permute          op_2        1 1 b out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        if (axis >= (int)dims.size())
            return false;

        int excount = 0;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] != i)
                excount++;
        }

        if (excount != 2)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        op->params["dim"] = dims[axis];
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx_1, 100)

class F_softmax_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.SoftmaxCaffe        op_0        1 1 input out arg0=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_tnn, 100)

} // namespace pnnx
