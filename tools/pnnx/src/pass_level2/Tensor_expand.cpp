// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_expand : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 sizes
prim::Constant          op_0        0 1 implicit value=*
aten::expand            op_1        3 1 input sizes implicit out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.expand";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_expand, 60)

class Tensor_expand_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 sizes
aten::expand            op_1        2 1 input sizes out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.expand";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_expand_1, 60)

class Tensor_expand_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Expand                  op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.expand";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.shape") == captured_params.end())
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("op_0.shape").type == 5)
        {
            op->params["sizes"] = captured_params.at("op_0.shape");
        }
        else // if (captured_params.at("op_0.shape").type == 2)
        {
            op->params["sizes"] = std::vector<int>{captured_params.at("op_0.shape").i};
        }

        // onnx set expand shape 1 for not changing the size of that dimension while torch uses -1
        for (size_t i = 0; i < op->params["sizes"].ai.size(); i++)
        {
            if (op->params["sizes"].ai[i] == 1)
                op->params["sizes"].ai[i] = -1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_expand_onnx, 60)

class Tensor_expand_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 sizes
tnn.Expand              op_0        2 1 input sizes out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.expand";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_expand_tnn, 61)

} // namespace pnnx
