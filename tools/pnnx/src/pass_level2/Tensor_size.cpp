// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_size : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
aten::size              op_1        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.size";
    }
};

class Tensor_size_dynamic : public Tensor_size
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
aten::size              op_1        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size, 10)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_dynamic, 12)

class Tensor_size_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::size              op_0        1 1 input shape
Gather                  op_1        1 1 shape out axis=0 indices=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.size";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_onnx, 11)

class Tensor_size_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input shape dim=%dim
Unsqueeze               op_1        1 1 shape out axes=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.size";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_onnx_1, 12)

class Tensor_size_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::size              op_0        1 1 input shape
Slice                   op_1        1 1 shape out axes=0 starts=%starts ends=%ends steps=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.size";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("starts").type != 2 || captured_params.at("ends").type != 2)
            return false;

        const int start = captured_params.at("starts").i;
        const int end = captured_params.at("ends").i;
        if (end != start + 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("starts");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_onnx_2, 11)

class Tensor_size_onnx_3 : public Tensor_size_onnx_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
aten::size              op_0        1 1 input shape
Slice                   op_1        1 1 shape a axes=0 starts=%starts ends=%ends steps=1
Squeeze                 op_2        1 1 a b axes=0
Reshape                 op_3        1 1 b out allowzero=0 shape=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_onnx_3, 10)

class Tensor_size_onnx_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::size              op_0        1 1 input out start=%start end=%end
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.size";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("start").type != 2 || captured_params.at("end").type != 2)
            return false;

        const int start = captured_params.at("start").i;
        const int end = captured_params.at("end").i;
        if (end != start + 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("start");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_size_onnx_4, 10)

} // namespace pnnx
