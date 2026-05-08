// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_to : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 non_blocking value=*
prim::Constant          op_2        0 1 copy value=%copy
prim::Constant          op_3        0 1 memory_format value=%memory_format
aten::to                op_4        5 1 input dtype non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.to";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dtype").type == 0)
        {
            op->params["dtype"] = Parameter();
        }
        else // if (captured_params.at("dtype").type == 2)
        {
            if (captured_params.at("dtype").i == 0) op->params["dtype"] = "torch.uint8";
            if (captured_params.at("dtype").i == 1) op->params["dtype"] = "torch.int8";
            if (captured_params.at("dtype").i == 2) op->params["dtype"] = "torch.short";
            if (captured_params.at("dtype").i == 3) op->params["dtype"] = "torch.int";
            if (captured_params.at("dtype").i == 4) op->params["dtype"] = "torch.long";
            if (captured_params.at("dtype").i == 5) op->params["dtype"] = "torch.half";
            if (captured_params.at("dtype").i == 6) op->params["dtype"] = "torch.float";
            if (captured_params.at("dtype").i == 7) op->params["dtype"] = "torch.double";
            if (captured_params.at("dtype").i == 8) op->params["dtype"] = "torch.complex32";
            if (captured_params.at("dtype").i == 9) op->params["dtype"] = "torch.complex64";
            if (captured_params.at("dtype").i == 10) op->params["dtype"] = "torch.complex128";
            if (captured_params.at("dtype").i == 11) op->params["dtype"] = "torch.bool";
            if (captured_params.at("dtype").i == 15) op->params["dtype"] = "torch.bfloat16";
        }

        op->params["copy"] = captured_params.at("copy");

        if (captured_params.at("memory_format").type == 2)
        {
            if (captured_params.at("memory_format").i == 0)
                op->params["memory_format"] = "torch.contiguous_format";
            if (captured_params.at("memory_format").i == 1)
                op->params["memory_format"] = "torch.preserve_format";
            if (captured_params.at("memory_format").i == 2)
                op->params["memory_format"] = "torch.channels_last";
        }
    }
};

class Tensor_to_1 : public Tensor_to
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 device value=*
prim::Constant          op_1        0 1 dtype value=%dtype
prim::Constant          op_2        0 1 non_blocking value=*
prim::Constant          op_3        0 1 copy value=%copy
prim::Constant          op_4        0 1 memory_format value=%memory_format
aten::to                op_5        6 1 input device dtype non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class Tensor_to_2 : public Tensor_to
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
prim::Constant          op_4        0 1 non_blocking value=*
prim::Constant          op_5        0 1 copy value=%copy
prim::Constant          op_6        0 1 memory_format value=%memory_format
aten::to                op_7        8 1 input dtype layout device pin_memory non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_1, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_2, 60)

class Tensor_to_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Cast                    op_0        1 1 input out to=%to
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.to";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int to = captured_params.at("to").i;

        op->params["non_blocking"] = false;
        op->params["copy"] = false;
        op->params["memory_format"] = "torch.preserve_format";

        if (to == 1) op->params["dtype"] = "torch.float";
        if (to == 2) op->params["dtype"] = "torch.uint8";
        if (to == 3) op->params["dtype"] = "torch.int8";
        if (to == 5) op->params["dtype"] = "torch.short";
        if (to == 6) op->params["dtype"] = "torch.int";
        if (to == 7) op->params["dtype"] = "torch.long";
        if (to == 9) op->params["dtype"] = "torch.bool";
        if (to == 10) op->params["dtype"] = "torch.half";
        if (to == 11) op->params["dtype"] = "torch.double";
        if (to == 14) op->params["dtype"] = "torch.complex64";
        if (to == 15) op->params["dtype"] = "torch.complex128";
        if (to == 16) op->params["dtype"] = "torch.bfloat16";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_onnx, 60)

class Tensor_to_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Cast                op_0        1 1 input out arg0=%to
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.to";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int to = captured_params.at("to").i;

        op->params["non_blocking"] = false;
        op->params["copy"] = false;
        op->params["memory_format"] = "torch.preserve_format";

        if (to == 0) op->params["dtype"] = "torch.float";
        if (to == 1) op->params["dtype"] = "torch.half";
        if (to == 2) op->params["dtype"] = "torch.int8";
        if (to == 3) op->params["dtype"] = "torch.int";
        if (to == 4) op->params["dtype"] = "torch.bfloat16";
        if (to == 5) op->params["dtype"] = "torch.long";
        if (to == 6) op->params["dtype"] = "torch.uint32";
        if (to == 8) op->params["dtype"] = "torch.uint8";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_tnn, 60)

} // namespace pnnx
