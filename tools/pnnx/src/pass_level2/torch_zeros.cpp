// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_zeros : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 size
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 requires_grad value=*
aten::zeros             op_4        5 1 size dtype layout device requires_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.zeros";
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
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_zeros, 20)

class torch_zeros_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 size
ConstantOfShape         op_0        1 1 size out value=0.0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.zeros";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["dtype"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_zeros_onnx, 20)

class torch_zeros_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: GRU/LSTM/RNN initial-state zeros have size/dtype etc. as
        // prim::Constant inputs; ncnn has no zeros layer, fold to all-zero data
        return R"PNNXIR(7767517
6 5
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 dtype value=%dtype
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
aten::zeros             op_4        4 1 size dtype device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("size").ai;

        Attribute& a = op->attrs["data"];
        a.type = op->outputs[0]->type;
        a.shape = shape;

        size_t es = 4;
        if (a.type == 2 || a.type == 5) es = 8;                 // f64/i64
        if (a.type == 3 || a.type == 6 || a.type == 13) es = 2; // f16/i16/bf16
        if (a.type == 7 || a.type == 8 || a.type == 9) es = 1;  // i8/u8/bool

        // reject dynamic / invalid (non-positive) dimensions before allocating
        size_t count = 1;
        bool valid = true;
        for (int s : shape)
        {
            if (s <= 0 || count > (size_t)-1 / (size_t)s)
            {
                valid = false;
                break;
            }
            count *= (size_t)s;
        }
        if (!valid || count > (size_t)-1 / es)
            count = 0;

        a.data.resize(count * es, 0);
        op->params.clear();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_zeros_fold, 30)

} // namespace pnnx
