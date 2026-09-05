// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"
#include "utils.h"

#include <string.h>

namespace pnnx {

class torch_full : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 size
pnnx.Input              input_1     0 1 fill_value
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 requires_grad value=*
aten::full              op_4        6 1 size fill_value dtype layout device requires_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.full";
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_full, 20)

class torch_full_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 size
ConstantOfShape         op_0        1 1 size out value=%fill_value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.full";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["fill_value"] = captured_params.at("fill_value");
        op->params["dtype"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_full_onnx, 21)

class torch_full_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 size
pnnx.Attribute          value       0 1 value @data=(1)f32
tnn.ConstantOfShape     op_0        2 1 size value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.full";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["fill_value"] = ((const float*)captured_attrs.at("value.data").data.data())[0];
        op->params["dtype"] = "torch.float";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_full_tnn, 21)

class torch_full_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: torch.full with constant size/fill_value/dtype (e.g. the
        // concretized full(x.size(), 1.5)); fold to an Attribute
        return R"PNNXIR(7767517
7 6
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 fill_value value=%fill_value
prim::Constant          op_2        0 1 dtype value=%dtype
prim::Constant          op_3        0 1 device value=*
prim::Constant          op_4        0 1 pin_memory value=*
aten::full              op_5        5 1 size fill_value dtype device pin_memory out
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

        double fv = 0;
        const Parameter& fill_value = captured_params.at("fill_value");
        if (fill_value.type == 3)
            fv = fill_value.f;
        else if (fill_value.type == 2)
            fv = fill_value.i;

        Attribute& a = op->attrs["data"];
        a.type = op->outputs[0]->type;
        a.shape = shape;

        size_t es = 4;
        if (a.type == 2 || a.type == 5) es = 8;                 // f64/i64
        if (a.type == 3 || a.type == 6 || a.type == 13) es = 2; // f16/i16/bf16
        if (a.type == 7 || a.type == 8 || a.type == 9) es = 1;  // i8/u8/bool
        if (a.type == 10) es = 8;                               // complex64
        if (a.type == 11) es = 16;                              // complex128

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

        a.data.resize(count * es);
        char* d = a.data.data();
        if (a.type == 1) // f32
        {
            float* p = (float*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = (float)fv;
        }
        else if (a.type == 2) // f64
        {
            double* p = (double*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = fv;
        }
        else if (a.type == 3) // f16
        {
            const unsigned short v = float32_to_float16((float)fv);
            unsigned short* p = (unsigned short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = v;
        }
        else if (a.type == 13) // bf16
        {
            unsigned int bits;
            float f = (float)fv;
            memcpy(&bits, &f, 4);
            const unsigned short v = (unsigned short)((bits + 0x8000) >> 16);
            unsigned short* p = (unsigned short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = v;
        }
        else if (a.type == 4 || a.type == 5) // i32/i64
        {
            long long v = (long long)fv;
            for (size_t i = 0; i < count; i++)
            {
                if (a.type == 4)
                    ((int*)d)[i] = (int)v;
                else
                    ((long long*)d)[i] = v;
            }
        }
        else if (a.type == 6) // i16
        {
            short* p = (short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = (short)fv;
        }
        else if (a.type == 7 || a.type == 8 || a.type == 9) // i8/u8/bool
        {
            memset(d, fv ? 1 : 0, count);
        }
        else // fallback (complex etc. kept simple)
        {
            memset(d, 0, count * es);
        }
        op->params.clear();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_full_fold, 30)

class Tensor_new_full_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: Tensor.new_full(self, size, fill_value, ...) with constant
        // arguments; fold to an Attribute filled with fill_value
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 fill_value value=%fill_value
prim::Constant          op_2        0 1 dtype value=%dtype
prim::Constant          op_3        0 1 layout value=*
prim::Constant          op_4        0 1 device value=*
prim::Constant          op_5        0 1 pin_memory value=*
aten::new_full          op_6        7 1 input size fill_value dtype layout device pin_memory out
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

        double fv = 0;
        const Parameter& fill_value = captured_params.at("fill_value");
        if (fill_value.type == 3)
            fv = fill_value.f;
        else if (fill_value.type == 2)
            fv = fill_value.i;

        Attribute& a = op->attrs["data"];
        a.type = op->outputs[0]->type;
        a.shape = shape;

        size_t es = 4;
        if (a.type == 2 || a.type == 5) es = 8;                 // f64/i64
        if (a.type == 3 || a.type == 6 || a.type == 13) es = 2; // f16/i16/bf16
        if (a.type == 7 || a.type == 8 || a.type == 9) es = 1;  // i8/u8/bool
        if (a.type == 10) es = 8;                               // complex64
        if (a.type == 11) es = 16;                              // complex128

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

        a.data.resize(count * es);
        char* d = a.data.data();
        if (a.type == 1) // f32
        {
            float* p = (float*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = (float)fv;
        }
        else if (a.type == 2) // f64
        {
            double* p = (double*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = fv;
        }
        else if (a.type == 3) // f16
        {
            const unsigned short v = float32_to_float16((float)fv);
            unsigned short* p = (unsigned short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = v;
        }
        else if (a.type == 13) // bf16
        {
            unsigned int bits;
            float f = (float)fv;
            memcpy(&bits, &f, 4);
            const unsigned short v = (unsigned short)((bits + 0x8000) >> 16);
            unsigned short* p = (unsigned short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = v;
        }
        else if (a.type == 4 || a.type == 5) // i32/i64
        {
            const long long v = (long long)fv;
            for (size_t i = 0; i < count; i++)
            {
                if (a.type == 4)
                    ((int*)d)[i] = (int)v;
                else
                    ((long long*)d)[i] = v;
            }
        }
        else if (a.type == 6) // i16
        {
            short* p = (short*)d;
            for (size_t i = 0; i < count; i++)
                p[i] = (short)fv;
        }
        else if (a.type == 7 || a.type == 8 || a.type == 9) // i8/u8/bool
        {
            memset(d, fv ? 1 : 0, count);
        }
        else
        {
            memset(d, 0, count * es);
        }
        op->params.clear();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_new_full_fold, 30)

} // namespace pnnx
