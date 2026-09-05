// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <cstring>
#include <set>

#include "utils.h"

namespace pnnx {

class torch_arange : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 end
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 requires_grad value=*
aten::arange            op_4        5 1 end dtype layout device requires_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.arange";
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange, 20)

class torch_arange_1 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
pnnx.Input              input_2     0 1 step
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 requires_grad value=*
aten::arange            op_4        7 1 start end step dtype layout device requires_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_1, 20)

class torch_arange_2 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
aten::arange            op_4        6 1 start end dtype layout device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_2, 20)

class torch_arange_3 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 pin_memory value=*
aten::arange            op_3        6 1 start end dtype layout layout pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_3, 20)

class torch_arange_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
pnnx.Input              input_2     0 1 step
Range                   op_0        3 1 start end step out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.arange";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dtype"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_onnx, 20)

class torch_arange_5 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: arange(end, *, dtype, device, pin_memory) (dynamo omits layout/requires_grad)
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 end
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 device value=*
prim::Constant          op_2        0 1 pin_memory value=*
aten::arange            op_3        4 1 end dtype device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_5, 20)

// pt2: arange(start, end, step, dtype, device, pin_memory) (aten::arange.start_step)
class torch_arange_6 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
pnnx.Input              input_2     0 1 step
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 device value=*
prim::Constant          op_2        0 1 pin_memory value=*
aten::arange.start_step op_3        6 1 start end step dtype device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_6, 20)

// pt2: arange(start, end, dtype, device, pin_memory) (aten::arange.start, no step)
class torch_arange_7 : public torch_arange
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 start
pnnx.Input              input_1     0 1 end
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 device value=*
prim::Constant          op_2        0 1 pin_memory value=*
aten::arange.start      op_3        5 1 start end dtype device pin_memory out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_7, 20)

class torch_arange_params : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // pt2 path: constant args fused into params (0 inputs); match and rewrite to torch.arange
        return R"PNNXIR(7767517
2 1
aten::arange            op_0        0 1 out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.arange";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // keys captured by %*=%* carry an "op_0." prefix; strip it before writing back
        for (const auto& x : captured_params)
        {
            std::string key = x.first;
            size_t dot = key.rfind('.');
            if (dot != std::string::npos)
                key = key.substr(dot + 1);
            op->params[key] = x.second;
        }

        // map the pnnx dtype value to a torch dtype string (consistent with torch_arange)
        if (op->params.find("dtype") != op->params.end() && op->params.at("dtype").type == 2)
        {
            int dtype = op->params.at("dtype").i;
            const char* dtype_str = 0;
            if (dtype == 0) dtype_str = "torch.uint8";
            if (dtype == 1) dtype_str = "torch.int8";
            if (dtype == 2) dtype_str = "torch.short";
            if (dtype == 3) dtype_str = "torch.int";
            if (dtype == 4) dtype_str = "torch.long";
            if (dtype == 5) dtype_str = "torch.half";
            if (dtype == 6) dtype_str = "torch.float";
            if (dtype == 7) dtype_str = "torch.double";
            if (dtype == 9) dtype_str = "torch.bool";
            if (dtype == 15) dtype_str = "torch.bfloat16";
            if (dtype_str)
                op->params["dtype"] = dtype_str;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_params, 20)

static void propagate_f32_dtype_from_arange(Operand* start)
{
    // starting from a folded f32 Attribute output, propagate dtype=1(f32) to
    // downstream ops that keep the dtype (views/type casts) so the ncnn
    // conversion needs no i64->f32 Cast. Integer arange results (e.g. position
    // ids) are exactly representable in f32, so the conversion is safe.
    std::vector<Operand*> queue;
    queue.push_back(start);
    std::set<std::string> visited;
    while (!queue.empty())
    {
        Operand* opd = queue.back();
        queue.pop_back();
        if (visited.find(opd->name) != visited.end())
            continue;
        visited.insert(opd->name);

        for (Operator* consumer : opd->consumers)
        {
            const std::string& t = consumer->type;
            const bool dtype_preserving = t == "aten::unsqueeze" || t == "Tensor.unsqueeze" || t == "aten::expand" || t == "Tensor.expand" || t == "aten::reshape" || t == "Tensor.reshape" || t == "aten::view" || t == "Tensor.view" || t == "aten::permute" || t == "Tensor.permute" || t == "aten::transpose" || t == "Tensor.transpose" || t == "aten::contiguous" || t == "Tensor.contiguous" || t == "aten::squeeze" || t == "Tensor.squeeze" || t == "aten::flatten" || t == "Tensor.flatten" || t == "aten::to" || t == "Tensor.to" || t == "aten::_to_copy";
            if (!dtype_preserving)
                continue;

            for (Operand* out : consumer->outputs)
            {
                out->type = 1;
                queue.push_back(out);
            }
        }
    }
}

class torch_arange_fold : public GraphRewriterPass
{
public:
    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        double start = 0;
        double end = 0;
        double step = 1;
        bool has_end = false;

        for (const auto& x : captured_params)
        {
            std::string key = x.first;
            size_t dot = key.rfind('.');
            if (dot != std::string::npos)
                key = key.substr(dot + 1);

            if (key == "start")
            {
                if (x.second.type == 2)
                    start = x.second.i;
                else if (x.second.type == 3)
                    start = x.second.f;
                else
                    return;
            }
            if (key == "end")
            {
                if (x.second.type == 2)
                {
                    end = x.second.i;
                    has_end = true;
                }
                else if (x.second.type == 3)
                {
                    end = x.second.f;
                    has_end = true;
                }
                else
                    return;
            }
            if (key == "step")
            {
                if (x.second.type == 2)
                    step = x.second.i;
                else if (x.second.type == 3)
                    step = x.second.f;
                else
                    return;
            }
        }

        // no end or step==0 (non-constant/invalid bounds): do not fold, keep the op
        if (!has_end || step == 0)
        {
            op->type = "torch.arange";
            return;
        }

        double dcount = (end - start) / step;
        int64_t count = (int64_t)ceil(dcount);
        if (count < 0) count = 0;

        Attribute& a = op->attrs["data"];
        a.type = op->outputs[0]->type;
        a.shape = {(int)count};
        op->outputs[0]->shape = a.shape;

        const int dtype = a.type;

        // keep the original dtype of integer arange (e.g. i64) so downstream
        // torch.gather index types stay correct. Note: do not unconditionally
        // cast to f32 here; if f32 is needed downstream, Tensor.to /
        // propagate_f32_dtype_from_arange handle it.
        int fold_dtype = dtype;

        if (fold_dtype == 5) // i64
        {
            a.data.resize(count * 8);
            int64_t* p = (int64_t*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (int64_t)(start + i * step);
        }
        else if (fold_dtype == 4) // i32
        {
            a.data.resize(count * 4);
            int* p = (int*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (int)(start + i * step);
        }
        else if (fold_dtype == 6) // i16
        {
            a.data.resize(count * 2);
            short* p = (short*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (short)(start + i * step);
        }
        else if (fold_dtype == 7) // i8
        {
            a.data.resize(count);
            signed char* p = (signed char*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (signed char)(start + i * step);
        }
        else if (fold_dtype == 8) // u8
        {
            a.data.resize(count);
            unsigned char* p = (unsigned char*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (unsigned char)(start + i * step);
        }
        else if (fold_dtype == 9) // bool
        {
            a.data.resize(count);
            unsigned char* p = (unsigned char*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = ((int64_t)(start + i * step)) != 0 ? 1 : 0;
        }
        else if (fold_dtype == 2) // f64
        {
            a.data.resize(count * 8);
            double* p = (double*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = start + i * step;
        }
        else if (fold_dtype == 3) // f16
        {
            a.data.resize(count * 2);
            unsigned short* p = (unsigned short*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = float32_to_float16((float)(start + i * step));
        }
        else if (fold_dtype == 13) // bf16
        {
            a.data.resize(count * 2);
            unsigned short* p = (unsigned short*)a.data.data();
            for (int64_t i = 0; i < count; i++)
            {
                float f = (float)(start + i * step);
                unsigned int bits;
                memcpy(&bits, &f, 4);
                // round-to-nearest-even (matching torch's float -> bfloat16
                // cast) instead of a plain top-bit truncation: e.g. 259 must
                // fold to 260, not 258
                unsigned int rounding_bias = 0x7fff + ((bits >> 16) & 1);
                p[i] = (unsigned short)((bits + rounding_bias) >> 16);
            }
        }
        else // f32
        {
            a.data.resize(count * 4);
            float* p = (float*)a.data.data();
            for (int64_t i = 0; i < count; i++)
                p[i] = (float)(start + i * step);
        }

        if (fold_dtype != dtype)
        {
            a.type = fold_dtype;
            op->outputs[0]->type = fold_dtype;
            propagate_f32_dtype_from_arange(op->outputs[0]);
        }

        op->params.clear();
    }
};

class torch_arange_fold_1 : public torch_arange_fold
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: arange(end), end is a constant
        return R"PNNXIR(7767517
4 3
prim::Constant          op_0        0 1 end value=%end
torch.arange            op_1        1 1 end out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_fold_1, 30)

class torch_arange_fold_2 : public torch_arange_fold
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: arange(start, end), start/end are constants
        return R"PNNXIR(7767517
5 4
prim::Constant          op_0        0 1 start value=%start
prim::Constant          op_1        0 1 end value=%end
torch.arange            op_2        2 1 start end out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_fold_2, 30)

class torch_arange_fold_3 : public torch_arange_fold
{
public:
    const char* match_pattern_graph() const
    {
        // pt2: arange(start, end, step), all constants
        return R"PNNXIR(7767517
6 5
prim::Constant          op_0        0 1 start value=%start
prim::Constant          op_1        0 1 end value=%end
prim::Constant          op_2        0 1 step value=%step
torch.arange            op_3        3 1 start end step out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_arange_fold_3, 30)

} // namespace pnnx
