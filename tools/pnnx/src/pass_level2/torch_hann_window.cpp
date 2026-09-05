// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"
#include "utils.h"

#include <math.h>
#include <string.h>

namespace pnnx {

// encode a float as bfloat16 (round-to-nearest-even, matching torch's
// float -> bfloat16 cast, rather than a plain top-bit truncation)
static inline unsigned short float32_to_bfloat16(float f)
{
    unsigned int bits;
    memcpy(&bits, &f, 4);
    unsigned int rounding_bias = 0x7fff + ((bits >> 16) & 1);
    return (unsigned short)((bits + rounding_bias) >> 16);
}

// fold torch.hann_window / torch.hamming_window into constant window data.
// On the pt2 path the stft/istft window is a torch.hann_window/hamming_window
// op (0 inputs, window_length etc. as params); ncnn has no such layer and the
// window values only depend on the length, so fold to a pnnx.Attribute.
static void fold_window(Operator* op, const std::map<std::string, Parameter>& captured_params, bool hamming)
{
    int n = 0;
    bool periodic = true;
    float alpha = 0.54f; // hamming_window defaults
    float beta = 0.46f;
    int out_type = 0; // 0 = follow the recorded output operand dtype
    for (const auto& x : captured_params)
    {
        std::string key = x.first;
        size_t dot = key.rfind('.');
        if (dot != std::string::npos)
            key = key.substr(dot + 1);

        if (key == "window_length")
        {
            if (x.second.type == 2)
                n = x.second.i;
        }
        else if (key == "periodic")
        {
            if (x.second.type == 1) // bool (type 9 is not a valid Parameter type)
                periodic = x.second.b;
            else if (x.second.type == 2)
                periodic = x.second.i != 0;
        }
        else if (key == "dtype")
        {
            if (x.second.type == 2) // pnnx type id carried by the loader
                out_type = x.second.i;
        }
        else if (key == "alpha")
        {
            if (x.second.type == 3) // float
                alpha = x.second.f;
        }
        else if (key == "beta")
        {
            if (x.second.type == 3) // float
                beta = x.second.f;
        }
    }

    if (n <= 0)
        return; // cannot fold

    // resolve the attribute dtype: an explicit dtype param wins, otherwise the
    // recorded output operand dtype, otherwise f32
    if (out_type == 0 && op->outputs[0]->type != 0)
        out_type = op->outputs[0]->type;
    if (out_type != 1 && out_type != 2 && out_type != 3 && out_type != 13)
        out_type = 1; // unknown -> f32

    const int elemsize = out_type == 1 ? 4 : (out_type == 2 ? 8 : 2);

    if (n == 1)
    {
        // torch returns [1.0] for a length-1 window; guard the div-by-zero case
        Attribute& a = op->attrs["data"];
        a.type = out_type;
        a.shape = {1};
        a.data.resize(elemsize);
        if (out_type == 1)
            ((float*)a.data.data())[0] = 1.0f;
        else if (out_type == 2)
            ((double*)a.data.data())[0] = 1.0;
        else if (out_type == 3)
            ((unsigned short*)a.data.data())[0] = float32_to_float16(1.f);
        else
            ((unsigned short*)a.data.data())[0] = float32_to_bfloat16(1.f);
        op->outputs[0]->type = out_type;
        op->outputs[0]->shape = a.shape;
        op->params.clear();
        return;
    }

    const int div = periodic ? n : n - 1;

    Attribute& a = op->attrs["data"];
    a.type = out_type;
    a.shape = {n};
    a.data.resize((size_t)n * (size_t)elemsize);
    for (int i = 0; i < n; i++)
    {
        // 2*pi as a literal avoids the MSVC _USE_MATH_DEFINES ordering pitfall
        const double v = hamming ? (double)alpha - (double)beta * cos(2. * 3.14159265358979323846 * i / div)
                         : 0.5 * (1. - cos(2. * 3.14159265358979323846 * i / div));
        if (out_type == 1)
            ((float*)a.data.data())[i] = (float)v;
        else if (out_type == 2)
            ((double*)a.data.data())[i] = v;
        else if (out_type == 3)
            ((unsigned short*)a.data.data())[i] = float32_to_float16((float)v);
        else
            ((unsigned short*)a.data.data())[i] = float32_to_bfloat16((float)v);
    }

    op->outputs[0]->type = out_type;
    op->outputs[0]->shape = a.shape;
    op->params.clear();
}

// a dtype param is only foldable when it stays in the float family; an
// integer/bool window folds to truncated values pnnx cannot express here
static bool is_foldable_window_dtype(const std::map<std::string, Parameter>& captured_params)
{
    for (const auto& x : captured_params)
    {
        std::string key = x.first;
        size_t dot = key.rfind('.');
        if (dot != std::string::npos)
            key = key.substr(dot + 1);
        if (key == "dtype" && x.second.type == 2)
        {
            const int t = x.second.i;
            if (t != 1 && t != 2 && t != 3 && t != 13) // f32/f64/f16/bf16
                return false;
        }
    }
    return true;
}

class torch_hann_window_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
2 1
torch.hann_window        op_0        0 1 out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return is_foldable_window_dtype(captured_params);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        fold_window(op, captured_params, false);
    }
};

class torch_hamming_window_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
2 1
torch.hamming_window     op_0        0 1 out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return is_foldable_window_dtype(captured_params);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        fold_window(op, captured_params, true);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_hann_window_fold, 30)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_hamming_window_fold, 30)

} // namespace pnnx
