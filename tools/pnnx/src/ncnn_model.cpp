// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "ncnn_model.h"

#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>

namespace pnnx {

namespace {

static const uint32_t NCNN_WEIGHT_FP16 = 0x01306B47;
static const uint32_t NCNN_WEIGHT_BF16 = 0x01348B83;
static const uint32_t NCNN_WEIGHT_INT8 = 0x000D4B38;
static const uint32_t NCNN_WEIGHT_RAW_EXTRA_SCALE = 0x0002C056;

static size_t align4(size_t n)
{
    return (n + 3) & ~(size_t)3;
}

static uint16_t read_u16(const char* p)
{
    return (uint16_t)((uint8_t)p[0] | ((uint16_t)(uint8_t)p[1] << 8));
}

static uint32_t read_u32(const char* p)
{
    return (uint32_t)((uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24));
}

static float read_f32(const char* p)
{
    const uint32_t bits = read_u32(p);
    float value = 0.f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void append_bytes(std::vector<char>& dst, const char* src, size_t n)
{
    dst.insert(dst.end(), src, src + n);
}

static std::vector<std::string> split_tokens(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string token;
    bool quoted = false;
    bool escaped = false;

    for (char ch : line)
    {
        if (escaped)
        {
            token.push_back(ch);
            escaped = false;
            continue;
        }

        if (quoted && ch == '\\')
        {
            escaped = true;
            continue;
        }

        if (ch == '"')
        {
            quoted = !quoted;
            token.push_back(ch);
            continue;
        }

        if (!quoted && std::isspace((unsigned char)ch))
        {
            if (!token.empty())
            {
                tokens.push_back(token);
                token.clear();
            }
            continue;
        }

        token.push_back(ch);
    }

    if (!token.empty())
        tokens.push_back(token);

    return tokens;
}

static bool parse_integer(const std::string& value, int& out)
{
    if (value.empty())
        return false;

    char* end = 0;
    long v = std::strtol(value.c_str(), &end, 10);
    if (*end != '\0')
        return false;

    out = (int)v;
    return true;
}

static bool parse_float(const std::string& value, float& out)
{
    if (value.empty())
        return false;

    char* end = 0;
    float v = std::strtof(value.c_str(), &end);
    if (*end != '\0')
        return false;

    out = v;
    return true;
}

static Parameter parse_ncnn_parameter(const std::string& value, bool array)
{
    if (array)
    {
        size_t comma = value.find(',');
        std::string countstr = comma == std::string::npos ? value : value.substr(0, comma);
        int count = 0;
        if (!parse_integer(countstr, count) || count < 0)
            return Parameter::parse_from_string(value);

        std::vector<std::string> elems;
        size_t begin = comma == std::string::npos ? value.size() : comma + 1;
        while (begin <= value.size())
        {
            size_t end = value.find(',', begin);
            elems.push_back(value.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos)
                break;
            begin = end + 1;
        }

        // ncnn encodes an empty array as "0" and otherwise stores the
        // element count followed by the values.
        if ((int)elems.size() < count)
            return Parameter::parse_from_string(value);

        bool floating = false;
        for (int i = 0; i < count; i++)
        {
            if (elems[i].find('.') != std::string::npos || elems[i].find('e') != std::string::npos || elems[i].find('E') != std::string::npos)
            {
                floating = true;
                break;
            }
        }

        if (floating)
        {
            std::vector<float> v;
            v.reserve(count);
            for (int i = 0; i < count; i++)
            {
                float f = 0.f;
                if (!parse_float(elems[i], f))
                    return Parameter::parse_from_string(value);
                v.push_back(f);
            }
            return Parameter(v);
        }

        std::vector<int> v;
        v.reserve(count);
        for (int i = 0; i < count; i++)
        {
            int x = 0;
            if (!parse_integer(elems[i], x))
                return Parameter::parse_from_string(value);
            v.push_back(x);
        }
        return Parameter(v);
    }

    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        return Parameter(value.substr(1, value.size() - 2));

    return Parameter::parse_from_string(value);
}

static void parse_param_token(const std::string& token, Operator* op)
{
    const size_t eq = token.find('=');
    if (eq == std::string::npos || eq == 0)
        return;

    std::string key = token.substr(0, eq);
    const std::string value = token.substr(eq + 1);

    int numeric_key = 0;
    if (!parse_integer(key, numeric_key))
    {
        op->params[key] = parse_ncnn_parameter(value, false);
        return;
    }

    // -23330 is ncnn's optional per-output shape hint, rather than a normal
    // layer parameter. Keep it under a private key so it cannot overwrite
    // parameter 0 when a layer line contains both forms.
    if (numeric_key == -23330)
    {
        op->params["__ncnn_shape_hint"] = parse_ncnn_parameter(value, true);
        return;
    }

    // -23300-id is the compact ncnn array encoding. Keep the positive
    // parameter id in the IR; the ncnn writer restores the compact form.
    bool array = numeric_key <= -23300;
    if (array)
        key = std::to_string(-23300 - numeric_key);

    op->params[key] = parse_ncnn_parameter(value, array);
}

static int parameter_i(const Operator* op, int id, int fallback = 0)
{
    const auto it = op->params.find(std::to_string(id));
    if (it == op->params.end())
        return fallback;

    if (it->second.type == 2)
        return it->second.i;
    if (it->second.type == 3)
        return (int)it->second.f;
    return fallback;
}


static void normalize_float_parameter(Operator* op, int id)
{
    const std::string key = std::to_string(id);
    const auto it = op->params.find(key);
    if (it == op->params.end() || it->second.type != 2)
        return;

    it->second = Parameter((float)it->second.i);
}

static void normalize_float_parameters(Operator* op, std::initializer_list<int> ids)
{
    for (int id : ids)
        normalize_float_parameter(op, id);
}

static bool is_convolution_type(const std::string& type);
static bool is_deconvolution_type(const std::string& type);

static void normalize_native_float_parameters(Operator* op)
{
    // Param files commonly spell an exact integer-valued float as "0".  The
    // native graph rewrites access these fields through Parameter::f, while
    // ncnn's ParamDict accepts either scalar spelling at runtime.  Normalize
    // every float field emitted by ModelWriter, but keep integer fields such
    // as Deconvolution's output padding (also parameter 18) untouched.
    if (op->type == "BatchNorm")
        normalize_float_parameters(op, {1});
    else if (op->type == "BinaryOp")
        normalize_float_parameters(op, {2});
    else if (op->type == "CELU")
        normalize_float_parameters(op, {0});
    else if (op->type == "Clip")
        normalize_float_parameters(op, {0, 1});
    else if (is_convolution_type(op->type))
        normalize_float_parameters(op, {18});
    else if (op->type == "DetectionOutput")
        normalize_float_parameters(op, {1, 4, 5, 6, 7, 8});
    else if (op->type == "Dropout")
        normalize_float_parameters(op, {0});
    else if (op->type == "ELU")
        normalize_float_parameters(op, {0});
    else if (op->type == "Exp" || op->type == "Log")
        normalize_float_parameters(op, {0, 1, 2});
    else if (op->type == "Gemm")
        normalize_float_parameters(op, {0, 1});
    else if (op->type == "GroupNorm")
        normalize_float_parameters(op, {2});
    else if (op->type == "HardSigmoid" || op->type == "HardSwish")
        normalize_float_parameters(op, {0, 1});
    else if (op->type == "InstanceNorm" || op->type == "LayerNorm")
        normalize_float_parameters(op, {1});
    else if (op->type == "Interp")
        normalize_float_parameters(op, {1, 2});
    else if (op->type == "LRN")
        normalize_float_parameters(op, {2, 3, 4});
    else if (op->type == "MultiHeadAttention" || op->type == "SDPA")
        normalize_float_parameters(op, {6});
    else if (op->type == "MVN" || op->type == "Normalize")
        normalize_float_parameters(op, {2});
    else if (op->type == "Padding")
        normalize_float_parameters(op, {5});
    else if (op->type == "Power")
        normalize_float_parameters(op, {0, 1, 2});
    else if (op->type == "PriorBox")
        normalize_float_parameters(op, {3, 4, 5, 6, 11, 12, 13});
    else if (op->type == "Proposal" || op->type == "PSROIPooling")
        normalize_float_parameters(op, op->type == "Proposal" ? std::initializer_list<int>{4} : std::initializer_list<int>{2});
    else if (op->type == "ReLU")
        normalize_float_parameters(op, {0});
    else if (op->type == "Reduction")
        normalize_float_parameters(op, {2});
    else if (op->type == "RMSNorm")
        normalize_float_parameters(op, {1});
    else if (op->type == "ROIAlign" || op->type == "ROIPooling")
        normalize_float_parameters(op, {2});
    else if (op->type == "Threshold")
        normalize_float_parameters(op, {0});
    else if (op->type == "Unfold")
        normalize_float_parameters(op, {18});
    else if (op->type == "YoloDetectionOutput" || op->type == "Yolov3DetectionOutput")
        normalize_float_parameters(op, {2, 3});
}

// Keep this in sync with ModelWriter's weight-block quantization validation.
// A quantize term in the 400+ range is not automatically a packed weight
// tensor: the format and block-size subfields must be valid first.  Gemm
// additionally validates transB when loading the layer, but it still reads
// the ordinary tagged matrix and the block scales for transB == 0.
static bool is_weight_block_quantize_term(int quantize_term, int& weight_bits, int& block_size, bool& has_input_scale)
{
    weight_bits = quantize_term / 100;
    const int format_code = quantize_term % 100 / 10;
    const int block_size_code = quantize_term % 10;
    if ((weight_bits != 4 && weight_bits != 6 && weight_bits != 8) || (format_code != 0 && format_code != 1) || block_size_code < 0 || block_size_code > 2)
        return false;

    block_size = block_size_code == 0 ? 32 : block_size_code == 1 ? 64 : 128;
    has_input_scale = format_code == 1;
    return true;
}

static bool attribute_is_float(const Attribute& attr)
{
    return attr.type == 1 || attr.type == 2 || attr.type == 3;
}

static bool attribute_float_values(const Operator* op, int index, std::vector<float>& values)
{
    const auto it = op->attrs.find(std::to_string(index));
    if (it == op->attrs.end() || !attribute_is_float(it->second))
        return false;

    values = it->second.get_float32_data();
    return !values.empty();
}

enum WeightMode
{
    WeightTagged,
    WeightRawFloat,
    WeightRawBytes,
    // ModelBin type 0 consumes an ncnn tag and returns the decoded Mat to
    // MemoryData.  Keep the tag in the IR so the native output remains valid
    // for load_type=0, including int8 and legacy lookup-table data.
    WeightAutoRaw
};

struct WeightSpec
{
    int count;
    int type;
    WeightMode mode;
};

static void add_spec(std::vector<WeightSpec>& specs, int count, WeightMode mode, int type = 1)
{
    if (count > 0)
        specs.push_back({count, type, mode});
}

static bool is_convolution_type(const std::string& type)
{
    return type == "Convolution" || type == "Convolution1D" || type == "Convolution3D" || type == "ConvolutionDepthWise" || type == "ConvolutionDepthWise1D" || type == "ConvolutionDepthWise3D";
}

static bool is_deconvolution_type(const std::string& type)
{
    return type == "Deconvolution" || type == "Deconvolution1D" || type == "Deconvolution3D" || type == "DeconvolutionDepthWise" || type == "DeconvolutionDepthWise1D" || type == "DeconvolutionDepthWise3D";
}

static bool is_dynamic_weight(const Operator* op)
{
    if (op->type == "Deconvolution" || op->type == "Deconvolution1D" || op->type == "Deconvolution3D" || op->type == "DeconvolutionDepthWise" || op->type == "DeconvolutionDepthWise1D" || op->type == "DeconvolutionDepthWise3D")
        return parameter_i(op, 28, 0) != 0;
    return parameter_i(op, 19, 0) != 0;
}

static std::vector<WeightSpec> weight_specs(const Operator* op)
{
    std::vector<WeightSpec> specs;
    const std::string& type = op->type;

    if (type == "BatchNorm")
    {
        const int channels = parameter_i(op, 0);
        add_spec(specs, channels, WeightRawFloat);
        add_spec(specs, channels, WeightRawFloat);
        add_spec(specs, channels, WeightRawFloat);
        add_spec(specs, channels, WeightRawFloat);
    }
    else if (type == "Bias")
    {
        add_spec(specs, parameter_i(op, 0), WeightRawFloat);
    }
    else if (is_convolution_type(type) || is_deconvolution_type(type))
    {
        if (!is_dynamic_weight(op))
        {
            add_spec(specs, parameter_i(op, 6), WeightTagged);
            if (parameter_i(op, 5, 0))
                add_spec(specs, parameter_i(op, 0), WeightRawFloat);

            // Only the 2D convolution implementations serialize int8
            // scale tensors.  The 1D/3D and deconvolution variants have no
            // int8 scale fields in modelwriter, even when an unknown param
            // id happens to be present in a hand-written param file.
            const bool has_int8_scales = type == "Convolution" || type == "ConvolutionDepthWise";
            const int int8_scale_term = has_int8_scales ? parameter_i(op, 8, 0) : 0;
            if (has_int8_scales && int8_scale_term)
            {
                int scale_count = parameter_i(op, 0);
                if (type == "ConvolutionDepthWise" || type == "ConvolutionDepthWise1D" || type == "ConvolutionDepthWise3D" || type == "DeconvolutionDepthWise" || type == "DeconvolutionDepthWise1D" || type == "DeconvolutionDepthWise3D")
                {
                    const int group = parameter_i(op, 7, 1);
                    scale_count = int8_scale_term % 100 == 1 ? group : 1;
                }
                add_spec(specs, scale_count, WeightRawFloat);
                add_spec(specs, 1, WeightRawFloat);
                if (int8_scale_term > 100)
                    add_spec(specs, 1, WeightRawFloat);
            }
        }
    }
    else if (type == "DeformableConv2D")
    {
        add_spec(specs, parameter_i(op, 6), WeightTagged);
        if (parameter_i(op, 5, 0))
            add_spec(specs, parameter_i(op, 0), WeightRawFloat);
    }
    else if (type == "InnerProduct")
    {
        add_spec(specs, parameter_i(op, 2), WeightTagged);
        if (parameter_i(op, 1, 0))
            add_spec(specs, parameter_i(op, 0), WeightRawFloat);
        if (parameter_i(op, 8, 0))
        {
            add_spec(specs, parameter_i(op, 0), WeightRawFloat);
            add_spec(specs, 1, WeightRawFloat);
        }
    }
    else if (type == "Embed")
    {
        add_spec(specs, parameter_i(op, 3), WeightTagged);
        if (parameter_i(op, 2, 0))
            add_spec(specs, parameter_i(op, 0), WeightRawFloat);
        if (parameter_i(op, 18, 0))
            add_spec(specs, 1, WeightRawFloat);
    }
    else if (type == "Gemm")
    {
        const int m = parameter_i(op, 7);
        const int n = parameter_i(op, 8);
        const int k = parameter_i(op, 9);
        const int constant_a = parameter_i(op, 4);
        const int constant_b = parameter_i(op, 5);
        const int constant_c = parameter_i(op, 6);
        const int broadcast_c = parameter_i(op, 10);
        const int trans_b = parameter_i(op, 3);
        const int quantize_term = parameter_i(op, 18);

        if (constant_a)
            add_spec(specs, m * k, WeightTagged);

        // Gemm weight-block quantization is only valid for the transposed-B
        // storage layout.  With transB=0 ncnn still stores a tagged matrix
        // and only uses the quantization term for the runtime path.
        int weight_bits = 0;
        int block_size = 0;
        bool has_input_scale = false;
        const bool weight_block_quantize = is_weight_block_quantize_term(quantize_term, weight_bits, block_size, has_input_scale);
        if (constant_b)
        {
            if (weight_block_quantize && trans_b)
            {
                const int packed_k = (k * weight_bits + 7) / 8;
                add_spec(specs, packed_k * n, WeightRawBytes, 7);
            }
            else
            {
                add_spec(specs, k * n, WeightTagged);
            }
        }

        if (constant_c && broadcast_c != -1)
        {
            int c_count = 1;
            if (broadcast_c == 1 || broadcast_c == 2)
                c_count = m;
            else if (broadcast_c == 3)
                c_count = m * n;
            else if (broadcast_c == 4)
                c_count = n;
            add_spec(specs, c_count, WeightTagged);
        }

        if (weight_block_quantize && constant_b)
        {
            const int blocks = (k + block_size - 1) / block_size;
            add_spec(specs, blocks * n, WeightRawFloat);
            if (has_input_scale)
                add_spec(specs, k, WeightRawFloat);
        }
        else if (quantize_term && !weight_block_quantize)
        {
            if (constant_a)
                add_spec(specs, m, WeightRawFloat);
            if (constant_b)
                add_spec(specs, 1, WeightRawFloat);
        }
    }
    else if (type == "MultiHeadAttention")
    {
        const int embed_dim = parameter_i(op, 0);
        const int qdim = embed_dim > 0 ? parameter_i(op, 2) / embed_dim : 0;
        const int kdim = parameter_i(op, 3, embed_dim);
        const int vdim = parameter_i(op, 4, embed_dim);
        const int quantize_term = parameter_i(op, 18);
        int weight_bits = 0;
        int block_size = 0;
        bool has_input_scale = false;
        const bool weight_block_quantize = is_weight_block_quantize_term(quantize_term, weight_bits, block_size, has_input_scale);

        if (embed_dim > 0 && qdim > 0)
        {
            if (weight_block_quantize && (weight_bits == 4 || weight_bits == 6 || weight_bits == 8))
            {
                add_spec(specs, ((qdim * weight_bits + 7) / 8) * embed_dim, WeightRawBytes, 7);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, ((kdim * weight_bits + 7) / 8) * embed_dim, WeightRawBytes, 7);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, ((vdim * weight_bits + 7) / 8) * embed_dim, WeightRawBytes, 7);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, ((embed_dim * weight_bits + 7) / 8) * qdim, WeightRawBytes, 7);
                add_spec(specs, qdim, WeightRawFloat);

                const int qblocks = (qdim + block_size - 1) / block_size;
                const int kblocks = (kdim + block_size - 1) / block_size;
                const int vblocks = (vdim + block_size - 1) / block_size;
                const int outblocks = (embed_dim + block_size - 1) / block_size;
                add_spec(specs, qblocks * embed_dim, WeightRawFloat);
                add_spec(specs, kblocks * embed_dim, WeightRawFloat);
                add_spec(specs, vblocks * embed_dim, WeightRawFloat);
                add_spec(specs, outblocks * qdim, WeightRawFloat);
                if (has_input_scale)
                {
                    add_spec(specs, qdim, WeightRawFloat);
                    add_spec(specs, kdim, WeightRawFloat);
                    add_spec(specs, vdim, WeightRawFloat);
                    add_spec(specs, embed_dim, WeightRawFloat);
                }
            }
            else
            {
                add_spec(specs, embed_dim * qdim, WeightTagged);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, embed_dim * kdim, WeightTagged);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, embed_dim * vdim, WeightTagged);
                add_spec(specs, embed_dim, WeightRawFloat);
                add_spec(specs, qdim * embed_dim, WeightTagged);
                add_spec(specs, qdim, WeightRawFloat);
                if (quantize_term)
                {
                    add_spec(specs, embed_dim, WeightRawFloat);
                    add_spec(specs, embed_dim, WeightRawFloat);
                    add_spec(specs, embed_dim, WeightRawFloat);
                    add_spec(specs, 1, WeightRawFloat);
                }
            }
        }
    }
    else if (type == "MemoryData")
    {
        // MemoryData follows the same dimension precedence as the ncnn
        // layer: an all-zero shape creates its implicit one-element tensor
        // without consuming model-bin data.
        const int w = parameter_i(op, 0, 0);
        const int h = parameter_i(op, 1, 0);
        const int d = parameter_i(op, 11, 0);
        const int c = parameter_i(op, 2, 0);
        int count = 0;
        if (d != 0)
            count = w * h * d * c;
        else if (c != 0)
            count = w * h * c;
        else if (h != 0)
            count = w * h;
        else if (w != 0)
            count = w;

        const int load_type = parameter_i(op, 21, 1);
        if (load_type == 0)
            add_spec(specs, count, WeightAutoRaw);
        else if (load_type == 1)
            add_spec(specs, count, WeightRawFloat);
        else
            add_spec(specs, count, WeightRawBytes, load_type == 8 ? 8 : 7);
    }
    else if (type == "PReLU")
    {
        add_spec(specs, parameter_i(op, 0), WeightRawFloat);
    }
    else if (type == "Scale")
    {
        const int n = parameter_i(op, 0);
        if (n != -233)
        {
            add_spec(specs, n, WeightRawFloat);
            if (parameter_i(op, 1, 0))
                add_spec(specs, n, WeightRawFloat);
        }
    }
    else if (type == "GroupNorm")
    {
        if (parameter_i(op, 3, 1))
        {
            const int n = parameter_i(op, 1);
            add_spec(specs, n, WeightRawFloat);
            add_spec(specs, n, WeightRawFloat);
        }
    }
    else if (type == "InstanceNorm")
    {
        if (parameter_i(op, 2, 1))
        {
            const int n = parameter_i(op, 0);
            add_spec(specs, n, WeightRawFloat);
            add_spec(specs, n, WeightRawFloat);
        }
    }
    else if (type == "LayerNorm")
    {
        if (parameter_i(op, 2, 1))
        {
            const int n = parameter_i(op, 0);
            add_spec(specs, n, WeightRawFloat);
            add_spec(specs, n, WeightRawFloat);
        }
    }
    else if (type == "RMSNorm")
    {
        if (parameter_i(op, 2, 1))
            add_spec(specs, parameter_i(op, 0), WeightRawFloat);
    }
    else if (type == "Normalize")
    {
        add_spec(specs, parameter_i(op, 3), WeightRawFloat);
    }
    else if (type == "Quantize")
    {
        add_spec(specs, parameter_i(op, 0), WeightRawFloat);
    }
    else if (type == "Dequantize")
    {
        add_spec(specs, parameter_i(op, 0), WeightRawFloat);
        add_spec(specs, parameter_i(op, 1), WeightRawFloat);
    }
    else if (type == "Requantize")
    {
        add_spec(specs, parameter_i(op, 0), WeightRawFloat);
        add_spec(specs, parameter_i(op, 1), WeightRawFloat);
        add_spec(specs, parameter_i(op, 2), WeightRawFloat);
    }
    else if (type == "Padding")
    {
        add_spec(specs, parameter_i(op, 6), WeightRawFloat);
    }
    else if (type == "GRU" || type == "RNN")
    {
        const int num_output = parameter_i(op, 0);
        const int weight_data_size = parameter_i(op, 1);
        const int direction = parameter_i(op, 2);
        const int num_directions = direction == 2 ? 2 : 1;
        if (type == "GRU")
        {
            add_spec(specs, weight_data_size, WeightTagged);
            add_spec(specs, num_output * 4 * num_directions, WeightTagged);
            add_spec(specs, num_output * num_output * 3 * num_directions, WeightTagged);
            if (parameter_i(op, 8, 0))
            {
                add_spec(specs, num_output * 3 * num_directions, WeightRawFloat);
                add_spec(specs, num_output * 3 * num_directions, WeightRawFloat);
            }
        }
        else
        {
            add_spec(specs, weight_data_size, WeightTagged);
            add_spec(specs, num_output * num_directions, WeightTagged);
            add_spec(specs, num_output * num_output * num_directions, WeightTagged);
            if (parameter_i(op, 8, 0))
            {
                add_spec(specs, num_output * num_directions, WeightRawFloat);
                add_spec(specs, num_output * num_directions, WeightRawFloat);
            }
        }
    }
    else if (type == "LSTM")
    {
        const int num_output = parameter_i(op, 0);
        const int weight_data_size = parameter_i(op, 1);
        const int direction = parameter_i(op, 2);
        const int hidden_size = parameter_i(op, 3, num_output);
        const int num_directions = direction == 2 ? 2 : 1;
        add_spec(specs, weight_data_size, WeightTagged);
        add_spec(specs, hidden_size * 4 * num_directions, WeightTagged);
        add_spec(specs, num_output * hidden_size * 4 * num_directions, WeightTagged);
        if (num_output != hidden_size)
            add_spec(specs, hidden_size * num_output * num_directions, WeightTagged);
        if (parameter_i(op, 8, 0))
        {
            add_spec(specs, hidden_size * 4 * num_directions, WeightRawFloat);
            add_spec(specs, hidden_size * 4 * num_directions, WeightRawFloat);
        }
    }

    return specs;
}

static int attr_type_from_raw(int type)
{
    if (type == 1)
        return 1;
    if (type == 2)
        return 2;
    if (type == 3 || type == 4 || type == 6 || type == 7)
        return 7;
    if (type == 8)
        return 8;
    return 7;
}

static bool quantize_weight_to_int8(Attribute& marker, Attribute& value, const std::vector<float>& scales, int rows)
{
    if (marker.type != 0 || marker.data.size() != 4 || value.type != 1 || rows <= 0 || scales.empty())
        return false;

    const uint32_t tag = read_u32(marker.data.data());
    if (tag != 0 && tag != NCNN_WEIGHT_FP16 && tag != NCNN_WEIGHT_BF16 && tag != NCNN_WEIGHT_RAW_EXTRA_SCALE)
        return false;

    const std::vector<float> source = value.get_float32_data();
    if (source.empty() || (int)source.size() % rows != 0 || ((int)scales.size() != 1 && (int)scales.size() != rows))
        return false;

    const int row_size = (int)source.size() / rows;
    std::vector<char> quantized(source.size());
    for (int row = 0; row < rows; row++)
    {
        const float scale = scales.size() == 1 ? scales[0] : scales[row];
        for (int i = 0; i < row_size; i++)
        {
            float scaled = source[row * row_size + i] * scale;
            int rounded = (int)std::round(scaled);
            rounded = std::max(-127, std::min(127, rounded));
            quantized[row * row_size + i] = (char)rounded;
        }
    }

    const uint32_t int8_tag = NCNN_WEIGHT_INT8;
    marker.data.resize(4);
    marker.data[0] = (char)(int8_tag & 0xff);
    marker.data[1] = (char)((int8_tag >> 8) & 0xff);
    marker.data[2] = (char)((int8_tag >> 16) & 0xff);
    marker.data[3] = (char)((int8_tag >> 24) & 0xff);
    value.type = 7;
    value.data.swap(quantized);
    return true;
}

static void quantize_runtime_weights(Operator* op)
{
    // ncnn converts these tagged weights to int8 in Layer::load_model when an
    // int8 scale term is present. Mirroring those conversions here keeps the
    // pnnx optimizer's bin output compatible with ncnnoptimize.  Embed is
    // intentionally excluded: its ncnn implementation consumes the stored
    // element type directly and does not perform this conversion.
    if (op->type == "Convolution" && parameter_i(op, 8, 0) != 0 && parameter_i(op, 19, 0) == 0)
    {
        std::vector<float> scales;
        const int scale_index = 2 + parameter_i(op, 5, 0);
        if (attribute_float_values(op, scale_index, scales))
            quantize_weight_to_int8(op->attrs["0"], op->attrs["1"], scales, parameter_i(op, 0));
    }
    else if (op->type == "ConvolutionDepthWise" && parameter_i(op, 8, 0) != 0 && parameter_i(op, 19, 0) == 0)
    {
        std::vector<float> scales;
        const int scale_index = 2 + parameter_i(op, 5, 0);
        const int scale_term = parameter_i(op, 8, 0);
        const int group = parameter_i(op, 7, 1);
        const int rows = scale_term % 100 == 1 ? group : 1;
        if (attribute_float_values(op, scale_index, scales))
            quantize_weight_to_int8(op->attrs["0"], op->attrs["1"], scales, rows);
    }
    else if (op->type == "InnerProduct" && parameter_i(op, 8, 0) != 0)
    {
        std::vector<float> scales;
        const int scale_index = 2 + parameter_i(op, 1, 0);
        if (attribute_float_values(op, scale_index, scales))
            quantize_weight_to_int8(op->attrs["0"], op->attrs["1"], scales, parameter_i(op, 0));
    }
}

class BinReader
{
public:
    explicit BinReader(const std::string& path)
        : offset(0), valid(true), null_bin(path == "null")
    {
        if (null_bin)
            return;

        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is.good())
        {
            valid = false;
            return;
        }

        is.seekg(0, std::ios::end);
        std::streamoff size = is.tellg();
        is.seekg(0, std::ios::beg);
        if (size < 0)
        {
            valid = false;
            return;
        }
        data.resize((size_t)size);
        if (size > 0)
            is.read(data.data(), size);
        valid = is.good() || is.eof();
    }

    bool read_weight(const WeightSpec& spec, Attribute& marker, Attribute& value)
    {
        marker = Attribute();
        value = Attribute();

        const size_t elem_size = spec.type == 1 ? 4 : (spec.type == 2 || spec.type == 3 ? 2 : 1);
        uint32_t tag = 0;
        if (spec.mode == WeightTagged || spec.mode == WeightAutoRaw)
        {
            if (null_bin)
            {
                tag = 0;
            }
            else if (!read_u32_at(tag))
            {
                return false;
            }
            marker.type = 0;
            marker.data.resize(4);
            marker.data[0] = (char)(tag & 0xff);
            marker.data[1] = (char)((tag >> 8) & 0xff);
            marker.data[2] = (char)((tag >> 16) & 0xff);
            marker.data[3] = (char)((tag >> 24) & 0xff);
        }

        if (null_bin)
        {
            value.type = spec.mode == WeightRawBytes ? attr_type_from_raw(spec.type) : 1;
            value.shape = {spec.count};
            value.data.assign((size_t)spec.count * elem_size, 0);
            return true;
        }

        if (spec.mode == WeightTagged)
        {
            size_t payload_size = 0;
            int payload_type = 1;
            if (tag == NCNN_WEIGHT_FP16 || tag == NCNN_WEIGHT_BF16)
            {
                payload_size = align4((size_t)spec.count * 2);
                payload_type = 1;
            }
            else if (tag == NCNN_WEIGHT_INT8)
            {
                payload_size = align4((size_t)spec.count);
                payload_type = 7;
            }
            else if (tag == NCNN_WEIGHT_RAW_EXTRA_SCALE)
            {
                payload_size = (size_t)spec.count * 4;
                payload_type = 1;
            }
            else if (tag == 0)
            {
                payload_size = (size_t)spec.count * 4;
                payload_type = 1;
            }
            else
            {
                // Quantized ncnn weights contain a 256-entry float table
                // followed by one byte index per value.
                payload_size = 256 * sizeof(float) + align4((size_t)spec.count);
                payload_type = 0;
            }

            if (offset + payload_size > data.size())
                return false;

            if (tag == NCNN_WEIGHT_FP16)
            {
                value.type = 1;
                value.shape = {spec.count};
                value.data.resize((size_t)spec.count * sizeof(float));
                float* dst = (float*)value.data.data();
                for (int i = 0; i < spec.count; i++)
                    dst[i] = float16_to_float32(read_u16(data.data() + offset + (size_t)i * 2));
            }
            else if (tag == NCNN_WEIGHT_BF16)
            {
                value.type = 1;
                value.shape = {spec.count};
                value.data.resize((size_t)spec.count * sizeof(float));
                float* dst = (float*)value.data.data();
                for (int i = 0; i < spec.count; i++)
                {
                    uint32_t bits = (uint32_t)read_u16(data.data() + offset + (size_t)i * 2) << 16;
                    memcpy(dst + i, &bits, sizeof(bits));
                }
            }
            else if (tag == NCNN_WEIGHT_INT8)
            {
                value.type = 7;
                value.shape = {spec.count};
                value.data.assign(data.data() + offset, data.data() + offset + spec.count);
            }
            else if (payload_type == 1)
            {
                value.type = 1;
                value.shape = {spec.count};
                value.data.assign(data.data() + offset, data.data() + offset + (size_t)spec.count * sizeof(float));
            }
            else
            {
                // ncnn's legacy quantized weights store a 256-entry float
                // lookup table followed by one byte index per element.  The
                // ncnn loader expands this to fp32 before ModelWriter sees
                // it, so do the same here and normalize the marker to the
                // ordinary fp32 tag for a valid output bin.
                value.type = 1;
                value.shape = {spec.count};
                value.data.resize((size_t)spec.count * sizeof(float));
                float* dst = (float*)value.data.data();
                const char* table = data.data() + offset;
                const char* indices = table + 256 * sizeof(float);
                for (int i = 0; i < spec.count; i++)
                    dst[i] = read_f32(table + (size_t)(uint8_t)indices[i] * sizeof(float));

                marker.data[0] = 0;
                marker.data[1] = 0;
                marker.data[2] = 0;
                marker.data[3] = 0;
            }

            offset += payload_size;
            return true;
        }

        // WeightAutoRaw has already consumed the input tag.  ModelBin's
        // automatic loader returns fp32 for fp16/bf16/lookup-table weights,
        // int8 for the int8 tag, and fp32 for ordinary tag 0/extra-scale
        // data.  Normalize those decoded values while retaining the marker
        // so MemoryData load_type=0 can be serialized as a valid auto-tagged
        // weight block.
        if (spec.mode == WeightAutoRaw)
        {
            size_t payload_size = 0;
            if (tag == NCNN_WEIGHT_FP16 || tag == NCNN_WEIGHT_BF16)
                payload_size = align4((size_t)spec.count * 2);
            else if (tag == NCNN_WEIGHT_INT8)
                payload_size = align4((size_t)spec.count);
            else if (tag == NCNN_WEIGHT_RAW_EXTRA_SCALE || tag == 0)
                payload_size = (size_t)spec.count * 4;
            else
                payload_size = 256 * sizeof(float) + align4((size_t)spec.count);

            if (offset + payload_size > data.size())
                return false;

            if (tag == NCNN_WEIGHT_INT8)
            {
                value.type = 7;
                value.shape = {spec.count};
                value.data.assign(data.data() + offset, data.data() + offset + spec.count);
            }
            else
            {
                value.type = 1;
                value.shape = {spec.count};
                value.data.resize((size_t)spec.count * sizeof(float));
                float* dst = (float*)value.data.data();
                if (tag == NCNN_WEIGHT_FP16)
                {
                    for (int i = 0; i < spec.count; i++)
                        dst[i] = float16_to_float32(read_u16(data.data() + offset + (size_t)i * 2));
                }
                else if (tag == NCNN_WEIGHT_BF16)
                {
                    for (int i = 0; i < spec.count; i++)
                    {
                        uint32_t bits = (uint32_t)read_u16(data.data() + offset + (size_t)i * 2) << 16;
                        memcpy(dst + i, &bits, sizeof(bits));
                    }
                }
                else if (tag == NCNN_WEIGHT_RAW_EXTRA_SCALE || tag == 0)
                {
                    memcpy(dst, data.data() + offset, (size_t)spec.count * sizeof(float));
                }
                else
                {
                    const char* table = data.data() + offset;
                    const char* indices = table + 256 * sizeof(float);
                    for (int i = 0; i < spec.count; i++)
                        dst[i] = read_f32(table + (size_t)(uint8_t)indices[i] * sizeof(float));

                    // ModelBin has expanded the lookup table to fp32.  The
                    // output must use an ordinary raw tag rather than
                    // reusing the legacy quantized marker with fp32 bytes.
                    marker.data[0] = 0;
                    marker.data[1] = 0;
                    marker.data[2] = 0;
                    marker.data[3] = 0;
                }
            }

            offset += payload_size;
            return true;
        }

        const size_t payload_size = align4((size_t)spec.count * elem_size);
        if (offset + payload_size > data.size())
            return false;

        value.type = spec.mode == WeightRawBytes ? attr_type_from_raw(spec.type) : 1;
        value.shape = {spec.count};
        const size_t value_size = (size_t)spec.count * elem_size;
        value.data.assign(data.data() + offset, data.data() + offset + value_size);
        offset += payload_size;
        return true;
    }

    size_t remaining() const
    {
        return data.size() - std::min(offset, data.size());
    }

    bool is_valid() const
    {
        return valid;
    }

private:
    bool read_u32_at(uint32_t& v)
    {
        if (offset + 4 > data.size())
            return false;
        v = read_u32(data.data() + offset);
        offset += 4;
        return true;
    }

private:
    std::vector<char> data;
    size_t offset;
    bool valid;
    bool null_bin;
};

static std::string attr_key(int index)
{
    return std::to_string(index);
}

static bool write_f32(FILE* fp, const std::vector<float>& data)
{
    if (data.empty())
        return true;

    // ModelWriter flushes tiny denormals before writing fp32 storage.  Keep
    // the same normalization here without copying the common case where no
    // denormal values are present.
    bool has_denormal = false;
    for (float value : data)
    {
        if (value != 0.f && std::fabs(value) < 1e-30f)
        {
            has_denormal = true;
            break;
        }
    }

    if (!has_denormal)
    {
        if (fwrite(data.data(), sizeof(float), data.size(), fp) != data.size())
            return false;
    }
    else
    {
        std::vector<float> normalized = data;
        for (float& value : normalized)
        {
            if (value != 0.f && std::fabs(value) < 1e-30f)
                value = 0.f;
        }
        if (fwrite(normalized.data(), sizeof(float), normalized.size(), fp) != normalized.size())
            return false;
    }

    const size_t padding = align4(data.size() * sizeof(float)) - data.size() * sizeof(float);
    static const unsigned char zero[4] = {0, 0, 0, 0};
    return padding == 0 || fwrite(zero, 1, padding, fp) == padding;
}

static bool write_padded(FILE* fp, const std::vector<char>& data)
{
    if (!data.empty() && fwrite(data.data(), 1, data.size(), fp) != data.size())
        return false;
    const size_t padding = align4(data.size()) - data.size();
    static const unsigned char zero[4] = {0, 0, 0, 0};
    return padding == 0 || fwrite(zero, 1, padding, fp) == padding;
}

// x86 ncnn builds use F16C for ModelWriter's fp32 -> fp16 conversion.  The
// scalar helper in pnnx intentionally flushes values smaller than the normal
// fp16 range, while F16C preserves representable fp16 subnormals.  Keep the
// storage writer independent so optimized bins match the native writer on
// both paths (round-to-nearest-even, including the subnormal range).
static unsigned short float32_to_float16_storage(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));

    const unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    const unsigned int exponent = (bits >> 23) & 0xffu;
    const unsigned int significand = bits & 0x7fffffu;

    if (exponent == 0xffu)
        return (unsigned short)(sign | 0x7c00u | (significand ? 0x0200u : 0));

    const int new_exponent = (int)exponent - 127 + 15;
    if (new_exponent >= 31)
        return (unsigned short)(sign | 0x7c00u);

    if (new_exponent <= 0)
    {
        if (new_exponent < -10)
            return sign;

        const uint32_t mantissa = significand | 0x800000u;
        const int shift = 14 - new_exponent;
        const uint32_t truncated = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        uint32_t rounded = truncated + (remainder > halfway || (remainder == halfway && (truncated & 1u)) ? 1u : 0u);
        if (rounded >= 0x400u)
            return (unsigned short)(sign | 0x0400u);
        return (unsigned short)(sign | rounded);
    }

    uint32_t rounded_significand = significand >> 13;
    const uint32_t remainder = significand & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_significand & 1u)))
        rounded_significand++;

    int output_exponent = new_exponent;
    if (rounded_significand == 0x400u)
    {
        rounded_significand = 0;
        output_exponent++;
        if (output_exponent >= 31)
            return (unsigned short)(sign | 0x7c00u);
    }

    return (unsigned short)(sign | ((unsigned short)output_exponent << 10) | (unsigned short)rounded_significand);
}

static bool write_float16_data(FILE* fp, const std::vector<float>& data)
{
    std::vector<unsigned short> converted(data.size());
    for (size_t i = 0; i < data.size(); i++)
        converted[i] = float32_to_float16_storage(data[i]);

    if (!converted.empty() && fwrite(converted.data(), sizeof(unsigned short), converted.size(), fp) != converted.size())
        return false;

    const size_t bytes = converted.size() * sizeof(unsigned short);
    const size_t padding = align4(bytes) - bytes;
    static const unsigned char zero[4] = {0, 0, 0, 0};
    return padding == 0 || fwrite(zero, 1, padding, fp) == padding;
}

static bool write_bfloat16_data(FILE* fp, const std::vector<float>& data)
{
    std::vector<unsigned short> converted(data.size());
    for (size_t i = 0; i < data.size(); i++)
    {
        uint32_t bits = 0;
        memcpy(&bits, &data[i], sizeof(bits));
        // Match ncnn::float32_to_bfloat16: round to nearest with the same
        // tie handling as the native model writer instead of truncating the
        // low 16 mantissa bits.
        converted[i] = (unsigned short)((bits + 0x8000u) >> 16);
    }

    if (!converted.empty() && fwrite(converted.data(), sizeof(unsigned short), converted.size(), fp) != converted.size())
        return false;

    const size_t bytes = converted.size() * sizeof(unsigned short);
    const size_t padding = align4(bytes) - bytes;
    static const unsigned char zero[4] = {0, 0, 0, 0};
    return padding == 0 || fwrite(zero, 1, padding, fp) == padding;
}

static std::vector<int> sorted_numeric_keys(const std::map<std::string, Parameter>& params)
{
    std::vector<int> keys;
    for (const auto& it : params)
    {
        int key = 0;
        if (parse_integer(it.first, key))
            keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

static std::vector<int> sorted_attr_indices(const std::map<std::string, Attribute>& attrs)
{
    std::vector<int> indices;
    for (const auto& it : attrs)
    {
        int index = 0;
        if (parse_integer(it.first, index))
            indices.push_back(index);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

static void write_parameter(FILE* fp, int id, const Parameter& param)
{
    if (param.type == 2)
    {
        fprintf(fp, " %d=%d", id, param.i);
    }
    else if (param.type == 3)
    {
        fprintf(fp, " %d=%s", id, float_to_string(param.f).c_str());
    }
    else if (param.type == 4)
    {
        bool simple = !param.s.empty() && (std::isalpha((unsigned char)param.s[0]) || param.s[0] == '_');
        for (char ch : param.s)
            simple = simple && (std::isalnum((unsigned char)ch) || ch == '_');
        if (simple)
            fprintf(fp, " %d=%s", id, param.s.c_str());
        else
            fprintf(fp, " %d=\"%s\"", id, param.s.c_str());
    }
    else if (param.type == 5)
    {
        fprintf(fp, " -%d=%d", 23300 + id, (int)param.ai.size());
        for (int value : param.ai)
            fprintf(fp, ",%d", value);
    }
    else if (param.type == 6)
    {
        fprintf(fp, " -%d=%d", 23300 + id, (int)param.af.size());
        for (float value : param.af)
            fprintf(fp, ",%s", float_to_string(value).c_str());
    }
}

static void write_shape_hint(FILE* fp, const Operator* op)
{
    const auto it = op->params.find("__ncnn_shape_hint");
    if (it == op->params.end() || it->second.type != 5)
        return;

    fprintf(fp, " -23330=%d", (int)it->second.ai.size());
    for (int value : it->second.ai)
        fprintf(fp, ",%d", value);
}

static bool tag_is_storage(uint32_t tag)
{
    return tag == 0 || tag == NCNN_WEIGHT_FP16 || tag == NCNN_WEIGHT_BF16 || tag == NCNN_WEIGHT_RAW_EXTRA_SCALE;
}

static bool write_attributes(FILE* fp, const Operator* op, int storage_type)
{
    const std::vector<int> indices = sorted_attr_indices(op->attrs);
    for (size_t i = 0; i < indices.size(); i++)
    {
        const Attribute& attr = op->attrs.at(attr_key(indices[i]));
        if (attr.type == 0 && attr.data.size() == 4)
        {
            uint32_t tag = read_u32(attr.data.data());
            if (i + 1 >= indices.size())
            {
                fprintf(stderr, "ncnn weight marker without payload in %s\n", op->name.c_str());
                return false;
            }

            const Attribute& data = op->attrs.at(attr_key(indices[++i]));
            if (storage_type >= 0 && tag_is_storage(tag) && data.type == 1)
            {
                uint32_t output_tag = storage_type == 1 ? NCNN_WEIGHT_FP16 : storage_type == 2 ? NCNN_WEIGHT_BF16 : 0;
                if (fwrite(&output_tag, sizeof(output_tag), 1, fp) != 1)
                    return false;
                const std::vector<float> values = data.get_float32_data();
                if (storage_type == 1)
                {
                    if (!write_float16_data(fp, values))
                        return false;
                }
                else if (storage_type == 2)
                {
                    if (!write_bfloat16_data(fp, values))
                        return false;
                }
                else
                {
                    if (!write_f32(fp, values))
                        return false;
                }
            }
            else
            {
                if (fwrite(attr.data.data(), 1, 4, fp) != 4)
                    return false;
                if (!write_padded(fp, data.data))
                    return false;
            }
            continue;
        }

        if (attr.type == 1)
        {
            if (!write_f32(fp, attr.get_float32_data()))
                return false;
        }
        else if (!write_padded(fp, attr.data))
            return false;
    }

    return true;
}

static void apply_shape_hints(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        const auto it = op->params.find("__ncnn_shape_hint");
        if (it == op->params.end())
            continue;

        const Parameter& hints = it->second;
        if (hints.type != 5 || op->outputs.empty())
        {
            op->params.erase(it);
            continue;
        }

        // ncnn stores one [dims,w,h,d,c] record per top.  Older files may
        // omit the unused fifth field for tensors with rank <= 3, which the
        // native loader also accepts.
        if (hints.ai.size() % op->outputs.size() != 0)
        {
            fprintf(stderr, "invalid ncnn shape hints at %s\n", op->name.c_str());
            op->params.erase(it);
            continue;
        }

        const size_t step = hints.ai.size() / op->outputs.size();
        if (step != 4 && step != 5)
        {
            fprintf(stderr, "invalid ncnn shape hint stride at %s\n", op->name.c_str());
            op->params.erase(it);
            continue;
        }

        for (size_t i = 0; i < op->outputs.size(); i++)
        {
            const size_t offset = i * step;
            const int dims = hints.ai[offset];
            if (dims <= 0)
                continue;
            if (dims > 4 || (dims == 4 && step != 5) || offset + (size_t)dims >= hints.ai.size())
            {
                fprintf(stderr, "invalid ncnn shape hint rank at %s\n", op->name.c_str());
                continue;
            }

            const int w = hints.ai[offset + 1];
            const int h = dims >= 2 ? hints.ai[offset + 2] : 1;
            const int d = dims == 4 ? hints.ai[offset + 3] : 1;
            const int c = dims >= 3 ? hints.ai[offset + step - 1] : 1;

            if (dims == 1)
                op->outputs[i]->shape = {w};
            else if (dims == 2)
                op->outputs[i]->shape = {w, h};
            else if (dims == 3)
                op->outputs[i]->shape = {w, h, c};
            else
                op->outputs[i]->shape = {w, h, d, c};
        }

        // A pass may replace the producer or redirect its output.  The
        // original per-layer hint would then describe the wrong tensor, so
        // retain the parsed shape on the operand and drop the private token.
        op->params.erase(it);
    }
}

} // namespace

bool is_ncnn_param_file(const std::string& path)
{
    std::ifstream is(path.c_str());
    if (!is.good())
        return false;

    std::string line;
    if (!std::getline(is, line))
        return false;

    std::istringstream iss(line);
    int magic = 0;
    iss >> magic;
    if (magic != 7767517)
        return false;

    // PNNX and ncnn share the same magic number. Distinguish the native
    // ncnn text form from a PNNX graph before using the positional optimizer
    // command implicitly.
    if (!std::getline(is, line) || !std::getline(is, line))
        return false;
    const std::vector<std::string> tokens = split_tokens(line);
    return tokens.empty() || (tokens[0] != "pnnx.Input" && tokens[0] != "pnnx.Output");
}

int load_ncnn_model(const std::string& parampath, const std::string& binpath, Graph& graph)
{
    std::ifstream is(parampath.c_str(), std::ios::in | std::ios::binary);
    if (!is.good())
    {
        fprintf(stderr, "open %s failed\n", parampath.c_str());
        return -1;
    }

    std::string line;
    if (!std::getline(is, line))
        return -1;
    {
        std::istringstream iss(line);
        int magic = 0;
        iss >> magic;
        if (magic != 7767517)
        {
            fprintf(stderr, "%s is not an ncnn param file\n", parampath.c_str());
            return -1;
        }
    }

    if (!std::getline(is, line))
        return -1;
    int operator_count = 0;
    int operand_count = 0;
    {
        std::istringstream iss(line);
        iss >> operator_count >> operand_count;
    }
    if (operator_count < 0 || operand_count < 0)
        return -1;

    std::vector<std::vector<std::string> > token_lines;
    token_lines.reserve(operator_count);
    for (int i = 0; i < operator_count; i++)
    {
        if (!std::getline(is, line))
            return -1;
        token_lines.push_back(split_tokens(line));
    }

    BinReader bin(binpath);
    if (!bin.is_valid())
    {
        fprintf(stderr, "open %s failed\n", binpath.c_str());
        return -1;
    }

    // ncnn permits in-place layers where an output reuses a blob name. Keep
    // the most recent operand for each name while retaining distinct IR
    // operands for the input/output edge of the in-place layer.
    std::map<std::string, Operand*> current_operands;

    for (size_t op_index = 0; op_index < token_lines.size(); op_index++)
    {
        const std::vector<std::string>& tokens = token_lines[op_index];
        if (tokens.size() < 4)
            return -1;

        const std::string& type = tokens[0];
        const std::string& name = tokens[1];
        int input_count = 0;
        int output_count = 0;
        if (!parse_integer(tokens[2], input_count) || !parse_integer(tokens[3], output_count) || input_count < 0 || output_count < 0)
            return -1;
        if (tokens.size() < (size_t)(4 + input_count + output_count))
            return -1;

        Operator* op = graph.new_operator(type, name);
        // Keep the source position private to the IR so cut ranges can still
        // refer to a layer that is removed by a later optimization pass.
        op->params["__ncnn_original_index"] = Parameter((int)op_index);
        size_t token_index = 4;
        for (int i = 0; i < input_count; i++)
        {
            const std::string& operand_name = tokens[token_index++];
            Operand* operand = 0;
            const auto current_it = current_operands.find(operand_name);
            if (current_it != current_operands.end())
                operand = current_it->second;
            else
            {
                operand = graph.new_operand(operand_name);
                operand->producer = 0;
                current_operands[operand_name] = operand;
            }
            operand->consumers.push_back(op);
            op->inputs.push_back(operand);
        }
        for (int i = 0; i < output_count; i++)
        {
            const std::string& operand_name = tokens[token_index++];
            Operand* operand = graph.new_operand(operand_name);
            current_operands[operand_name] = operand;
            operand->producer = op;
            op->outputs.push_back(operand);
        }

        for (; token_index < tokens.size(); token_index++)
            parse_param_token(tokens[token_index], op);

        normalize_native_float_parameters(op);

        int attr_index = 0;
        for (const WeightSpec& spec : weight_specs(op))
        {
            Attribute marker;
            Attribute value;
            if (!bin.read_weight(spec, marker, value))
            {
                fprintf(stderr, "read ncnn weight failed at %s %s (remaining %zu bytes)\n", type.c_str(), name.c_str(), bin.remaining());
                return -1;
            }

            if (spec.mode == WeightTagged || spec.mode == WeightAutoRaw)
            {
                op->attrs[attr_key(attr_index++)] = marker;
            }
            op->attrs[attr_key(attr_index++)] = value;
        }

        quantize_runtime_weights(op);
    }

    if (bin.remaining() != 0)
    {
        fprintf(stderr, "ncnn bin has %zu unconsumed bytes\n", bin.remaining());
        return -1;
    }

    return 0;
}

int prepare_ncnn_graph_for_passes(Graph& graph)
{
    apply_shape_hints(graph);

    // Native ncnn input layers are graph roots.  The ncnn passes use the
    // pnnx.Input marker to stop recursive pattern matching at those roots.
    for (Operator* op : graph.ops)
    {
        if (op->type == "Input" && op->inputs.empty() && !op->outputs.empty())
            op->type = "pnnx.Input";
    }

    // Hand-written native param files do not always contain an explicit
    // Input layer.  Turn any producerless blob that feeds a layer into a
    // temporary pnnx.Input so the graph remains matchable.
    std::vector<Operand*> producerless;
    for (Operand* operand : graph.operands)
    {
        if (!operand->producer && !operand->consumers.empty())
            producerless.push_back(operand);
    }

    int input_index = 0;
    for (Operand* operand : producerless)
    {
        if (operand->producer || operand->consumers.empty())
            continue;

        const Operator* first_consumer = operand->consumers[0];
        size_t first_index = graph.ops.size();
        for (const Operator* consumer : operand->consumers)
        {
            const auto it = std::find(graph.ops.begin(), graph.ops.end(), consumer);
            if (it != graph.ops.end())
                first_index = std::min(first_index, (size_t)(it - graph.ops.begin()));
        }

        Operator* input = 0;
        if (first_index < graph.ops.size())
            input = graph.new_operator_before("pnnx.Input", "__pnnx_native_input_" + std::to_string(input_index++), graph.ops[first_index]);
        else
            input = graph.new_operator("pnnx.Input", "__pnnx_native_input_" + std::to_string(input_index++));

        input->outputs.push_back(operand);
        operand->producer = input;
    }

    // Pattern rewriters identify the graph boundary through pnnx.Output.  A
    // boundary is needed for every native blob with no downstream consumer.
    std::vector<Operand*> terminal;
    for (Operand* operand : graph.operands)
    {
        if (operand->producer && operand->consumers.empty())
            terminal.push_back(operand);
    }

    int output_index = 0;
    for (Operand* operand : terminal)
    {
        if (!operand->producer || !operand->consumers.empty())
            continue;

        Operator* output = graph.new_operator_after("pnnx.Output", "__pnnx_native_output_" + std::to_string(output_index++), operand->producer);
        output->inputs.push_back(operand);
        operand->consumers.push_back(output);
    }

    return 0;
}

int save_ncnn_model(const Graph& graph, const std::string& parampath, const std::string& binpath, int storage_type, const std::string& cutstart, const std::string& cutend, int cutstart_index, int cutend_index)
{
    const bool use_original_indices = cutstart_index >= 0 || cutend_index >= 0;
    int first = 0;
    int last = use_original_indices ? INT_MAX : (int)graph.ops.size() - 1;
    if (cutstart_index >= 0)
    {
        first = cutstart_index;
        fprintf(stderr, "cutstart layer %d:%s\n", first, cutstart.c_str());
    }
    else if (!cutstart.empty())
    {
        first = -1;
        for (int i = 0; i < (int)graph.ops.size(); i++)
        {
            if (graph.ops[i]->name == cutstart)
            {
                first = i;
                break;
            }
        }
        if (first < 0)
        {
            fprintf(stderr, "not find target cutstart layer %s\n", cutstart.c_str());
            return -1;
        }
        fprintf(stderr, "cutstart layer %d:%s\n", first, cutstart.c_str());
    }
    if (cutend_index >= 0)
    {
        last = cutend_index;
        fprintf(stderr, "cutend layer %d:%s\n", last, cutend.c_str());
    }
    else if (!cutend.empty())
    {
        last = -1;
        for (int i = 0; i < (int)graph.ops.size(); i++)
        {
            if (graph.ops[i]->name == cutend)
            {
                last = i;
                break;
            }
        }
        if (last < 0)
        {
            fprintf(stderr, "not find target cutend layer %s\n", cutend.c_str());
            return -1;
        }
        fprintf(stderr, "cutend layer %d:%s\n", last, cutend.c_str());
    }
    if (first > last)
    {
        fprintf(stderr, "invalid ncnn cut range %d..%d\n", first, last);
        return -1;
    }

    FILE* paramfp = fopen(parampath.c_str(), "wb");
    if (!paramfp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath.c_str());
        return -1;
    }
    FILE* binfp = fopen(binpath.c_str(), "wb");
    if (!binfp)
    {
        fclose(paramfp);
        fprintf(stderr, "fopen %s failed\n", binpath.c_str());
        return -1;
    }

    auto op_index_for_cut = [use_original_indices](const Operator* op, int index) {
        if (!use_original_indices)
            return index;

        const auto it = op->params.find("__ncnn_original_index");
        if (it != op->params.end() && it->second.type == 2)
            return it->second.i;
        return index;
    };

    // Build the exact operator subset that will be emitted.  The native ncnn
    // loader consumes the header count literally, so cut ranges must affect
    // both the header and the layer lines (and their blob count).
    std::vector<const Operator*> selected_ops;
    std::set<std::string> blob_names;
    for (int i = 0; i < (int)graph.ops.size(); i++)
    {
        const Operator* op = graph.ops[i];
        const int cut_index = op_index_for_cut(op, i);
        if (cut_index < first || cut_index > last)
            continue;

        selected_ops.push_back(op);
        for (const Operand* operand : op->inputs)
            blob_names.insert(operand->name);
        for (const Operand* operand : op->outputs)
            blob_names.insert(operand->name);
    }

    fprintf(paramfp, "7767517\n%d %d\n", (int)selected_ops.size(), (int)blob_names.size());

    for (const Operator* op : selected_ops)
    {
        fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());
        for (const Operand* operand : op->inputs)
            fprintf(paramfp, " %s", operand->name.c_str());
        for (const Operand* operand : op->outputs)
            fprintf(paramfp, " %s", operand->name.c_str());

        write_shape_hint(paramfp, op);
        for (int id : sorted_numeric_keys(op->params))
        {
            const auto it = op->params.find(std::to_string(id));
            if (it != op->params.end())
                write_parameter(paramfp, id, it->second);
        }
        fprintf(paramfp, "\n");

        if (!write_attributes(binfp, op, storage_type))
        {
            fclose(paramfp);
            fclose(binfp);
            return -1;
        }
    }

    fclose(paramfp);
    fclose(binfp);
    return 0;
}

} // namespace pnnx
