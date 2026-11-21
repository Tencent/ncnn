// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "datareader.h"
#include "layer_type.h"
#include "net.h"
#include "testutil.h"

// memory readers without a length cannot detect truncated binary input
class BoundedNetReader : public ncnn::DataReader
{
public:
    BoundedNetReader(const unsigned char* p, size_t size)
        : data(p), remaining(size)
    {
    }

    virtual size_t read(void* buf, size_t size) const
    {
        if (size > remaining) size = remaining;
        if (size != 0)
        {
            memcpy(buf, data, size);
            data += size;
            remaining -= size;
        }
        return size;
    }

private:
    mutable const unsigned char* data;
    mutable size_t remaining;
};

#if NCNN_VALIDATION
static bool empty_net(const ncnn::Net& net)
{
    return net.layers().empty() && net.blobs().empty() && net.input_indexes().empty() && net.output_indexes().empty()
           && net.input_names().empty() && net.output_names().empty();
}

static int load_binary(ncnn::Net& net, const unsigned char* data, size_t size)
{
    BoundedNetReader dr(data, size);
    return net.load_param_bin(dr);
}

static void append_int(std::vector<unsigned char>& data, int value)
{
    const unsigned int v = (unsigned int)value;
    for (int i = 0; i < 4; i++)
        data.push_back((unsigned char)(v >> (i * 8)));
}

static std::vector<unsigned char> binary_header(int layers, int blobs)
{
    std::vector<unsigned char> data;
    append_int(data, 7767517);
    append_int(data, layers);
    append_int(data, blobs);
    return data;
}

static void append_layer_header(std::vector<unsigned char>& data, int type, int bottoms, int tops)
{
    append_int(data, type);
    append_int(data, bottoms);
    append_int(data, tops);
}

// built-in layers whose forward() indexes bottom_blobs[N] or writes top_blobs[0]
// must be rejected at load time when the model declares fewer blobs
static int check_builtin_arity(int binary, const char* type, int typeindex, int param_count, const int* params, int bottom_count, int top_count, int expected_ret, const char* extra_text = "", const std::vector<unsigned char>& extra_binary = std::vector<unsigned char>())
{
    ncnn::Net net;
    int ret;
    if (binary)
    {
        std::vector<unsigned char> data = binary_header(bottom_count + 1, bottom_count + top_count);
        for (int i = 0; i < bottom_count; i++)
        {
            append_layer_header(data, ncnn::LayerType::Input, 0, 1);
            append_int(data, i);
            append_int(data, -233);
        }
        append_layer_header(data, typeindex, bottom_count, top_count);
        for (int i = 0; i < bottom_count + top_count; i++)
            append_int(data, i);
        for (int i = 0; i < param_count * 2; i++)
            append_int(data, params[i]);
        data.insert(data.end(), extra_binary.begin(), extra_binary.end());
        append_int(data, -233);
        ret = load_binary(net, &data[0], data.size());
    }
    else
    {
        std::string text = "7767517\n";
        char buf[64];
        snprintf(buf, sizeof(buf), "%d %d\n", bottom_count + 1, bottom_count + top_count);
        text += buf;
        for (int i = 0; i < bottom_count; i++)
        {
            snprintf(buf, sizeof(buf), "Input in%d 0 1 b%d\n", i, i);
            text += buf;
        }
        snprintf(buf, sizeof(buf), "%s op %d %d", type, bottom_count, top_count);
        text += buf;
        for (int i = 0; i < bottom_count; i++)
        {
            snprintf(buf, sizeof(buf), " b%d", i);
            text += buf;
        }
        for (int i = 0; i < top_count; i++)
        {
            snprintf(buf, sizeof(buf), " t%d", i);
            text += buf;
        }
        for (int i = 0; i < param_count; i++)
        {
            snprintf(buf, sizeof(buf), " %d=%d", params[i * 2], params[i * 2 + 1]);
            text += buf;
        }
        text += extra_text;
        text += "\n";
        ret = net.load_param_mem(text.c_str());
    }

    if (ret != expected_ret || (ret != 0 && !empty_net(net)))
    {
        fprintf(stderr, "test_scale_arity builtin arity failed binary=%d type=%s bottoms=%d tops=%d ret=%d expected=%d\n", binary, type, bottom_count, top_count, ret, expected_ret);
        return -1;
    }

    return 0;
}

static int test_builtin_layer_arity()
{
    struct Case
    {
        const char* type;
        int typeindex;
        int min_bottom;
        int min_top;
        int param_count;
        int params[10];
    };
    const Case cases[] = {
        {"BinaryOp", ncnn::LayerType::BinaryOp, 2, 1},
        {"CopyTo", ncnn::LayerType::CopyTo, 2, 1},
        {"DeformableConv2D", ncnn::LayerType::DeformableConv2D, 2, 1, 3, {0, 1, 1, 1, 6, 1}},
        {"Eltwise", ncnn::LayerType::Eltwise, 2, 1},
        {"Gemm", ncnn::LayerType::Gemm, 2, 1},
        {"GridSample", ncnn::LayerType::GridSample, 2, 1},
        {"MatMul", ncnn::LayerType::MatMul, 2, 1},
        {"PSROIPooling", ncnn::LayerType::PSROIPooling, 2, 1},
        {"ROIAlign", ncnn::LayerType::ROIAlign, 2, 1, 2, {0, 2, 1, 2}},
        {"ROIPooling", ncnn::LayerType::ROIPooling, 2, 1},
        {"Scale", ncnn::LayerType::Scale, 2, 1, 1, {0, -233}},
        {"Interp", ncnn::LayerType::Interp, 2, 1, 2, {0, 1, 5, 1}},
        {"DetectionOutput", ncnn::LayerType::DetectionOutput, 3, 1},
        {"Proposal", ncnn::LayerType::Proposal, 3, 1},
        {"RotaryEmbed", ncnn::LayerType::RotaryEmbed, 3, 1},
        {"SDPA", ncnn::LayerType::SDPA, 3, 1},
        {"SDPA", ncnn::LayerType::SDPA, 4, 1, 1, {5, 1}},
        {"SDPA", ncnn::LayerType::SDPA, 5, 3, 1, {7, 1}},
        {"SDPA", ncnn::LayerType::SDPA, 6, 3, 2, {5, 1, 7, 1}},
        {"MultiHeadAttention", ncnn::LayerType::MultiHeadAttention, 1, 1, 3, {0, 1, 1, 1, 2, 1}},
        {"MultiHeadAttention", ncnn::LayerType::MultiHeadAttention, 2, 1, 4, {0, 1, 1, 1, 2, 1, 5, 1}},
        {"MultiHeadAttention", ncnn::LayerType::MultiHeadAttention, 3, 3, 4, {0, 1, 1, 1, 2, 1, 7, 1}},
        {"MultiHeadAttention", ncnn::LayerType::MultiHeadAttention, 4, 3, 5, {0, 1, 1, 1, 2, 1, 5, 1, 7, 1}},
        {"Convolution", ncnn::LayerType::Convolution, 2, 1, 1, {19, 1}},
        {"Convolution", ncnn::LayerType::Convolution, 3, 1, 2, {5, 1, 19, 1}},
        {"Crop", ncnn::LayerType::Crop, 2, 1},
        {"Concat", ncnn::LayerType::Concat, 1, 1},
        {"Slice", ncnn::LayerType::Slice, 1, 1},
        {"Split", ncnn::LayerType::Split, 1, 1},
    };
    for (int binary = 0; binary < 2; binary++)
    {
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        {
            const Case& c = cases[i];
            if (check_builtin_arity(binary, c.type, c.typeindex, c.param_count, c.params, c.min_bottom, c.min_top, 0)
                    || check_builtin_arity(binary, c.type, c.typeindex, c.param_count, c.params, c.min_bottom - 1, c.min_top, -1)
                    || check_builtin_arity(binary, c.type, c.typeindex, c.param_count, c.params, 0, c.min_top, -1)
                    || check_builtin_arity(binary, c.type, c.typeindex, c.param_count, c.params, c.min_bottom, c.min_top - 1, -1)
                    || check_builtin_arity(binary, c.type, c.typeindex, c.param_count, c.params, c.min_bottom, 0, -1))
                return -1;
        }

        // one_blob_only forms of the same layer types keep loading with a single bottom blob
        const int binaryop_scalar[] = {0, 0, 1, 1};
        const int crop_roi[] = {3, 1};
        const int scale_static[] = {0, 1};
        if (check_builtin_arity(binary, "BinaryOp", ncnn::LayerType::BinaryOp, 2, binaryop_scalar, 1, 1, 0)
                || check_builtin_arity(binary, "Scale", ncnn::LayerType::Scale, 1, scale_static, 1, 1, 0)
                || check_builtin_arity(binary, "Crop", ncnn::LayerType::Crop, 1, crop_roi, 1, 1, 0))
            return -1;

        // a dynamic Scale loads no bias_data, so bias_term is rejected whatever the blob count
        const int scale_dynamic_bias[] = {0, -233, 1, 1};
        if (check_builtin_arity(binary, "Scale", ncnn::LayerType::Scale, 2, scale_dynamic_bias, 2, 1, -1)
                || check_builtin_arity(binary, "Scale", ncnn::LayerType::Scale, 2, scale_dynamic_bias, 1, 1, -1))
            return -1;
    }

    return 0;
}

static int test_builtin_expression_arity()
{
    const char* expressions[] = {"4,4", "0w,0h", "1w,1h", "2w,0h", "9w,0h"};
    const int required[] = {1, 1, 2, 3, 10};
    for (int binary = 0; binary < 2; binary++)
    {
        for (int reshape = 0; reshape < 2; reshape++)
        {
            const char* type = reshape ? "Reshape" : "Interp";
            const int typeindex = reshape ? ncnn::LayerType::Reshape : ncnn::LayerType::Interp;
            const int expr_id = reshape ? 6 : 9;
            for (int dynamic = 0; dynamic < (reshape ? 1 : 2); dynamic++)
            {
                const int params[] = {0, 1, 5, dynamic};
                for (size_t i = 0; i < sizeof(expressions) / sizeof(expressions[0]); i++)
                {
                    const char* expr = expressions[i];
                    const size_t len = strlen(expr);
                    std::vector<unsigned char> extra;
                    append_int(extra, -23400 - expr_id);
                    append_int(extra, (int)len);
                    extra.insert(extra.end(), expr, expr + len);
                    for (size_t j = len; j % 4 != 0; j++)
                        extra.push_back(0);
                    char text[64];
                    snprintf(text, sizeof(text), " %d=\"%s\"", expr_id, expr);
                    const int min_bottom = dynamic && required[i] < 2 ? 2 : required[i];
                    for (int bottoms = 0; bottoms <= min_bottom; bottoms++)
                    {
                        if (check_builtin_arity(binary, type, typeindex, reshape ? 0 : 2, params, bottoms, 1, bottoms == min_bottom ? 0 : -1, text, extra))
                            return -1;
                    }
                    if (check_builtin_arity(binary, type, typeindex, reshape ? 0 : 2, params, min_bottom, 0, -1, text, extra))
                        return -1;
                }
            }
        }

        // A dynamic Interp must reject an invalid expression at load time.
        const int params[] = {0, 1, 5, 1};
        std::vector<unsigned char> extra;
        append_int(extra, -23409);
        append_int(extra, 1);
        append_int(extra, '(');
        if (check_builtin_arity(binary, "Interp", ncnn::LayerType::Interp, 2, params, 2, 1, -1, " 9=\"(\"", extra))
            return -1;
    }
    return 0;
}

static int test_builtin_priorbox_arity()
{
    for (int binary = 0; binary < 2; binary++)
    {
        for (int has_max = 0; has_max < 2; has_max++)
        {
            std::vector<unsigned char> extra;
            append_int(extra, -23300);
            append_int(extra, 1);
            append_int(extra, 0x41000000); // 8.f
            if (has_max)
            {
                append_int(extra, -23301);
                append_int(extra, 1);
                append_int(extra, 0x41800000); // 16.f
            }
            const char* text = has_max ? " -23300=1,8.0 -23301=1,16.0" : " -23300=1,8.0";
            for (int dynamic_w = 0; dynamic_w < 2; dynamic_w++)
            {
                for (int dynamic_h = 0; dynamic_h < 2; dynamic_h++)
                {
                    const int params[] = {9, dynamic_w ? -233 : 32, 10, dynamic_h ? -233 : 32};
                    const int min_bottom = (dynamic_w || dynamic_h) && (has_max || !dynamic_w || !dynamic_h) ? 2 : 1;
                    for (int bottoms = 0; bottoms <= 2; bottoms++)
                    {
                        if (check_builtin_arity(binary, "PriorBox", ncnn::LayerType::PriorBox, 2, params, bottoms, 1, bottoms >= min_bottom ? 0 : -1, text, extra))
                            return -1;
                    }
                    if (check_builtin_arity(binary, "PriorBox", ncnn::LayerType::PriorBox, 2, params, min_bottom, 0, -1, text, extra))
                        return -1;
                }
            }
        }
    }
    return 0;
}
#endif // NCNN_VALIDATION

// a dynamic Scale without bias_term takes its scale from the second input and runs
static int test_scale_dynamic_forward()
{
    ncnn::Net net;
    net.opt.num_threads = 1;
    const char* param = "7767517\n3 3\nInput in0 0 1 a\nInput in1 0 1 s\nScale op 2 1 a s out 0=-233\n";
    BoundedNetReader dr(0, 0);
    if (net.load_param_mem(param) != 0 || net.load_model(dr) != 0)
    {
        fprintf(stderr, "test_scale_arity dynamic scale load failed\n");
        return -1;
    }

    ncnn::Mat a(5, 3, 8);
    ncnn::Mat s(8);
    for (int q = 0; q < a.c; q++)
    {
        float* p = a.channel(q);
        for (int i = 0; i < a.w * a.h; i++)
            p[i] = (i - 7) * 0.25f;
        s[q] = q + 1.f;
    }

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat out;
    int ret = ex.input("a", a);
    if (ret == 0)
        ret = ex.input("s", s);
    if (ret == 0)
        ret = ex.extract("out", out);
    if (ret != 0 || out.w != a.w || out.h != a.h || out.c != a.c)
    {
        fprintf(stderr, "test_scale_arity dynamic scale forward failed ret=%d\n", ret);
        return -1;
    }

    for (int q = 0; q < out.c; q++)
    {
        const float* p = out.channel(q);
        for (int i = 0; i < out.w * out.h; i++)
        {
            if (p[i] != (i - 7) * 0.25f * (q + 1.f))
            {
                fprintf(stderr, "test_scale_arity dynamic scale forward value mismatch q=%d i=%d\n", q, i);
                return -1;
            }
        }
    }

    return 0;
}

int main()
{
    return 0
#if NCNN_VALIDATION
           || test_builtin_layer_arity()
           || test_builtin_expression_arity()
           || test_builtin_priorbox_arity()
#endif
           || test_scale_dynamic_forward();
}
