// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"
#include "multiheadattention.h"

#include <string.h>

static void pack_signed_weight(unsigned char* ptr, int k, int bits, int q)
{
    const unsigned int mask = (1u << bits) - 1u;
    const unsigned int v = (unsigned int)q & mask;
    const int bit_offset = k * bits;

    for (int b = 0; b < bits; b++)
    {
        if (v & (1u << b))
        {
            const int out_bit = bit_offset + b;
            ptr[out_bit / 8] |= (unsigned char)(1u << (out_bit % 8));
        }
    }
}

static int float2int_weight(float v, int bits)
{
    const int qmax = (1 << (bits - 1)) - 1;
    int q = (int)roundf(v);
    if (q > qmax) q = qmax;
    if (q < -qmax) q = -qmax;
    return q;
}

static int weight_block_quantize_term(int bits, int block_size, int input_scale = 0)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : block_size == 128 ? 2 : -1;
    if ((bits != 4 && bits != 6 && bits != 8) || block_size_code < 0)
        return 0;

    return bits * 100 + (input_scale ? 10 : 0) + block_size_code;
}

static int weight_quantize_packed_k_bytes(int K, int bits)
{
    return (K * bits + 7) / 8;
}

static ncnn::Mat make_input_scales(int K)
{
    ncnn::Mat input_scales(K);
    float* ptr = input_scales;
    for (int k = 0; k < K; k++)
        ptr[k] = 0.75f + (k % 5) * 0.15f;

    return input_scales;
}

static ncnn::Mat scale_weight_by_input_scales(const ncnn::Mat& weight_data, const ncnn::Mat& input_scales, int inverse)
{
    const int K = weight_data.w;
    const int N = weight_data.h;

    ncnn::Mat weight_data1(K, N);
    const float* input_scale_ptr = input_scales;

    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* outptr = weight_data1.row(n);

        for (int k = 0; k < K; k++)
            outptr[k] = inverse ? ptr[k] / input_scale_ptr[k] : ptr[k] * input_scale_ptr[k];
    }

    return weight_data1;
}

static int quantize_weight(const ncnn::Mat& weight_data, int bits, int block_size, ncnn::Mat& weight_data_quantized, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_dequantized)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int packed_k_bytes = weight_quantize_packed_k_bytes(K, bits);
    const int block_count = (K + block_size - 1) / block_size;

    weight_data_quantized.create(packed_k_bytes, N, (size_t)1u);
    weight_data_quantize_scales.create(block_count, N);
    weight_data_dequantized.create(K, N);
    if (weight_data_quantized.empty() || weight_data_quantize_scales.empty() || weight_data_dequantized.empty())
        return -100;

    memset(weight_data_quantized.data, 0, weight_data_quantized.total() * weight_data_quantized.elemsize);

    const int qmax = (1 << (bits - 1)) - 1;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* scale_ptr = weight_data_quantize_scales.row(n);
        unsigned char* qptr = weight_data_quantized.row<unsigned char>(n);
        float* deqptr = weight_data_dequantized.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = block_size < K - k0 ? block_size : K - k0;

            float absmax = 0.f;
            for (int k = 0; k < max_kk; k++)
            {
                const float v = (float)fabs(ptr[k0 + k]);
                if (v > absmax)
                    absmax = v;
            }

            const float scale = absmax == 0.f ? 1.f : (float)qmax / absmax;
            scale_ptr[b] = scale;

            for (int k = 0; k < max_kk; k++)
            {
                const int q = float2int_weight(ptr[k0 + k] * scale, bits);
                pack_signed_weight(qptr, k0 + k, bits, q);
                deqptr[k0 + k] = q / scale;
            }
        }
    }

    return 0;
}

static int make_mha_weights(int qdim, int kdim, int vdim, int embed_dim, int bits, int block_size, std::vector<ncnn::Mat>& weights, std::vector<ncnn::Mat>& ref_weights, int input_scale = 0)
{
    ncnn::Mat q_weight_data = RandomMat(qdim, embed_dim, -1.f, 1.f);
    ncnn::Mat k_weight_data = RandomMat(kdim, embed_dim, -1.f, 1.f);
    ncnn::Mat v_weight_data = RandomMat(vdim, embed_dim, -1.f, 1.f);
    ncnn::Mat out_weight_data = RandomMat(embed_dim, qdim, -1.f, 1.f);

    ncnn::Mat q_input_scales;
    ncnn::Mat k_input_scales;
    ncnn::Mat v_input_scales;
    ncnn::Mat out_input_scales;

    if (input_scale)
    {
        q_input_scales = make_input_scales(qdim);
        k_input_scales = make_input_scales(kdim);
        v_input_scales = make_input_scales(vdim);
        out_input_scales = make_input_scales(embed_dim);

        q_weight_data = scale_weight_by_input_scales(q_weight_data, q_input_scales, 1);
        k_weight_data = scale_weight_by_input_scales(k_weight_data, k_input_scales, 1);
        v_weight_data = scale_weight_by_input_scales(v_weight_data, v_input_scales, 1);
        out_weight_data = scale_weight_by_input_scales(out_weight_data, out_input_scales, 1);
    }

    ncnn::Mat q_weight_data_quantized;
    ncnn::Mat k_weight_data_quantized;
    ncnn::Mat v_weight_data_quantized;
    ncnn::Mat out_weight_data_quantized;
    ncnn::Mat q_weight_data_scales;
    ncnn::Mat k_weight_data_scales;
    ncnn::Mat v_weight_data_scales;
    ncnn::Mat out_weight_data_scales;
    ncnn::Mat q_weight_data_dequantized;
    ncnn::Mat k_weight_data_dequantized;
    ncnn::Mat v_weight_data_dequantized;
    ncnn::Mat out_weight_data_dequantized;

    int ret = quantize_weight(q_weight_data, bits, block_size, q_weight_data_quantized, q_weight_data_scales, q_weight_data_dequantized);
    if (ret != 0)
        return ret;
    ret = quantize_weight(k_weight_data, bits, block_size, k_weight_data_quantized, k_weight_data_scales, k_weight_data_dequantized);
    if (ret != 0)
        return ret;
    ret = quantize_weight(v_weight_data, bits, block_size, v_weight_data_quantized, v_weight_data_scales, v_weight_data_dequantized);
    if (ret != 0)
        return ret;
    ret = quantize_weight(out_weight_data, bits, block_size, out_weight_data_quantized, out_weight_data_scales, out_weight_data_dequantized);
    if (ret != 0)
        return ret;

    if (input_scale)
    {
        q_weight_data_dequantized = scale_weight_by_input_scales(q_weight_data_dequantized, q_input_scales, 0);
        k_weight_data_dequantized = scale_weight_by_input_scales(k_weight_data_dequantized, k_input_scales, 0);
        v_weight_data_dequantized = scale_weight_by_input_scales(v_weight_data_dequantized, v_input_scales, 0);
        out_weight_data_dequantized = scale_weight_by_input_scales(out_weight_data_dequantized, out_input_scales, 0);
    }

    ncnn::Mat q_bias_data = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat k_bias_data = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat v_bias_data = RandomMat(embed_dim, -1.f, 1.f);
    ncnn::Mat out_bias_data = RandomMat(qdim, -1.f, 1.f);

    weights.resize(input_scale ? 16 : 12);
    weights[0] = q_weight_data_quantized;
    weights[1] = q_bias_data;
    weights[2] = k_weight_data_quantized;
    weights[3] = k_bias_data;
    weights[4] = v_weight_data_quantized;
    weights[5] = v_bias_data;
    weights[6] = out_weight_data_quantized;
    weights[7] = out_bias_data;
    weights[8] = q_weight_data_scales;
    weights[9] = k_weight_data_scales;
    weights[10] = v_weight_data_scales;
    weights[11] = out_weight_data_scales;

    if (input_scale)
    {
        weights[12] = q_input_scales;
        weights[13] = k_input_scales;
        weights[14] = v_input_scales;
        weights[15] = out_input_scales;
    }

    ref_weights.resize(8);
    ref_weights[0] = q_weight_data_dequantized.reshape(embed_dim * qdim);
    ref_weights[1] = q_bias_data;
    ref_weights[2] = k_weight_data_dequantized.reshape(embed_dim * kdim);
    ref_weights[3] = k_bias_data;
    ref_weights[4] = v_weight_data_dequantized.reshape(embed_dim * vdim);
    ref_weights[5] = v_bias_data;
    ref_weights[6] = out_weight_data_dequantized.reshape(qdim * embed_dim);
    ref_weights[7] = out_bias_data;

    return 0;
}

static ncnn::ParamDict make_mha_param(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int attn_mask, int kv_cache, int quantize_term)
{
    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, kv_cache);
    if (quantize_term)
        pd.set(18, quantize_term);

    return pd;
}

static int run_mha_layer(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& inputs, int top_blob_count, std::vector<ncnn::Mat>& outputs)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    return test_layer_cpu(ncnn::LayerType::MultiHeadAttention, pd, weights, opt, inputs, top_blob_count, outputs, std::vector<ncnn::Mat>(), TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_multiheadattention_block_quant(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int bits, int block_size, int attn_mask, int input_scale = 0)
{
    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, weights, ref_weights, input_scale);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = RandomMat(qdim, 5, -1.f, 1.f);
    inputs[1] = RandomMat(kdim, 6, -1.f, 1.f);
    inputs[2] = RandomMat(vdim, 6, -1.f, 1.f);

    if (attn_mask)
        inputs.push_back(RandomMat(6, 5, -1.f, 0.f));

    const int quantize_term = weight_block_quantize_term(bits, block_size, input_scale);
    const ncnn::ParamDict pd = make_mha_param(qdim, kdim, vdim, embed_dim, num_heads, attn_mask, 0, quantize_term);
    const ncnn::ParamDict ref_pd = make_mha_param(qdim, kdim, vdim, embed_dim, num_heads, attn_mask, 0, 0);

    std::vector<ncnn::Mat> outputs;
    std::vector<ncnn::Mat> refs;
    ret = run_mha_layer(pd, weights, inputs, 1, outputs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant failed ret=%d qdim=%d kdim=%d vdim=%d embed_dim=%d bits=%d block_size=%d attn_mask=%d input_scale=%d\n", ret, qdim, kdim, vdim, embed_dim, bits, block_size, attn_mask, input_scale);
        return ret;
    }

    ret = run_mha_layer(ref_pd, ref_weights, inputs, 1, refs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant reference failed ret=%d qdim=%d kdim=%d vdim=%d embed_dim=%d\n", ret, qdim, kdim, vdim, embed_dim);
        return ret;
    }

    ret = CompareMat(outputs, refs, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant compare failed qdim=%d kdim=%d vdim=%d embed_dim=%d bits=%d block_size=%d attn_mask=%d input_scale=%d\n", qdim, kdim, vdim, embed_dim, bits, block_size, attn_mask, input_scale);
        return ret;
    }

    return 0;
}

static int test_multiheadattention_block_quant_kvcache(int attn_mask = 0, int input_scale = 0)
{
    const int qdim = 10;
    const int embed_dim = 8;
    const int num_heads = 2;
    const int bits = 4;
    const int block_size = 64;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, qdim, qdim, embed_dim, bits, block_size, weights, ref_weights, input_scale);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> inputs(attn_mask ? 4 : 3);
    inputs[0] = RandomMat(qdim, 3, -1.f, 1.f);
    if (attn_mask)
    {
        inputs[1] = RandomMat(8, 3, -1.f, 0.f);
        inputs[2] = RandomMat(5, embed_dim, -1.f, 1.f);
        inputs[3] = RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        inputs[1] = RandomMat(5, embed_dim, -1.f, 1.f);
        inputs[2] = RandomMat(5, embed_dim, -1.f, 1.f);
    }

    const int quantize_term = weight_block_quantize_term(bits, block_size, input_scale);
    const ncnn::ParamDict pd = make_mha_param(qdim, qdim, qdim, embed_dim, num_heads, attn_mask, 1, quantize_term);
    const ncnn::ParamDict ref_pd = make_mha_param(qdim, qdim, qdim, embed_dim, num_heads, attn_mask, 1, 0);

    std::vector<ncnn::Mat> outputs;
    std::vector<ncnn::Mat> refs;
    ret = run_mha_layer(pd, weights, inputs, 3, outputs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_kvcache failed ret=%d attn_mask=%d input_scale=%d\n", ret, attn_mask, input_scale);
        return ret;
    }

    ret = run_mha_layer(ref_pd, ref_weights, inputs, 3, refs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_kvcache reference failed ret=%d attn_mask=%d input_scale=%d\n", ret, attn_mask, input_scale);
        return ret;
    }

    ret = CompareMat(outputs, refs, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_kvcache compare failed attn_mask=%d input_scale=%d\n", attn_mask, input_scale);
        return ret;
    }

    return 0;
}

static int test_multiheadattention_block_quant_cross_kvcache(int attn_mask = 0, int input_scale = 0)
{
    const int qdim = 65;
    const int kdim = 33;
    const int vdim = 49;
    const int embed_dim = 64;
    const int num_heads = 4;
    const int bits = 6;
    const int block_size = 32;

    std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> ref_weights;
    int ret = make_mha_weights(qdim, kdim, vdim, embed_dim, bits, block_size, weights, ref_weights, input_scale);
    if (ret != 0)
        return ret;

    std::vector<ncnn::Mat> inputs(attn_mask ? 6 : 5);
    inputs[0] = RandomMat(qdim, 3, -1.f, 1.f);
    inputs[1] = RandomMat(kdim, 2, -1.f, 1.f);
    inputs[2] = RandomMat(vdim, 2, -1.f, 1.f);
    if (attn_mask)
    {
        inputs[3] = RandomMat(5, 3, -1.f, 0.f);
        inputs[4] = RandomMat(5, embed_dim, -1.f, 1.f);
        inputs[5] = RandomMat(5, embed_dim, -1.f, 1.f);
    }
    else
    {
        inputs[3] = RandomMat(5, embed_dim, -1.f, 1.f);
        inputs[4] = RandomMat(5, embed_dim, -1.f, 1.f);
    }

    const int quantize_term = weight_block_quantize_term(bits, block_size, input_scale);
    const ncnn::ParamDict pd = make_mha_param(qdim, kdim, vdim, embed_dim, num_heads, attn_mask, 1, quantize_term);
    const ncnn::ParamDict ref_pd = make_mha_param(qdim, kdim, vdim, embed_dim, num_heads, attn_mask, 1, 0);

    std::vector<ncnn::Mat> outputs;
    std::vector<ncnn::Mat> refs;
    ret = run_mha_layer(pd, weights, inputs, 3, outputs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_cross_kvcache failed ret=%d attn_mask=%d input_scale=%d\n", ret, attn_mask, input_scale);
        return ret;
    }

    ret = run_mha_layer(ref_pd, ref_weights, inputs, 3, refs);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_cross_kvcache reference failed ret=%d attn_mask=%d input_scale=%d\n", ret, attn_mask, input_scale);
        return ret;
    }

    ret = CompareMat(outputs, refs, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant_cross_kvcache compare failed attn_mask=%d input_scale=%d\n", attn_mask, input_scale);
        return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

#if !NCNN_WEIGHT_QUANT
    ncnn::ParamDict pd = make_mha_param(5, 5, 5, 4, 2, 0, 0, 410);

    ncnn::MultiHeadAttention mha;
    if (mha.load_param(pd) == 0)
    {
        fprintf(stderr, "test_multiheadattention_block_quant failed NCNN_WEIGHT_QUANT=OFF accepted weight block quantization\n");
        return -1;
    }

    return 0;
#else
    return 0
           || test_multiheadattention_block_quant(13, 9, 11, 8, 2, 4, 32, 0)
           || test_multiheadattention_block_quant(10, 10, 10, 8, 2, 6, 64, 1)
           || test_multiheadattention_block_quant(12, 7, 9, 8, 2, 8, 128, 0)
           || test_multiheadattention_block_quant(13, 9, 11, 8, 2, 4, 64, 1, 1)
           || test_multiheadattention_block_quant(65, 65, 65, 64, 4, 6, 64, 1)
           || test_multiheadattention_block_quant(65, 33, 49, 64, 4, 4, 32, 0, 1)
           || test_multiheadattention_block_quant_kvcache()
           || test_multiheadattention_block_quant_kvcache(0, 1)
           || test_multiheadattention_block_quant_kvcache(1)
           || test_multiheadattention_block_quant_kvcache(1, 1)
           || test_multiheadattention_block_quant_cross_kvcache()
           || test_multiheadattention_block_quant_cross_kvcache(1)
           || test_multiheadattention_block_quant_cross_kvcache(0, 1);
#endif
}
