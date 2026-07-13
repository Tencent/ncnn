// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "gemm.h"
#include "layer_type.h"

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

static ncnn::ParamDict make_gemm_param(int M, int N, int K, int quantize_term, float alpha = 1.f, float beta = 1.f, int constantC = 0, int broadcast_type_C = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, 0); // transA
    pd.set(3, 1); // transB
    pd.set(4, 0); // constantA
    pd.set(5, 1); // constantB
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, broadcast_type_C);
    if (quantize_term)
        pd.set(18, quantize_term);

    return pd;
}

static int test_gemm_block_quant(int M, int N, int K, int bits, int block_size, int broadcast_type_C, int input_scale = 0, float alpha = 1.f, float beta = 1.f, int constantC = 0, int dims3 = 0)
{
    const int quantize_term = weight_block_quantize_term(bits, block_size, input_scale);

    ncnn::Mat A = dims3 ? RandomMat(K, 1, M, -2.f, 2.f) : RandomMat(K, M, -2.f, 2.f);
    ncnn::Mat B = RandomMat(K, N, -2.f, 2.f);
    ncnn::Mat C;

    ncnn::Mat B_input_scales;
    if (input_scale)
    {
        B_input_scales = make_input_scales(K);
        B = scale_weight_by_input_scales(B, B_input_scales, 1);
    }

    ncnn::Mat B_quantized;
    ncnn::Mat B_quantize_scales;
    ncnn::Mat B_dequantized;
    int ret = quantize_weight(B, bits, block_size, B_quantized, B_quantize_scales, B_dequantized);
    if (ret != 0)
        return ret;

    if (input_scale)
        B_dequantized = scale_weight_by_input_scales(B_dequantized, B_input_scales, 0);

    if (broadcast_type_C == 0) C = RandomMat(1);
    if (broadcast_type_C == 1) C = RandomMat(M);
    if (broadcast_type_C == 2) C = RandomMat(1, M);
    if (broadcast_type_C == 3) C = RandomMat(N, M);
    if (broadcast_type_C == 4) C = RandomMat(N);

    std::vector<ncnn::Mat> weights;
    weights.push_back(B_quantized);
    if (constantC)
        weights.push_back(C);
    weights.push_back(B_quantize_scales);
    if (input_scale)
        weights.push_back(B_input_scales);

    std::vector<ncnn::Mat> ref_weights;
    ref_weights.push_back(B_dequantized);
    if (constantC)
        ref_weights.push_back(C);

    std::vector<ncnn::Mat> inputs;
    inputs.push_back(A);

    if (!constantC && !C.empty())
        inputs.push_back(C);

    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    std::vector<ncnn::Mat> outputs;
    std::vector<ncnn::Mat> refs;
    ret = test_layer_cpu(ncnn::LayerType::Gemm, make_gemm_param(M, N, K, quantize_term, alpha, beta, constantC, broadcast_type_C), weights, opt, inputs, 1, outputs, std::vector<ncnn::Mat>(), TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_block_quant failed ret=%d M=%d N=%d K=%d bits=%d block_size=%d broadcast_type_C=%d input_scale=%d constantC=%d dims3=%d\n", ret, M, N, K, bits, block_size, broadcast_type_C, input_scale, constantC, dims3);
        return ret;
    }

    ret = test_layer_cpu(ncnn::LayerType::Gemm, make_gemm_param(M, N, K, 0, alpha, beta, constantC, broadcast_type_C), ref_weights, opt, inputs, 1, refs, std::vector<ncnn::Mat>(), TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_block_quant reference failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
        return ret;
    }

    ret = CompareMat(outputs, refs, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_block_quant compare failed M=%d N=%d K=%d bits=%d block_size=%d broadcast_type_C=%d input_scale=%d constantC=%d dims3=%d\n", M, N, K, bits, block_size, broadcast_type_C, input_scale, constantC, dims3);
        return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

#if !NCNN_WEIGHT_QUANT
    ncnn::ParamDict pd = make_gemm_param(3, 4, 5, 410);

    ncnn::Gemm gemm;
    if (gemm.load_param(pd) == 0)
    {
        fprintf(stderr, "test_gemm_block_quant failed NCNN_WEIGHT_QUANT=OFF accepted weight block quantization\n");
        return -1;
    }

    return 0;
#else
    return 0
           || test_gemm_block_quant(3, 5, 65, 4, 64, -1)
           || test_gemm_block_quant(3, 4, 33, 4, 32, 0)
           || test_gemm_block_quant(4, 7, 67, 6, 64, 3)
           || test_gemm_block_quant(3, 4, 33, 6, 32, 2)
           || test_gemm_block_quant(2, 3, 5, 8, 32, 4)
           || test_gemm_block_quant(2, 5, 65, 8, 32, -1)
           || test_gemm_block_quant(3, 4, 129, 6, 128, -1)
           || test_gemm_block_quant(3, 5, 65, 4, 64, -1, 1)
           || test_gemm_block_quant(2, 4, 31, 6, 32, 1, 1)
           || test_gemm_block_quant(3, 4, 32, 4, 32, -1)
           || test_gemm_block_quant(3, 4, 33, 4, 32, 4, 0, 1.7f, 0.3f, 1)
           || test_gemm_block_quant(5, 3, 65, 6, 64, -1, 0, 1.f, 1.f, 0, 1);
#endif
}
