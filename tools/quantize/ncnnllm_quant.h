// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNNLLM_QUANT_H
#define NCNNLLM_QUANT_H

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits.h>

enum
{
    LLM_QUANT_METHOD_MINMAX = 0,
    LLM_QUANT_METHOD_MSECLIP = 1,
    LLM_QUANT_METHOD_AWQ = 2,
    LLM_QUANT_METHOD_GPTQ = 3
};

static inline const char* llm_quant_method_to_string(int method)
{
    if (method == LLM_QUANT_METHOD_MINMAX)
        return "minmax";
    if (method == LLM_QUANT_METHOD_MSECLIP)
        return "mseclip";
    if (method == LLM_QUANT_METHOD_AWQ)
        return "awq";
    if (method == LLM_QUANT_METHOD_GPTQ)
        return "gptq";

    return "";
}

static inline int llm_quant_method_from_string(const char* method)
{
    if (strcmp(method, "minmax") == 0)
        return LLM_QUANT_METHOD_MINMAX;
    if (strcmp(method, "mseclip") == 0)
        return LLM_QUANT_METHOD_MSECLIP;
    if (strcmp(method, "awq") == 0)
        return LLM_QUANT_METHOD_AWQ;
    if (strcmp(method, "gptq") == 0)
        return LLM_QUANT_METHOD_GPTQ;

    return -1;
}

static inline int llm_weight_block_quantize_term(int weight_bits, int block_size, bool input_scale = false)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : block_size == 128 ? 2 : -1;
    if ((weight_bits != 4 && weight_bits != 6 && weight_bits != 8) || block_size_code < 0)
        return 0;

    return weight_bits * 100 + (input_scale ? 10 : 0) + block_size_code;
}

static inline int llm_weight_quantize_packed_k_bytes(int constantK, int weight_bits)
{
    if (constantK <= 0 || weight_bits <= 0)
        return -1;

    const size_t packed_k_bytes = ((size_t)constantK * weight_bits + 7) / 8;
    if (packed_k_bytes > (size_t)INT_MAX)
        return -1;

    return (int)packed_k_bytes;
}

static inline int float2int_weight(float v, int weight_bits)
{
    const int qmax = (1 << (weight_bits - 1)) - 1;
    int q = static_cast<int>(roundf(v));
    if (q > qmax)
        q = qmax;
    if (q < -qmax)
        q = -qmax;

    return q;
}

static inline void pack_signed_weight(unsigned char* ptr, int k, int weight_bits, int q)
{
    const unsigned int mask = (1u << weight_bits) - 1u;
    const unsigned int v = (unsigned int)q & mask;
    const int bit_offset = k * weight_bits;

    for (int b = 0; b < weight_bits; b++)
    {
        if (v & (1u << b))
        {
            const int out_bit = bit_offset + b;
            ptr[out_bit / 8] |= (unsigned char)(1u << (out_bit % 8));
        }
    }
}

static inline const char* gemm_name(const ncnn::Gemm* gemm)
{
#if NCNN_STRING
    return gemm->name.c_str();
#else
    (void)gemm;
    return "";
#endif
}

static inline void print_skip_gemm(const ncnn::Gemm* gemm)
{
    fprintf(stderr, "skip_gemm %s\n", gemm_name(gemm));
}

static inline const char* multiheadattention_name(const ncnn::MultiHeadAttention* mha)
{
#if NCNN_STRING
    return mha->name.c_str();
#else
    (void)mha;
    return "";
#endif
}

static inline void print_skip_multiheadattention(const ncnn::MultiHeadAttention* mha)
{
    fprintf(stderr, "skip_multiheadattention %s\n", multiheadattention_name(mha));
}

static inline bool is_supported_llm_gemm(const ncnn::Gemm* gemm)
{
    if (gemm->alpha != 1.f || gemm->beta != 1.f)
        return false;

    if (gemm->transA != 0 || gemm->transB != 1)
        return false;

    if (gemm->constantA != 0 || gemm->constantB != 1)
        return false;

    if (gemm->constantC != 0 && gemm->constantC != 1)
        return false;

    if (gemm->constantC == 1 && (gemm->constant_broadcast_type_C < -1 || gemm->constant_broadcast_type_C > 4))
        return false;

    if (gemm->output_N1M != 0 || gemm->output_elempack != 0 || (gemm->output_elemtype != 0 && gemm->output_elemtype != 1) || gemm->output_transpose != 0)
        return false;

    if (gemm->quantize_term != 0)
        return false;

    if (gemm->constant_TILE_M != 0 || gemm->constant_TILE_N != 0 || gemm->constant_TILE_K != 0)
        return false;

    const int constantN = gemm->constantN;
    const int constantK = gemm->constantK;

    if (constantN <= 0 || constantK <= 0 || gemm->B_data.w != constantK || gemm->B_data.h != constantN || gemm->B_data.elemsize != 4u)
        return false;

    return true;
}

static inline bool is_supported_llm_multiheadattention(const ncnn::MultiHeadAttention* mha)
{
    if (mha->quantize_term != 0)
        return false;

    if (mha->embed_dim <= 0 || mha->num_heads <= 0 || mha->embed_dim % mha->num_heads != 0 || mha->weight_data_size <= 0 || mha->weight_data_size % mha->embed_dim != 0 || mha->kdim <= 0 || mha->vdim <= 0)
        return false;

    const int qdim = mha->weight_data_size / mha->embed_dim;
    if (mha->q_weight_data.elemsize != 4u || mha->q_weight_data.w != mha->embed_dim * qdim)
        return false;

    if (mha->k_weight_data.elemsize != 4u || mha->k_weight_data.w != mha->embed_dim * mha->kdim)
        return false;

    if (mha->v_weight_data.elemsize != 4u || mha->v_weight_data.w != mha->embed_dim * mha->vdim)
        return false;

    if (mha->out_weight_data.elemsize != 4u || mha->out_weight_data.w != qdim * mha->embed_dim)
        return false;

    if (mha->q_bias_data.elemsize != 4u || mha->q_bias_data.w != mha->embed_dim || mha->k_bias_data.elemsize != 4u || mha->k_bias_data.w != mha->embed_dim || mha->v_bias_data.elemsize != 4u || mha->v_bias_data.w != mha->embed_dim || mha->out_bias_data.elemsize != 4u || mha->out_bias_data.w != qdim)
        return false;

    return true;
}

static inline float choose_weight_scale(const float* ptr, int size, int weight_bits, int method)
{
    float absmax = 0.f;
    for (int i = 0; i < size; i++)
    {
        absmax = std::max(absmax, (float)fabs(ptr[i]));
    }

    if (absmax == 0.f)
        return 1.f;

    const int qmax = (1 << (weight_bits - 1)) - 1;
    if (method != LLM_QUANT_METHOD_MSECLIP)
        return (float)qmax / absmax;

    float best_scale = (float)qmax / absmax;
    float best_error = 0.f;
    for (int i = 0; i < size; i++)
    {
        const int q = float2int_weight(ptr[i] * best_scale, weight_bits);
        const float deq = q / best_scale;
        const float diff = ptr[i] - deq;
        best_error += diff * diff;
    }

    const int search_steps = 20;
    for (int s = 1; s <= search_steps; s++)
    {
        const float shrink = 1.f - 0.5f * s / search_steps;
        const float scale = (float)qmax / (absmax * shrink);

        float error = 0.f;
        for (int i = 0; i < size; i++)
        {
            const int q = float2int_weight(ptr[i] * scale, weight_bits);
            const float deq = q / scale;
            const float diff = ptr[i] - deq;
            error += diff * diff;
        }

        if (error < best_error)
        {
            best_error = error;
            best_scale = scale;
        }
    }

    return best_scale;
}

static inline int make_weight_scales(const ncnn::Mat& weight_data, int block_size, int weight_bits, int method, ncnn::Mat& weight_data_quantize_scales, int num_threads = 1)
{
    const int constantN = weight_data.h;
    const int constantK = weight_data.w;
    const int block_count = (constantK + block_size - 1) / block_size;

    weight_data_quantize_scales.create(block_count, constantN);
    if (weight_data_quantize_scales.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int n = 0; n < constantN; n++)
    {
        const float* ptr = weight_data.row(n);
        float* scale_ptr = weight_data_quantize_scales.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = std::min(block_size, constantK - k0);
            scale_ptr[b] = choose_weight_scale(ptr + k0, max_kk, weight_bits, method);
        }
    }

    return 0;
}

static inline int pack_weight_data(const ncnn::Mat& weight_data, const ncnn::Mat& weight_data_quantize_scales, int block_size, int weight_bits, ncnn::Mat& weight_data_quantized)
{
    const int constantN = weight_data.h;
    const int constantK = weight_data.w;
    const int block_count = (constantK + block_size - 1) / block_size;

    if (weight_data_quantize_scales.w != block_count || weight_data_quantize_scales.h != constantN)
    {
        fprintf(stderr, "weight scale shape mismatch\n");
        return -1;
    }

    const int packed_k_bytes = llm_weight_quantize_packed_k_bytes(constantK, weight_bits);

    weight_data_quantized.create(packed_k_bytes, constantN, (size_t)1u);
    if (weight_data_quantized.empty())
        return -100;

    memset(weight_data_quantized.data, 0, weight_data_quantized.total() * weight_data_quantized.elemsize);

    for (int n = 0; n < constantN; n++)
    {
        const float* ptr = weight_data.row(n);
        const float* scale_ptr = weight_data_quantize_scales.row(n);
        unsigned char* qptr = weight_data_quantized.row<unsigned char>(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = std::min(block_size, constantK - k0);
            const float scale = scale_ptr[b];

            for (int k = 0; k < max_kk; k++)
            {
                const int q = float2int_weight(ptr[k0 + k] * scale, weight_bits);
                pack_signed_weight(qptr, k0 + k, weight_bits, q);
            }
        }
    }

    return 0;
}

static inline int quantize_weight_data(const ncnn::Mat& weight_data, int block_size, int weight_bits, int method, ncnn::Mat& weight_data_quantized, ncnn::Mat& weight_data_quantize_scales)
{
    int ret = make_weight_scales(weight_data, block_size, weight_bits, method, weight_data_quantize_scales);
    if (ret != 0)
        return ret;

    return pack_weight_data(weight_data, weight_data_quantize_scales, block_size, weight_bits, weight_data_quantized);
}

static inline int write_llm_table_row(FILE* fp, const char* key, int weight_bits, int block_size, int method, const ncnn::Mat& scales)
{
    if (weight_bits != 4 && weight_bits != 6 && weight_bits != 8)
        return -1;

    fprintf(fp, "%s bits=%d block=%d method=%s ", key, weight_bits, block_size, llm_quant_method_to_string(method));

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%.9g ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

static inline int write_llm_input_scale_row(FILE* fp, const char* key, int method, const ncnn::Mat& scales)
{
    fprintf(fp, "%s method=%s ", key, llm_quant_method_to_string(method));

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%.9g ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

static inline int write_llm_qweight_table_row(FILE* fp, const char* key, int weight_bits, int block_size, int method, const char* qweight, const ncnn::Mat& scales)
{
    if (weight_bits != 4 && weight_bits != 6 && weight_bits != 8)
        return -1;

    fprintf(fp, "%s bits=%d block=%d method=%s qweight=%s ", key, weight_bits, block_size, llm_quant_method_to_string(method), qweight);

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%.9g ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

#endif // NCNNLLM_QUANT_H
