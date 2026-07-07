// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNNLLM_QUANT_H
#define NCNNLLM_QUANT_H

#include <algorithm>
#include <cmath>
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

static inline const char* llm_quant_bits_to_dtype(int weight_bits)
{
    if (weight_bits == 4)
        return "int4";
    if (weight_bits == 6)
        return "int6";
    if (weight_bits == 8)
        return "int8";

    return "";
}

static inline int llm_quant_dtype_to_bits(const char* dtype)
{
    if (strcmp(dtype, "int4") == 0)
        return 4;
    if (strcmp(dtype, "int6") == 0)
        return 6;
    if (strcmp(dtype, "int8") == 0)
        return 8;

    return 0;
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

static inline void print_skip_gemm(const ncnn::Gemm* gemm, const char* reason)
{
    fprintf(stderr, "skip_gemm %s %s\n", gemm_name(gemm), reason);
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

static inline void print_skip_multiheadattention(const ncnn::MultiHeadAttention* mha, const char* reason)
{
    fprintf(stderr, "skip_multiheadattention %s %s\n", multiheadattention_name(mha), reason);
}

static inline bool is_supported_llm_gemm(const ncnn::Gemm* gemm, const char** reason)
{
    if (gemm->alpha != 1.f || gemm->beta != 1.f)
    {
        *reason = "alpha/beta is not 1";
        return false;
    }

    if (gemm->transA != 0 || gemm->transB != 1)
    {
        *reason = "requires transA=0 transB=1";
        return false;
    }

    if (gemm->constantA != 0 || gemm->constantB != 1 || gemm->constantC != 1 || gemm->constant_broadcast_type_C != -1)
    {
        *reason = "requires constantA=0 constantB=1 constantC=1 broadcastC=-1";
        return false;
    }

    if (gemm->constantM != 0)
    {
        *reason = "requires dynamic M";
        return false;
    }

    if (gemm->output_N1M != 0 || gemm->output_elempack != 0 || gemm->output_elemtype != 0 || gemm->output_transpose != 0)
    {
        *reason = "unsupported output layout";
        return false;
    }

    if (gemm->quantize_term != 0)
    {
        *reason = "already quantized";
        return false;
    }

    if (gemm->constant_TILE_M != 0 || gemm->constant_TILE_N != 0 || gemm->constant_TILE_K != 0)
    {
        *reason = "tiled Gemm is not supported";
        return false;
    }

    const int constantN = gemm->constantN;
    const int constantK = gemm->constantK;

    if (constantN <= 0 || constantK <= 0 || gemm->B_data.w != constantK || gemm->B_data.h != constantN || gemm->B_data.elemsize != 4u)
    {
        *reason = "B weight shape or storage is unsupported";
        return false;
    }

    return true;
}

static inline bool is_supported_llm_multiheadattention(const ncnn::MultiHeadAttention* mha, const char** reason)
{
    if (mha->quantize_term != 0)
    {
        *reason = "already quantized";
        return false;
    }

    if (mha->embed_dim <= 0 || mha->num_heads <= 0 || mha->embed_dim % mha->num_heads != 0 || mha->weight_data_size <= 0 || mha->weight_data_size % mha->embed_dim != 0 || mha->kdim <= 0 || mha->vdim <= 0)
    {
        *reason = "requires valid embed_dim/num_heads/weight_data_size/kdim/vdim";
        return false;
    }

    const int qdim = mha->weight_data_size / mha->embed_dim;
    if (mha->q_weight_data.elemsize != 4u || mha->q_weight_data.w != mha->embed_dim * qdim)
    {
        *reason = "q weight shape or storage is unsupported";
        return false;
    }

    if (mha->k_weight_data.elemsize != 4u || mha->k_weight_data.w != mha->embed_dim * mha->kdim)
    {
        *reason = "k weight shape or storage is unsupported";
        return false;
    }

    if (mha->v_weight_data.elemsize != 4u || mha->v_weight_data.w != mha->embed_dim * mha->vdim)
    {
        *reason = "v weight shape or storage is unsupported";
        return false;
    }

    if (mha->out_weight_data.elemsize != 4u || mha->out_weight_data.w != qdim * mha->embed_dim)
    {
        *reason = "out weight shape or storage is unsupported";
        return false;
    }

    if (mha->q_bias_data.elemsize != 4u || mha->q_bias_data.w != mha->embed_dim || mha->k_bias_data.elemsize != 4u || mha->k_bias_data.w != mha->embed_dim || mha->v_bias_data.elemsize != 4u || mha->v_bias_data.w != mha->embed_dim || mha->out_bias_data.elemsize != 4u || mha->out_bias_data.w != qdim)
    {
        *reason = "bias shape or storage is unsupported";
        return false;
    }

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

static inline int make_gemm_B_scales(const ncnn::Mat& B_data, int block_size, int weight_bits, int method, ncnn::Mat& B_data_quantize_scales, int num_threads = 1)
{
    const int constantN = B_data.h;
    const int constantK = B_data.w;
    const int block_count = (constantK + block_size - 1) / block_size;

    B_data_quantize_scales.create(block_count, constantN);
    if (B_data_quantize_scales.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int n = 0; n < constantN; n++)
    {
        const float* ptr = B_data.row(n);
        float* scale_ptr = B_data_quantize_scales.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = std::min(block_size, constantK - k0);
            scale_ptr[b] = choose_weight_scale(ptr + k0, max_kk, weight_bits, method);
        }
    }

    return 0;
}

static inline int pack_gemm_B_from_scales(const ncnn::Mat& B_data, const ncnn::Mat& B_data_quantize_scales, int block_size, int weight_bits, ncnn::Mat& B_data_quantized)
{
    const int constantN = B_data.h;
    const int constantK = B_data.w;
    const int block_count = (constantK + block_size - 1) / block_size;

    if (B_data_quantize_scales.w != block_count || B_data_quantize_scales.h != constantN)
    {
        fprintf(stderr, "Gemm B scale shape mismatch expected=%d,%d got=%d,%d\n", block_count, constantN, B_data_quantize_scales.w, B_data_quantize_scales.h);
        return -1;
    }

    for (int n = 0; n < constantN; n++)
    {
        const float* scale_ptr = B_data_quantize_scales.row(n);
        for (int b = 0; b < block_count; b++)
        {
            const float scale = scale_ptr[b];
            if (!(scale > 0.f) || !std::isfinite(scale))
            {
                fprintf(stderr, "Gemm B scale is invalid n=%d block=%d scale=%f\n", n, b, scale);
                return -1;
            }
        }
    }

    const int packed_k_bytes = llm_weight_quantize_packed_k_bytes(constantK, weight_bits);
    if (packed_k_bytes <= 0)
        return -1;

    B_data_quantized.create(packed_k_bytes, constantN, (size_t)1u);
    if (B_data_quantized.empty())
        return -100;

    memset(B_data_quantized.data, 0, B_data_quantized.total() * B_data_quantized.elemsize);

    for (int n = 0; n < constantN; n++)
    {
        const float* ptr = B_data.row(n);
        const float* scale_ptr = B_data_quantize_scales.row(n);
        unsigned char* qptr = B_data_quantized.row<unsigned char>(n);

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

static inline int make_and_pack_gemm_B(const ncnn::Mat& B_data, int block_size, int weight_bits, int method, ncnn::Mat& B_data_quantized, ncnn::Mat& B_data_quantize_scales)
{
    int ret = make_gemm_B_scales(B_data, block_size, weight_bits, method, B_data_quantize_scales);
    if (ret != 0)
        return ret;

    return pack_gemm_B_from_scales(B_data, B_data_quantize_scales, block_size, weight_bits, B_data_quantized);
}

static inline int write_llm_table_row(FILE* fp, const char* key, int weight_bits, int block_size, int method, const ncnn::Mat& scales)
{
    const char* dtype = llm_quant_bits_to_dtype(weight_bits);
    if (dtype[0] == '\0')
        return -1;

    fprintf(fp, "%s format=block_symmetric dtype=%s block=%d scale_dtype=fp32 scale_encoding=quant method=%s ", key, dtype, block_size, llm_quant_method_to_string(method));

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%f ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

static inline int write_llm_input_scale_row(FILE* fp, const char* key, int method, const ncnn::Mat& scales)
{
    fprintf(fp, "%s format=input_scale scale_dtype=fp32 scale_encoding=mul method=%s ", key, llm_quant_method_to_string(method));

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%f ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

static inline int write_llm_qweight_table_row(FILE* fp, const char* key, int weight_bits, int block_size, int method, const char* qweight, const ncnn::Mat& scales)
{
    const char* dtype = llm_quant_bits_to_dtype(weight_bits);
    if (dtype[0] == '\0')
        return -1;

    fprintf(fp, "%s format=block_symmetric_qweight dtype=%s block=%d scale_dtype=fp32 scale_encoding=quant qweight=%s qweight_encoding=sint%d_packed layout=nk method=%s ", key, dtype, block_size, qweight, weight_bits, llm_quant_method_to_string(method));

    const float* ptr = scales;
    const size_t size = (size_t)scales.w * scales.h * scales.d * scales.c;
    for (size_t i = 0; i < size; i++)
    {
        fprintf(fp, "%f ", ptr[i]);
    }
    fprintf(fp, "\n");

    return 0;
}

#endif // NCNNLLM_QUANT_H
