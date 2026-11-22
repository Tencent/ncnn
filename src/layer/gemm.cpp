// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm.h"

#include "cpu.h"

#include <algorithm>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

namespace ncnn {

int Gemm::get_weight_block_quantize_params(int& weight_bits, int& block_size, bool& has_input_scale) const
{
    weight_bits = quantize_term / 100;
    const int format_code = quantize_term % 100 / 10;
    const int block_size_code = quantize_term % 10;

    if (weight_bits != 4 && weight_bits != 6 && weight_bits != 8)
        return -1;

    if (format_code != 0 && format_code != 1)
        return -1;

    if (block_size_code < 0 || block_size_code > 2)
        return -1;

    block_size = block_size_code == 0 ? 32 : block_size_code == 1 ? 64 : 128;
    has_input_scale = format_code == 1;

    return 0;
}

#if NCNN_INT8 || NCNN_WEIGHT_QUANT
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}
#endif // NCNN_INT8 || NCNN_WEIGHT_QUANT

#if NCNN_WEIGHT_QUANT
static int gemm_weight_quantize_packed_k_bytes(int constantK, int weight_bits)
{
    if (constantK <= 0 || weight_bits <= 0)
        return -1;

    const uint64_t packed_k_bytes = ((uint64_t)constantK * weight_bits + 7) / 8;
    if (packed_k_bytes > INT_MAX)
        return -1;

    return (int)packed_k_bytes;
}

static void weight_block_quantize_activation_row_int8(const Mat& A, int transA, int i, signed char* outptr, float* descale_ptr, int K, int block_size)
{
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* ptrA = (const float*)A + (transA ? 0 : i * A_hstep);

    for (int g = 0; g < block_count; g++)
    {
        const int k0 = g * block_size;
        const int max_kk = block_size < K - k0 ? block_size : K - k0;

        float absmax = 0.f;
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int k = k0 + kk;
            float v = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
            v = fabsf(v);
            if (v > absmax)
                absmax = v;
        }

        if (absmax == 0.f)
        {
            descale_ptr[g] = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
                outptr[k0 + kk] = 0;
            continue;
        }

        const float scale = 127.f / absmax;
        descale_ptr[g] = absmax / 127.f;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int k = k0 + kk;
            float v = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
            outptr[k] = float2int8(v * scale);
        }
    }
}

static void weight_block_scale_quantize_activation_row_int8(const Mat& A, int transA, int i, signed char* outptr, float* descale_ptr, int K, int block_size, const float* input_scale_ptr)
{
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const float* ptrA = (const float*)A + (transA ? 0 : i * A_hstep);

    for (int g = 0; g < block_count; g++)
    {
        const int k0 = g * block_size;
        const int max_kk = block_size < K - k0 ? block_size : K - k0;

        float absmax = 0.f;
        for (int kk = 0; kk < max_kk; kk++)
        {
            const int k = k0 + kk;
            float v = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
            v = fabsf(v) * input_scale_ptr[k];
            if (v > absmax)
                absmax = v;
        }

        if (absmax == 0.f)
        {
            descale_ptr[g] = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
                outptr[k0 + kk] = 0;
            continue;
        }

        const float scale = 127.f / absmax;
        descale_ptr[g] = absmax / 127.f;

        for (int kk = 0; kk < max_kk; kk++)
        {
            const int k = k0 + kk;
            float v = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
            v *= input_scale_ptr[k];
            outptr[k] = float2int8(v * scale);
        }
    }
}

#if defined(__GNUC__) && defined(__mips_loongson_mmi)
// NOTE gcc loongson mmi optimized version produce wrong result
// so I have to disable vectorize here
__attribute__((optimize("no-tree-vectorize")))
#endif
static int
gemm_transB_int8sa(const Mat& A, int transA, const Mat& BT, const Mat& BT_scales, const Mat& input_scales, const Mat& C, Mat& top_blob, int M, int N, int K, int block_size,
                   int quantize_term, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int block_count = (K + block_size - 1) / block_size;

    Mat A_int8;
    A_int8.create(K, M, (size_t)1u, opt.workspace_allocator);
    if (A_int8.empty())
        return -100;

    Mat A_descales;
    A_descales.create(block_count, M, (size_t)4u, opt.workspace_allocator);
    if (A_descales.empty())
        return -100;

    if (input_scales.empty() || quantize_term == 900)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < M; i++)
        {
            signed char* outptr = A_int8.row<signed char>(i);
            float* descale_ptr = A_descales.row(i);
            weight_block_quantize_activation_row_int8(A, transA, i, outptr, descale_ptr, K, block_size);
        }
    }
    else
    {
        const float* input_scale_ptr = input_scales;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < M; i++)
        {
            signed char* outptr = A_int8.row<signed char>(i);
            float* descale_ptr = A_descales.row(i);
            weight_block_scale_quantize_activation_row_int8(A, transA, i, outptr, descale_ptr, K, block_size, input_scale_ptr);
        }
    }

    const float* ptrC = C;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int mn = 0; mn < M * N; mn++)
    {
        const int i = mn / N;
        const int j = mn % N;

        float sum = 0.f;
        if (ptrC)
        {
            if (broadcast_type_C == 0)
                sum = ptrC[0];
            if (broadcast_type_C == 1)
                sum = ptrC[i];
            if (broadcast_type_C == 2)
                sum = ptrC[i];
            if (broadcast_type_C == 3)
                sum = ptrC[i * N + j];
            if (broadcast_type_C == 4)
                sum = ptrC[j];

            sum *= beta;
        }

        const signed char* ptrA = A_int8.row<const signed char>(i);
        const signed char* ptrB = BT.row<const signed char>(j);
        const float* A_descale_ptr = A_descales.row(i);
        const float* B_scale_ptr = BT_scales.row(j);

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = block_size < K - k0 ? block_size : K - k0;

            int sum_int32 = 0;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                sum_int32 += ptrA[k] * ptrB[k];
            }

            const float v = sum_int32 * A_descale_ptr[g];
            // Q8_0 stores a multiplier. Block-quantized weights store a divisor.
            sum += quantize_term == 900 ? v * B_scale_ptr[g] : v / B_scale_ptr[g];
        }

        sum *= alpha;

        if (output_transpose)
            ((float*)top_blob)[(size_t)j * out_hstep + i] = sum;
        else
            ((float*)top_blob)[(size_t)i * out_hstep + j] = sum;
    }

    return 0;
}
#endif // NCNN_WEIGHT_QUANT

Gemm::Gemm()
{
    one_blob_only = false;
    support_inplace = false;

    weight_block_quantize = 0;
    weight_block_quantize_bits = 0;
    weight_block_quantize_block_size = 0;
    weight_block_quantize_has_input_scale = false;
    weight_int8_rotation = 0;
}

int Gemm::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 1.f);
    beta = pd.get(1, 1.f);
    transA = pd.get(2, 0);
    transB = pd.get(3, 0);
    constantA = pd.get(4, 0);
    constantB = pd.get(5, 0);
    constantC = pd.get(6, 0);
    constantM = pd.get(7, 0);
    constantN = pd.get(8, 0);
    constantK = pd.get(9, 0);
    constant_broadcast_type_C = pd.get(10, 0);
    output_N1M = pd.get(11, 0);
    output_elempack = pd.get(12, 0);
    output_elemtype = pd.get(13, 0);
    output_transpose = pd.get(14, 0);
    quantize_term = pd.get(18, 0);
    weight_block_quantize_bits = 0;
    weight_block_quantize_block_size = 0;
    weight_block_quantize_has_input_scale = false;
    weight_block_quantize = get_weight_block_quantize_params(weight_block_quantize_bits, weight_block_quantize_block_size, weight_block_quantize_has_input_scale) == 0;
    constant_TILE_M = pd.get(20, 0);
    constant_TILE_N = pd.get(21, 0);
    constant_TILE_K = pd.get(22, 0);

    const bool weight_int8 = quantize_term >= 900 && quantize_term <= 902;
    weight_int8_rotation = weight_int8 ? pd.get(23, 0) : 0;

    if (output_elempack < 0)
    {
        NCNN_LOGE("Gemm invalid output_elempack %d", output_elempack);
        return -1;
    }

    if (quantize_term == 4 || quantize_term == 5 || quantize_term == 6)
    {
        NCNN_LOGE("Gemm unsupported quantize_term %d", quantize_term);
        return -1;
    }

    if (quantize_term >= 400 && !weight_block_quantize && !weight_int8)
    {
        NCNN_LOGE("Gemm unsupported quantize_term %d", quantize_term);
        return -1;
    }

    if (weight_int8)
    {
#if NCNN_WEIGHT_QUANT
        if (constantA != 0 || constantB != 1 || transB != 1 || (transA != 0 && transA != 1)
                || constantN <= 0 || constantK <= 0 || constantN > INT_MAX / constantK
                || (output_N1M != 0 && output_N1M != 1) || (output_transpose != 0 && output_transpose != 1)
                || (output_elemtype != 0 && output_elemtype != 1))
        {
            NCNN_LOGE("Gemm invalid constant B layout for int8 weights");
            return -1;
        }

        if (quantize_term == 900 && (constantK % 32 != 0 || weight_int8_rotation != 0))
        {
            NCNN_LOGE("Gemm Q8_0 requires K divisible by 32 and no rotation");
            return -1;
        }

        if (weight_int8_rotation != 0)
        {
            int group = weight_int8_rotation;
            while (group > 1 && group % 4 == 0)
                group /= 4;
            if (weight_int8_rotation < 4 || group != 1 || constantK % weight_int8_rotation != 0)
            {
                NCNN_LOGE("Gemm int8 rotation group %d must be a power of four dividing K", weight_int8_rotation);
                return -1;
            }
        }

        if (quantize_term == 902 && constantK > INT_MAX / (128 * 128))
        {
            NCNN_LOGE("Gemm K exceeds int32 accumulation range for channel-wise W8A8");
            return -1;
        }

        weight_block_quantize_bits = 8;
        weight_block_quantize_block_size = quantize_term == 900 ? 32 : constantK;

        support_packing = false;
        support_fp16_storage = false;
        support_bf16_storage = false;
#else
        NCNN_LOGE("please build ncnn with NCNN_WEIGHT_QUANT enabled for weight quantized inference");
        return -1;
#endif
    }
    else if (weight_block_quantize)
    {
#if NCNN_WEIGHT_QUANT
        if (constantA != 0 || constantB != 1 || transB != 1 || (transA != 0 && transA != 1) || constantN <= 0 || constantK <= 0)
        {
            NCNN_LOGE("Gemm unsupported weight block quantize");
            return -1;
        }

        if ((output_elemtype != 0 && output_elemtype != 1) || (output_transpose != 0 && output_transpose != 1))
        {
            NCNN_LOGE("Gemm unsupported weight block quantize");
            return -1;
        }

        if (weight_block_quantize_bits != 8)
        {
            support_packing = false;
            support_bf16_storage = false;
            support_fp16_storage = false;
        }
        if (weight_block_quantize_bits == 8 || weight_block_quantize_has_input_scale)
        {
            support_vulkan = false;
            support_vulkan_packing = false;
        }
#else
        NCNN_LOGE("please build ncnn with NCNN_WEIGHT_QUANT enabled for weight quantized inference");
        return -1;
#endif
    }
    else if (quantize_term)
    {
#if !NCNN_INT8
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

#if NCNN_VALIDATION
    if (constantA == 1 && (constantM <= 0 || constantK <= 0))
        return -1;

    if (constantB == 1 && (constantN <= 0 || constantK <= 0))
        return -1;

    if (constantC == 1 && (constant_broadcast_type_C < -1 || constant_broadcast_type_C > 4))
    {
        NCNN_LOGE("constant_broadcast_type_C must be -1 or 0~4 when constantC enabled");
        return -1;
    }
#endif // NCNN_VALIDATION

    if (constantA == 0 && constantB == 1 && constantC == 1)
        one_blob_only = true;

    if (constantA == 1 && constantB == 0 && constantC == 1)
        one_blob_only = true;

    if (constantA == 1 && constantB == 1 && constantC == 0)
        one_blob_only = true;

    return 0;
}

int Gemm::load_model(const ModelBin& mb)
{
    const bool weight_int8 = quantize_term >= 900 && quantize_term <= 902;

    if (constantA == 1)
    {
        if (transA == 0)
            A_data = mb.load(constantK, constantM, 0);
        else
            A_data = mb.load(constantM, constantK, 0);
        if (A_data.empty())
            return -100;
    }

    if (constantB == 1)
    {
        if (transB == 0)
            B_data = mb.load(constantN, constantK, 0);
#if NCNN_WEIGHT_QUANT
        else if (weight_int8)
        {
            B_data = mb.load(constantK, constantN, 3);
            if (B_data.empty() || B_data.elemsize != 1u)
                return -100;
        }
        else if (weight_block_quantize)
        {
            const int packed_k_bytes = gemm_weight_quantize_packed_k_bytes(constantK, weight_block_quantize_bits);
            if (packed_k_bytes <= 0 || constantN > INT_MAX / packed_k_bytes)
                return -100;
            B_data = mb.load(packed_k_bytes, constantN, weight_block_quantize_bits);
            if (B_data.empty() || B_data.elemsize != 1u)
                return -100;
        }
#endif // NCNN_WEIGHT_QUANT
        else
            B_data = mb.load(constantK, constantN, 0);
        if (B_data.empty())
            return -100;
    }

    if (constantC == 1 && constant_broadcast_type_C != -1)
    {
        if (constant_broadcast_type_C == 0)
            C_data = mb.load(1, 0);
        if (constant_broadcast_type_C == 1)
            C_data = mb.load(constantM, 0);
        if (constant_broadcast_type_C == 2)
            C_data = mb.load(1, constantM, 0);
        if (constant_broadcast_type_C == 3)
            C_data = mb.load(constantN, constantM, 0);
        if (constant_broadcast_type_C == 4)
            C_data = mb.load(constantN, 1, 0);
        if (C_data.empty())
            return -100;
    }

#if NCNN_WEIGHT_QUANT
    if (weight_int8)
    {
        if (quantize_term == 900)
        {
            const int blocks = constantK / 32;
            const int count = blocks * constantN;
            Mat scales = mb.load(count * 2, 3);
            if (scales.empty() || scales.elemsize != 1u || scales.w != count * 2)
                return -100;

            B_data_quantize_scales.create(blocks, constantN);
            if (B_data_quantize_scales.empty())
                return -100;

            const unsigned char* ptr = scales;
            for (int i = 0; i < count; i++)
            {
                unsigned short fp16 = ptr[0] | (ptr[1] << 8);
                if ((fp16 & 0x7c00) == 0x7c00)
                {
                    NCNN_LOGE("Gemm Q8_0 weight scale is not finite");
                    return -1;
                }
                B_data_quantize_scales[i] = float16_to_float32(fp16);
                ptr += 2;
            }
        }
        else
        {
            B_data_quantize_scales = mb.load(1, constantN, 1);
            if (B_data_quantize_scales.empty() || B_data_quantize_scales.elemsize != 4u)
                return -100;

            const float* scales = B_data_quantize_scales;
            for (int i = 0; i < constantN; i++)
            {
                // inspect the stored bits because fast math assumes finite floats
                uint32_t bits;
                memcpy(&bits, scales + i, sizeof(bits));
                if ((bits & 0x7f800000) == 0x7f800000 || bits > 0x80000000)
                {
                    NCNN_LOGE("Gemm invalid int8 weight scale at output channel %d", i);
                    return -1;
                }
            }
        }
    }
    else if (weight_block_quantize)
    {
        const int block_count = (constantK - 1) / weight_block_quantize_block_size + 1;

        B_data_quantize_scales = mb.load(block_count, constantN, 1);
        if (B_data_quantize_scales.empty())
            return -100;

        if (weight_block_quantize_bits != 8)
        {
            const float* scales = B_data_quantize_scales;
            for (int i = 0; i < block_count * constantN; i++)
            {
                uint32_t bits;
                memcpy(&bits, scales + i, sizeof(bits));
                if (bits == 0 || bits >= 0x7f800000)
                {
                    NCNN_LOGE("Gemm invalid weight block scale at index %d", i);
                    return -1;
                }
            }
        }

        if (weight_block_quantize_has_input_scale)
        {
            B_data_input_scales = mb.load(constantK, 1);
            if (B_data_input_scales.empty())
                return -100;
        }
    }
#endif // NCNN_WEIGHT_QUANT

#if NCNN_INT8
    if (quantize_term && !weight_block_quantize && !weight_int8)
    {
        if (constantA == 1)
        {
            A_data_int8_scales = mb.load(constantM, 1);
        }

        if (constantB == 1)
        {
            B_data_int8_scale = mb.load(1, 1)[0];
        }
    }
#endif // NCNN_INT8

    return 0;
}

static void gemm_transB(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;
    const int K = A.w; // assert A.w == BT.w

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        const size_t BT_hstep = BT.dims == 3 ? BT.cstep : (size_t)BT.w;

        const float* ptrA = (const float*)A + i * A_hstep;
        const float* ptrC = C;

        for (int j = 0; j < N; j++)
        {
            const float* ptrBT = (const float*)BT + j * BT_hstep;

            float sum = 0.f;
            if (ptrC)
            {
                if (broadcast_type_C == 0)
                {
                    sum = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    sum = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    sum = ptrC[j];
                }

                sum *= beta;
            }

            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrBT[k];
            }

            sum *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum;
            }
        }
    }
}

#if NCNN_WEIGHT_QUANT
static void dequantize_B_row_int4_scalar(const unsigned char* p, const float* scales, int K, int block_size, float* outptr)
{
    const int block_count = (K - 1) / block_size + 1;
    for (int g = 0; g < block_count; g++)
    {
        const int max_kk = std::min(K - g * block_size, block_size);
        const float descale = 1.f / scales[g];

        int kk = 0;
        for (; kk + 1 < max_kk; kk += 2)
        {
            const int v = *p++;
            outptr[0] = (((v & 15) ^ 8) - 8) * descale;
            outptr[1] = (((v >> 4) ^ 8) - 8) * descale;
            outptr += 2;
        }
        if (kk < max_kk)
            *outptr++ = (((*p & 15) ^ 8) - 8) * descale;
    }
}

static void dequantize_B_row_int6_scalar(const unsigned char* p, const float* scales, int K, int block_size, float* outptr)
{
    const int block_count = (K - 1) / block_size + 1;
    for (int g = 0; g < block_count; g++)
    {
        const int max_kk = std::min(K - g * block_size, block_size);
        const float descale = 1.f / scales[g];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            const int v = p[0] | (p[1] << 8) | (p[2] << 16);
            outptr[0] = (((v & 63) ^ 32) - 32) * descale;
            outptr[1] = ((((v >> 6) & 63) ^ 32) - 32) * descale;
            outptr[2] = ((((v >> 12) & 63) ^ 32) - 32) * descale;
            outptr[3] = (((v >> 18) ^ 32) - 32) * descale;
            p += 3;
            outptr += 4;
        }
        for (int kk0 = 0; kk < max_kk; kk++, kk0++)
        {
            const int byte_offset = kk0 * 6 / 8;
            const int shift = kk0 * 6 % 8;
            unsigned int v = p[byte_offset];
            if (shift > 2)
                v |= (unsigned int)p[byte_offset + 1] << 8;
            const int w = (v >> shift) & 63;
            *outptr++ = ((w ^ 32) - 32) * descale;
        }
    }
}

int Gemm::forward_weight_block_quantize(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.empty() || top_blobs.empty())
        return -1;

    const Mat& A = bottom_blobs[0];

    const int K = transA ? (A.dims == 3 ? A.c : A.h) * A.elempack : A.w;
    const int block_size = weight_block_quantize_block_size;

    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int N = constantN;
    if (K != constantK || M <= 0 || A.empty() || A.elempack != 1 || A.elemsize != 4u || A.dims > 3 || (A.dims == 3 && A.h != 1))
        return -1;

    Mat C;
    const int broadcast_type_C = get_weight_quantize_broadcast_type_C(bottom_blobs, C, M);
    if (!C.empty() && broadcast_type_C < 0)
        return -1;

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, (size_t)4u, opt.blob_allocator);
        else
            top_blob.create(M, N, (size_t)4u, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, (size_t)4u, opt.blob_allocator);
        else
            top_blob.create(N, M, (size_t)4u, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const float* input_scale_ptr = B_data_input_scales;
    const float* ptrC = C;

    Mat A_rows;
    if (transA)
    {
        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        A_rows.create(K, M, (size_t)4u, opt.workspace_allocator);
        if (A_rows.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < M; i++)
        {
            float* outptr = A_rows.row(i);
            for (int k = 0; k < K; k++)
                outptr[k] = ((const float*)A)[k * A_hstep + i];
        }
    }
    else
    {
        A_rows = A;
    }
    const size_t A_rows_hstep = A_rows.dims == 3 ? A_rows.cstep : (size_t)A_rows.w;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const int num_threads = std::min(opt.num_threads, N);
    Mat B_row(K, num_threads, (size_t)4u, opt.workspace_allocator);
    if (B_row.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int j = 0; j < N; j++)
    {
        float* decoded = B_row.row(get_omp_thread_num());

        // reuse the decoded row for all input tokens
        const unsigned char* ptr = B_data.row<const unsigned char>(j);
        const float* scales = B_data_quantize_scales.row(j);
        if (weight_block_quantize_bits == 4)
            dequantize_B_row_int4_scalar(ptr, scales, K, block_size, decoded);
        else
            dequantize_B_row_int6_scalar(ptr, scales, K, block_size, decoded);

        for (int i = 0; i < M; i++)
        {
            const float* ptrA = (const float*)A_rows + i * A_rows_hstep;
            float sum = 0.f;
            if (ptrC)
            {
                if (broadcast_type_C == 0)
                    sum = ptrC[0];
                if (broadcast_type_C == 1)
                    sum = ptrC[i];
                if (broadcast_type_C == 2)
                    sum = ptrC[i];
                if (broadcast_type_C == 3)
                    sum = ptrC[(size_t)i * N + j];
                if (broadcast_type_C == 4)
                    sum = ptrC[j];

                sum *= beta;
            }

            if (input_scale_ptr)
            {
                for (int k = 0; k < K; k++)
                    sum += ptrA[k] * input_scale_ptr[k] * decoded[k];
            }
            else
            {
                for (int k = 0; k < K; k++)
                    sum += ptrA[k] * decoded[k];
            }

            ((float*)top_blob)[output_transpose ? j * out_hstep + i : i * out_hstep + j] = sum * alpha;
        }
    }

    B_row.release();
    A_rows.release();

    if (output_elempack > 1)
    {
        Mat packed;
        convert_packing(top_blob, packed, output_elempack, opt);
        if (packed.empty())
            return -100;
        top_blob = packed;
    }

    return 0;
}

static void rotate_convrot_row(float* ptr, int K, int group)
{
    // Kronecker powers of the normalized regular 4x4 Hadamard matrix.
    for (int k = 0; k < K; k += group)
    {
        float* g = ptr + k;
        for (int stride = 1; stride < group; stride *= 4)
        {
            for (int i = 0; i < group; i += stride * 4)
            {
                for (int j = 0; j < stride; j++)
                {
                    float* v = g + i + j;
                    const float a = v[0];
                    const float b = v[stride];
                    const float c = v[stride * 2];
                    const float d = v[stride * 3];
                    const float sum = (a + b + c + d) * 0.5f;
                    v[0] = sum - d;
                    v[stride] = sum - c;
                    v[stride * 2] = sum - b;
                    v[stride * 3] = sum - a;
                }
            }
        }
    }
}

int Gemm::get_weight_quantize_broadcast_type_C(const std::vector<Mat>& bottom_blobs, Mat& C, int M) const
{
    if (constantC)
    {
        C = C_data;
        if (weight_block_quantize && weight_block_quantize_bits != 8
                && constant_broadcast_type_C >= 1 && constant_broadcast_type_C <= 3 && M != constantM)
            return -1;
        return constant_broadcast_type_C;
    }

    if (bottom_blobs.size() != 2)
        return -1;

    C = bottom_blobs[1];
    const int N = constantN;
    if ((C.dims == 1 && C.w == N) || (C.dims == 2 && C.w == N && C.h == 1))
        return 4;
    if (C.dims == 2 && C.w == N && C.h == M)
        return 3;
    if (C.dims == 2 && C.w == 1 && C.h == M)
        return 2;
    if (C.dims == 1 && C.w == M)
        return 1;
    if (C.dims == 1 && C.w == 1)
        return 0;

    return weight_block_quantize && weight_block_quantize_bits != 8 ? 0 : -1;
}

int Gemm::forward_int8s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.empty() || top_blobs.empty())
        return -1;

    const Mat& A = bottom_blobs[0];
    const int rows = A.dims == 3 ? A.c : A.h;
    const int M = transA ? A.w : rows;
    const int K = transA ? rows : A.w;
    const int N = constantN;
    if (A.empty() || A.dims > 3 || (A.dims == 3 && A.h != 1) || A.elempack != 1 || A.elemsize != 4u
            || M <= 0 || N > INT_MAX / M || K != constantK || (constantM > 0 && M != constantM))
    {
        NCNN_LOGE("Gemm invalid activation shape for int8 weights");
        return -1;
    }

    Mat C;
    const int broadcast_type_C = get_weight_quantize_broadcast_type_C(bottom_blobs, C, M);
    if (!C.empty() && (broadcast_type_C < 0 || C.elemsize != 4u || C.elempack != 1))
    {
        NCNN_LOGE("Gemm invalid addend for int8 weights");
        return -1;
    }

    const int outw = output_transpose ? M : N;
    const int outh = output_transpose ? N : M;
    Mat& top_blob = top_blobs[0];
    if (output_N1M)
        top_blob.create(outw, 1, outh, (size_t)4u, opt.blob_allocator);
    else
        top_blob.create(outw, outh, (size_t)4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const bool block32 = quantize_term == 900;
    const int num_threads = std::min(opt.num_threads, M);
    Mat A_work;
    if (transA || weight_int8_rotation)
    {
        A_work.create(K, num_threads, (size_t)4u, opt.workspace_allocator);
        if (A_work.empty())
            return -100;
    }

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const size_t out_hstep = output_N1M ? top_blob.cstep : (size_t)outw;
    const float* ptrC = C;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++)
    {
        const int tid = get_omp_thread_num();
        const float* ptrA = (const float*)A + (transA ? 0 : i * A_hstep);
        if (!A_work.empty())
        {
            float* ptr = A_work.row(tid);
            for (int k = 0; k < K; k++)
                ptr[k] = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
            if (weight_int8_rotation)
                rotate_convrot_row(ptr, K, weight_int8_rotation);
            ptrA = ptr;
        }

        for (int j = 0; j < N; j++)
        {
            const signed char* ptrB = B_data.row<const signed char>(j);
            float sum = 0.f;
            if (block32)
            {
                const float* scales = B_data_quantize_scales.row(j);
                for (int k = 0; k < K; k += 32)
                {
                    float dot = 0.f;
                    for (int kk = 0; kk < 32; kk++)
                        dot += ptrA[k + kk] * ptrB[k + kk];
                    sum += dot * scales[k / 32];
                }
            }
            else
            {
                for (int k = 0; k < K; k++)
                    sum += ptrA[k] * ptrB[k];
                sum *= B_data_quantize_scales[j];
            }

            if (ptrC)
            {
                if (broadcast_type_C == 0)
                    sum += ptrC[0] * beta;
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    sum += ptrC[i] * beta;
                if (broadcast_type_C == 3)
                    sum += ptrC[i * N + j] * beta;
                if (broadcast_type_C == 4)
                    sum += ptrC[j] * beta;
            }

            const size_t offset = output_transpose ? j * out_hstep + i : i * out_hstep + j;
            ((float*)top_blob)[offset] = sum * alpha;
        }
    }

    if (output_elempack > 1)
    {
        Mat top_blob_packed;
        convert_packing(top_blob, top_blob_packed, output_elempack, opt);
        if (top_blob_packed.empty())
            return -100;
        top_blob = top_blob_packed;
    }

    return 0;
}

int Gemm::forward_int8sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.empty() || top_blobs.empty())
        return -1;

    const Mat& A = bottom_blobs[0];
    const int rows = (A.dims == 3 ? A.c : A.h) * A.elempack;
    const int M = transA ? A.w : rows;
    const int K = transA ? rows : A.w;
    const int N = constantN;
    if (K != constantK || M <= 0)
        return -1;

    if (!weight_block_quantize && (A.empty() || A.dims > 3 || (A.dims == 3 && A.h != 1) || A.elempack != 1 || A.elemsize != 4u || N > INT_MAX / M || (constantM > 0 && M != constantM)))
    {
        NCNN_LOGE("Gemm invalid activation shape for int8 weights");
        return -1;
    }

    Mat C;
    int broadcast_type_C = get_weight_quantize_broadcast_type_C(bottom_blobs, C, M);
    if (weight_block_quantize && !constantC && broadcast_type_C < 0)
        broadcast_type_C = 0;
    if (!weight_block_quantize && !C.empty() && (broadcast_type_C < 0 || C.elemsize != 4u || C.elempack != 1))
    {
        NCNN_LOGE("Gemm invalid addend for int8 weights");
        return -1;
    }

    const int outw = output_transpose ? M : N;
    const int outh = output_transpose ? N : M;
    Mat& top_blob = top_blobs[0];
    if (output_N1M)
        top_blob.create(outw, 1, outh, (size_t)4u, opt.blob_allocator);
    else
        top_blob.create(outw, outh, (size_t)4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (weight_block_quantize || quantize_term == 900)
    {
        int ret = gemm_transB_int8sa(A, transA, B_data, B_data_quantize_scales, B_data_input_scales, C, top_blob, M, N, K, weight_block_quantize_block_size,
                                     quantize_term, alpha, beta, broadcast_type_C, output_transpose, opt);
        if (ret != 0 || weight_block_quantize)
            return ret;
    }
    else
    {
        const int num_threads = std::min(opt.num_threads, M);
        Mat A_work;
        if (transA || weight_int8_rotation)
        {
            A_work.create(K, num_threads, (size_t)4u, opt.workspace_allocator);
            if (A_work.empty())
                return -100;
        }

        Mat A_int8(K, num_threads, (size_t)1u, opt.workspace_allocator);
        if (A_int8.empty())
            return -100;

        const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
        const size_t out_hstep = output_N1M ? top_blob.cstep : (size_t)outw;
        const float* ptrC = C;

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < M; i++)
        {
            const int tid = get_omp_thread_num();
            const float* ptrA = (const float*)A + (transA ? 0 : i * A_hstep);
            if (!A_work.empty())
            {
                float* ptr = A_work.row(tid);
                for (int k = 0; k < K; k++)
                    ptr[k] = transA ? ((const float*)A)[k * A_hstep + i] : ptrA[k];
                if (weight_int8_rotation)
                    rotate_convrot_row(ptr, K, weight_int8_rotation);
                ptrA = ptr;
            }

            float absmax = 0.f;
            for (int k = 0; k < K; k++)
            {
                const float a = fabsf(ptrA[k]);
                if (a > absmax)
                    absmax = a;
            }
            const float a_descale = absmax > 1.27e-28f ? absmax / 127.f : 1e-30f;
            signed char* ptrA_int8 = A_int8.row<signed char>(tid);
            for (int k = 0; k < K; k++)
                ptrA_int8[k] = float2int8(ptrA[k] / a_descale);

            for (int j = 0; j < N; j++)
            {
                const signed char* ptrB = B_data.row<const signed char>(j);
                int dot = 0;
                for (int k = 0; k < K; k++)
                    dot += ptrA_int8[k] * ptrB[k];
                float sum = dot * a_descale * B_data_quantize_scales[j];

                if (ptrC)
                {
                    if (broadcast_type_C == 0)
                        sum += ptrC[0] * beta;
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        sum += ptrC[i] * beta;
                    if (broadcast_type_C == 3)
                        sum += ptrC[i * N + j] * beta;
                    if (broadcast_type_C == 4)
                        sum += ptrC[j] * beta;
                }

                const size_t offset = output_transpose ? j * out_hstep + i : i * out_hstep + j;
                ((float*)top_blob)[offset] = sum * alpha;
            }
        }
    }

    if (output_elempack > 1)
    {
        Mat top_blob_packed;
        convert_packing(top_blob, top_blob_packed, output_elempack, opt);
        if (top_blob_packed.empty())
            return -100;
        top_blob = top_blob_packed;
    }

    return 0;
}
#endif // NCNN_WEIGHT_QUANT

#if NCNN_INT8
static void gemm_transB_int8(const Mat& A_int8, const Mat& BT_int8, const Mat& A_int8_scales, float BT_int8_scale, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A_int8.h;
    const int N = BT_int8.h;
    const int K = A_int8.w; // assert A_int8.w == BT_int8.w

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

        const signed char* ptrA = A_int8.row<const signed char>(i);
        const float* ptrC = C;

        const float descale = 1.f / (A_int8_scales[i] * BT_int8_scale);

        for (int j = 0; j < N; j++)
        {
            const signed char* ptrBT = BT_int8.row<const signed char>(j);

            int sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrBT[k];
#if __mips_loongson_mmi && !__mips_msa
                // workaround gcc mis-vectorization on loongson-mmi
                asm volatile("" ::
                             : "memory");
#endif
            }

            float sum_fp32 = sum * descale;

            if (ptrC)
            {
                float c = 0.f;
                if (broadcast_type_C == 0)
                {
                    c = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    c = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    c = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    c = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    c = ptrC[j];
                }

                sum_fp32 += c * beta;
            }

            sum_fp32 *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum_fp32;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum_fp32;
            }
        }
    }
}
#endif // NCNN_INT8

int Gemm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1, top_blob);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Gemm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_WEIGHT_QUANT
    if (quantize_term >= 900 && quantize_term <= 902)
    {
        if (quantize_term != 901 && opt.use_int8_inference)
            return forward_int8sa(bottom_blobs, top_blobs, opt);
        return forward_int8s(bottom_blobs, top_blobs, opt);
    }

    if (weight_block_quantize)
    {
        if (weight_block_quantize_bits == 8)
            return forward_int8sa(bottom_blobs, top_blobs, opt);
        return forward_weight_block_quantize(bottom_blobs, top_blobs, opt);
    }
#else
    if (weight_block_quantize)
    {
        NCNN_LOGE("please build ncnn with NCNN_WEIGHT_QUANT enabled for weight quantized inference");
        return -1;
    }
#endif

#if NCNN_INT8
    if (quantize_term)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#else
    if (quantize_term)
    {
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
    }
#endif // NCNN_INT8

    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        A.create((A0.dims == 3 ? A0.c : A0.h), A0.w, elemsize, opt.workspace_allocator);
        if (A.empty())
            return -100;

        const size_t A0_hstep = A0.dims == 3 ? A0.cstep : (size_t)A0.w;

        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++)
            {
                ptr[j] = A0[j * A0_hstep + i];
            }
        }
    }

    Mat BT;
    if (transB == 0)
    {
        // transpose B to col-major
        BT.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);
        if (BT.empty())
            return -100;

        const size_t B0_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;

        for (int i = 0; i < BT.h; i++)
        {
            float* ptr = BT.row(i);
            for (int j = 0; j < BT.w; j++)
            {
                ptr[j] = B0[j * B0_hstep + i];
            }
        }
    }
    else
    {
        BT = B0;
    }

    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB && bottom_blobs.size() == 1)
        {
            C = bottom_blobs[0];
        }
        else if ((constantA || constantB) && bottom_blobs.size() == 2)
        {
            C = bottom_blobs[1];
        }
        else if (bottom_blobs.size() == 3)
        {
            C = bottom_blobs[2];
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, elemsize, opt.blob_allocator);
        else
            top_blob.create(M, N, elemsize, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, elemsize, opt.blob_allocator);
        else
            top_blob.create(N, M, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    gemm_transB(A, BT, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);

    return 0;
}

#if NCNN_INT8
int Gemm::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        if (A0.elemsize == 1)
        {
            A.create(A0.h, A0.w, (size_t)1u, 1, opt.workspace_allocator);
            if (A.empty())
                return -100;

            for (int i = 0; i < A.h; i++)
            {
                signed char* ptr = A.row<signed char>(i);
                for (int j = 0; j < A.w; j++)
                {
                    ptr[j] = A0.row<const signed char>(j)[i];
                }
            }
        }
        else
        {
            A.create(A0.dims == 3 ? A0.c : A0.h, A0.w, (size_t)4u, 1, opt.workspace_allocator);
            if (A.empty())
                return -100;

            for (int i = 0; i < A.h; i++)
            {
                float* ptr = A.row(i);
                for (int j = 0; j < A.w; j++)
                {
                    ptr[j] = A0.dims == 3 ? A0.channel(j)[i] : A0.row(j)[i];
                }
            }
        }
    }

    // dynamic quantize A
    Mat A_int8 = A;
    Mat A_int8_scales = A_data_int8_scales;
    if (A_int8.elemsize != 1)
    {
        A_int8.create(A.w, A.dims == 3 ? A.c : A.h, (size_t)1u, 1, opt.workspace_allocator);
        if (A_int8.empty())
            return -100;
        A_int8_scales.create(A_int8.h, (size_t)4u, 1, opt.workspace_allocator);
        if (A_int8_scales.empty())
            return -100;

        for (int i = 0; i < A_int8.h; i++)
        {
            const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
            const float* ptr = (const float*)A + i * A_hstep;

            float absmax = 0.f;
            for (int k = 0; k < A_int8.w; k++)
            {
                absmax = std::max(absmax, (float)fabs(ptr[k]));
            }

            float A_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;
            A_int8_scales[i] = A_int8_scale;

            signed char* ptrAi = A_int8.row<signed char>(i);

            for (int k = 0; k < A_int8.w; k++)
            {
                ptrAi[k] = float2int8(ptr[k] * A_int8_scale);
            }
        }
    }

    // dynamic quantize B
    Mat B0_int8 = B0;
    float B_int8_scale = B_data_int8_scale;
    if (B0_int8.elemsize != 1)
    {
        B0_int8.create(B0.w, B0.dims == 3 ? B0.c : B0.h, (size_t)1u, 1, opt.workspace_allocator);
        if (B0_int8.empty())
            return -100;

        float absmax = 0.f;
        for (int i = 0; i < B0_int8.h; i++)
        {
            const size_t B_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;
            const float* ptr = (const float*)B0 + i * B_hstep;

            for (int k = 0; k < B0_int8.w; k++)
            {
                absmax = std::max(absmax, (float)fabs(ptr[k]));
            }
        }

        B_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;

        for (int i = 0; i < B0_int8.h; i++)
        {
            const size_t B_hstep = B0.dims == 3 ? B0.cstep : (size_t)B0.w;
            const float* ptr = (const float*)B0 + i * B_hstep;

            signed char* ptrBi = B0_int8.row<signed char>(i);

            for (int k = 0; k < B0_int8.w; k++)
            {
                ptrBi[k] = float2int8(ptr[k] * B_int8_scale);
            }
        }
    }

    Mat BT_int8;
    if (transB == 0)
    {
        // transpose B to col-major
        BT_int8.create(B0_int8.h, B0_int8.w, (size_t)1u, 1, opt.workspace_allocator);
        if (BT_int8.empty())
            return -100;

        for (int i = 0; i < BT_int8.h; i++)
        {
            signed char* ptr = BT_int8.row<signed char>(i);
            for (int j = 0; j < BT_int8.w; j++)
            {
                ptr[j] = B0_int8.row<const signed char>(j)[i];
            }
        }
    }
    else
    {
        BT_int8 = B0_int8;
    }

    const int M = A_int8.h;
    const int N = BT_int8.h;

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB && bottom_blobs.size() == 1)
        {
            C = bottom_blobs[0];
        }
        else if ((constantA || constantB) && bottom_blobs.size() == 2)
        {
            C = bottom_blobs[1];
        }
        else if (bottom_blobs.size() == 3)
        {
            C = bottom_blobs[2];
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, 4u, opt.blob_allocator);
        else
            top_blob.create(M, N, 4u, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, 4u, opt.blob_allocator);
        else
            top_blob.create(N, M, 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    gemm_transB_int8(A_int8, BT_int8, A_int8_scales, B_int8_scale, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
