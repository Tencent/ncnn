// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "prelu_bf16s.h"
#endif

PReLU_x86::PReLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int PReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    if (dims == 1)
    {
        const int size = w * elempack;

        if (num_slope > 1)
        {
            float* ptr = bottom_top_blob;
            const float* slope = slope_data;

            int nn_size = 0;
            int remain_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            nn_size = (size - remain_size_start) / 16;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 16;
                __m512 _p512 = _mm512_loadu_ps(ptr + i);
                __m512 _slope512 = _mm512_loadu_ps(slope + i);
                _mm512_storeu_ps(ptr + i, prelu_avx512(_p512, _slope512));
            }
            remain_size_start += nn_size * 16;
#endif // __AVX512F__
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p256 = _mm256_loadu_ps(ptr + i);
                __m256 _slope256 = _mm256_loadu_ps(slope + i);
                _mm256_storeu_ps(ptr + i, prelu_avx(_p256, _slope256));
            }
            remain_size_start += nn_size * 8;
#endif // __AVX__
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p128 = _mm_load_ps(ptr + i);
                __m128 _slope128 = _mm_loadu_ps(slope + i);
                _mm_store_ps(ptr + i, prelu_sse(_p128, _slope128));
            }
            remain_size_start += nn_size * 4;
#endif // __SSE2__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
        }
        else
        {
            float* ptr = bottom_top_blob;
            const float slope = slope_data[0];

            int nn_size = 0;
            int remain_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            nn_size = (size - remain_size_start) / 16;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 16;
                __m512 _p512 = _mm512_loadu_ps(ptr + i);
                __m512 _slope512 = _mm512_set1_ps(slope);
                _mm512_storeu_ps(ptr + i, prelu_avx512(_p512, _slope512));
            }
            remain_size_start += nn_size * 16;
#endif // __AVX512F__
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p256 = _mm256_loadu_ps(ptr + i);
                __m256 _slope256 = _mm256_set1_ps(slope);
                _mm256_storeu_ps(ptr + i, prelu_avx(_p256, _slope256));
            }
            remain_size_start += nn_size * 8;
#endif // __AVX__
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p128 = _mm_load_ps(ptr + i);
                __m128 _slope128 = _mm_set1_ps(slope);
                _mm_store_ps(ptr + i, prelu_sse(_p128, _slope128));
            }
            remain_size_start += nn_size * 4;
#endif // __SSE2__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            int j = 0;

            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];
#if __SSE2__
            __m128 _slope128 = num_slope > 1 && (elempack == 4) ? _mm_loadu_ps((const float*)slope_data + i * 4) : _mm_set1_ps(slope);
#if __AVX__
            __m256 _slope256 = num_slope > 1 && (elempack == 8) ? _mm256_loadu_ps((const float*)slope_data + i * 8) : combine4x2_ps(_slope128, _slope128);
#if __AVX512F__
            __m512 _slope512 = num_slope > 1 && (elempack == 16) ? _mm512_loadu_ps((const float*)slope_data + i * 16) : combine8x2_ps(_slope256, _slope256);

            for (; j + 15 < size; j += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));
                ptr += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < size; j += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));
                ptr += 8;
            }
#endif // __AVX__
            for (; j + 3 < size; j += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _mm_storeu_ps(ptr, prelu_sse(_p128, _slope128));
                ptr += 4;
            }
#endif // __SSE2__
            for (; j < size; j++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            int i = 0;

            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];
#if __SSE2__
            __m128 _slope128 = num_slope > 1 && (elempack == 4) ? _mm_loadu_ps((const float*)slope_data + q * 4) : _mm_set1_ps(slope);
#if __AVX__
            __m256 _slope256 = num_slope > 1 && (elempack == 8) ? _mm256_loadu_ps((const float*)slope_data + q * 8) : combine4x2_ps(_slope128, _slope128);
#if __AVX512F__
            __m512 _slope512 = num_slope > 1 && (elempack == 16) ? _mm512_loadu_ps((const float*)slope_data + q * 16) : combine8x2_ps(_slope256, _slope256);

            for (; i + 15 < size; i += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, prelu_avx512(_p512, _slope512));
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, prelu_avx(_p256, _slope256));
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p128 = _mm_load_ps(ptr);
                _mm_store_ps(ptr, prelu_sse(_p128, _slope128));
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int PReLU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (num_slope > 1)
        {
            // per-element slope
            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
                __m512 _slope = _mm512_loadu_ps((const float*)slope_data + i);
                __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                __m512 _ps = _mm512_mul_ps(_p, _slope);
                _p = _mm512_mask_mov_ps(_p, _mask, _ps);
                _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
                __m256 _slope = _mm256_loadu_ps((const float*)slope_data + i);
                __m256 _ps = _mm256_mul_ps(_p, _slope);
                _p = _mm256_blendv_ps(_p, _ps, _mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ));
                _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
                __m128 _slope = _mm_loadu_ps((const float*)slope_data + i);
                __m128 _ps = _mm_mul_ps(_p, _slope);
                __m128 _mask = _mm_cmplt_ps(_p, _zero);
                _p = _mm_or_ps(_mm_andnot_ps(_mask, _p), _mm_and_ps(_mask, _ps));
                _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    v *= slope_data[i];
                ptr[i] = float32_to_bfloat16(v);
            }
        }
        else
        {
            const float slope = slope_data[0];
            prelu_bf16s_sse(ptr, &slope, size, 1);
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];
            const float* sptr = num_slope > 1 ? (const float*)slope_data + i * elempack : &slope;
            int ep = num_slope > 1 ? elempack : 1;

            prelu_bf16s_sse(ptr, sptr, size, ep);
        }
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];
            const float* sptr = num_slope > 1 ? (const float*)slope_data + q * elempack : &slope;
            int ep = num_slope > 1 ? elempack : 1;

            prelu_bf16s_sse(ptr, sptr, size, ep);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
