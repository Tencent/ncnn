// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "scale_bf16s.h"
#endif

Scale_x86::Scale_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_x86::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;

    const int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    if (dims == 1)
    {
        float* ptr = (float*)bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
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
                __m512 _s512 = _mm512_loadu_ps(scale + i);
                __m512 _bias512 = _mm512_loadu_ps(bias + i);
                _mm512_storeu_ps(ptr + i, _mm512_fmadd_ps(_p512, _s512, _bias512));
            }
            remain_size_start += nn_size * 16;
#endif // __AVX512F__
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p256 = _mm256_loadu_ps(ptr + i);
                __m256 _s256 = _mm256_loadu_ps(scale + i);
                __m256 _bias256 = _mm256_loadu_ps(bias + i);
                _mm256_storeu_ps(ptr + i, _mm256_comp_fmadd_ps(_p256, _s256, _bias256));
            }
            remain_size_start += nn_size * 8;
#endif // __AVX__
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p128 = _mm_load_ps(ptr + i);
                __m128 _s128 = _mm_load_ps(scale + i);
                __m128 _bias128 = _mm_loadu_ps(bias + i);
                _mm_store_ps(ptr + i, _mm_comp_fmadd_ps(_p128, _s128, _bias128));
            }
            remain_size_start += nn_size * 4;
#endif // __SSE2__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
        }
        else
        {
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
                __m512 _s512 = _mm512_loadu_ps(scale + i);
                _mm512_storeu_ps(ptr + i, _mm512_mul_ps(_p512, _s512));
            }
            remain_size_start += nn_size * 16;
#endif // __AVX512F__
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p256 = _mm256_loadu_ps(ptr + i);
                __m256 _s256 = _mm256_loadu_ps(scale + i);
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(_p256, _s256));
            }
            remain_size_start += nn_size * 8;
#endif // __AVX__
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p128 = _mm_load_ps(ptr + i);
                __m128 _s128 = _mm_load_ps(scale + i);
                _mm_store_ps(ptr + i, _mm_mul_ps(_p128, _s128));
            }
            remain_size_start += nn_size * 4;
#endif // __SSE2__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i];
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

            float s = scale[i];
#if __SSE2__
            __m128 _s128 = (elempack == 4) ? _mm_loadu_ps(scale + i * 4) : _mm_set1_ps(s);
#if __AVX__
            __m256 _s256 = (elempack == 8) ? _mm256_loadu_ps(scale + i * 8) : combine4x2_ps(_s128, _s128);
#if __AVX512F__
            __m512 _s512 = (elempack == 16) ? _mm512_loadu_ps(scale + i * 16) : combine8x2_ps(_s256, _s256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            if (bias_term)
            {
                float b = bias[i];
#if __SSE2__
                __m128 _b128 = (elempack == 4) ? _mm_loadu_ps(bias + i * 4) : _mm_set1_ps(b);
#if __AVX__
                __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps(bias + i * 8) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
                __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps(bias + i * 16) : combine8x2_ps(_b256, _b256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < size; j += 16)
                {
                    __m512 _p512 = _mm512_loadu_ps(ptr);
                    _mm512_storeu_ps(ptr, _mm512_fmadd_ps(_p512, _s512, _b512));
                    ptr += 16;
                }
#endif // __AVX512F__
                for (; j + 7 < size; j += 8)
                {
                    __m256 _p256 = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_comp_fmadd_ps(_p256, _s256, _b256));
                    ptr += 8;
                }
#endif // __AVX__
                for (; j + 3 < size; j += 4)
                {
                    __m128 _p128 = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, _mm_comp_fmadd_ps(_p128, _s128, _b128));
                    ptr += 4;
                }
#endif // __SSE__
                for (; j < size; j++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; j + 15 < size; j += 16)
                {
                    __m512 _p512 = _mm512_loadu_ps(ptr);
                    _mm512_storeu_ps(ptr, _mm512_mul_ps(_p512, _s512));
                    ptr += 16;
                }
#endif // __AVX512F__
                for (; j + 7 < size; j += 8)
                {
                    __m256 _p256 = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_mul_ps(_p256, _s256));
                    ptr += 8;
                }
#endif // __AVX__
                for (; j + 3 < size; j += 4)
                {
                    __m128 _p128 = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, _mm_mul_ps(_p128, _s128));
                    ptr += 4;
                }
#endif // __SSE__
                for (; j < size; j++)
                {
                    *ptr = *ptr * s;
                    ptr++;
                }
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale[q];
#if __SSE2__
            __m128 _s128 = (elempack == 4) ? _mm_loadu_ps(scale + q * 4) : _mm_set1_ps(s);
#if __AVX__
            __m256 _s256 = (elempack == 8) ? _mm256_loadu_ps(scale + q * 8) : combine4x2_ps(_s128, _s128);
#if __AVX512F__
            __m512 _s512 = (elempack == 16) ? _mm512_loadu_ps(scale + q * 16) : combine8x2_ps(_s256, _s256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            if (bias_term)
            {
                float b = bias[q];
#if __SSE2__
                __m128 _b128 = (elempack == 4) ? _mm_loadu_ps(bias + q * 4) : _mm_set1_ps(b);
#if __AVX__
                __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps(bias + q * 8) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
                __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps(bias + q * 16) : combine8x2_ps(_b256, _b256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p512 = _mm512_loadu_ps(ptr);
                    _mm512_storeu_ps(ptr, _mm512_fmadd_ps(_p512, _s512, _b512));
                    ptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p256 = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_comp_fmadd_ps(_p256, _s256, _b256));
                    ptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p128 = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, _mm_comp_fmadd_ps(_p128, _s128, _b128));
                    ptr += 4;
                }
#endif // __SSE__
                for (; i < size; i++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p512 = _mm512_loadu_ps(ptr);
                    _mm512_storeu_ps(ptr, _mm512_mul_ps(_p512, _s512));
                    ptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p256 = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_mul_ps(_p256, _s256));
                    ptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p128 = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, _mm_mul_ps(_p128, _s128));
                    ptr += 4;
                }
#endif // __SSE__
                for (; i < size; i++)
                {
                    *ptr = *ptr * s;
                    ptr++;
                }
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int Scale_x86::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    // scale_blob may be bf16 (from second input) or fp32 (from scale_data weight)
    const float* scale = 0;
    Mat scale_fp32;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;
        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }
    const float* bias = bias_data;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        const int size = w * elempack;

        int i = 0;
        if (bias_term)
        {
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
                __m512 _s = _mm512_loadu_ps(scale + i);
                __m512 _bias = _mm512_loadu_ps(bias + i);
                _p = _mm512_fmadd_ps(_p, _s, _bias);
                _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
                __m256 _s = _mm256_loadu_ps(scale + i);
                __m256 _bias = _mm256_loadu_ps(bias + i);
                _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
                __m128 _s = _mm_loadu_ps(scale + i);
                __m128 _bias = _mm_loadu_ps(bias + i);
                _p = _mm_comp_fmadd_ps(_p, _s, _bias);
                _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
            }
        }
        else
        {
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
                __m512 _s = _mm512_loadu_ps(scale + i);
                _p = _mm512_mul_ps(_p, _s);
                _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
                __m256 _s = _mm256_loadu_ps(scale + i);
                _p = _mm256_mul_ps(_p, _s);
                _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
                __m128 _s = _mm_loadu_ps(scale + i);
                _p = _mm_mul_ps(_p, _s);
                _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
            }
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;

            if (bias_term)
            {
                const float* bptr = bias + i * elempack;
                scale_bf16s_sse(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_sse(ptr, sptr, size, elempack);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;

            if (bias_term)
            {
                const float* bptr = bias + q * elempack;
                scale_bf16s_sse(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_sse(ptr, sptr, size, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
