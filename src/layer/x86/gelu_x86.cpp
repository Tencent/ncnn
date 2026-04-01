// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "gelu_bf16s.h"
#endif

GELU_x86::GELU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int GELU_x86::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int GELU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (fast_gelu)
        {
            __m512 _half512 = _mm512_set1_ps(0.5f);
            __m512 _one512 = _mm512_set1_ps(1.f);
            __m512 _fast1c512 = _mm512_set1_ps(0.79788452f);
            __m512 _fast2c512 = _mm512_set1_ps(0.044715f);
            for (; i + 15 < size; i += 16)
            {
                __m512 _pLoad = _mm512_loadu_ps(ptr);

                __m512 _cube = _mm512_mul_ps(_pLoad, _pLoad);
                _cube = _mm512_mul_ps(_pLoad, _cube);

                __m512 _blob = _mm512_mul_ps(_fast2c512, _cube);
                _blob = _mm512_add_ps(_pLoad, _blob);
                _blob = _mm512_mul_ps(_fast1c512, _blob);
                _blob = tanh512_ps(_blob);
                _blob = _mm512_add_ps(_one512, _blob);

                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm512_storeu_ps(ptr, _blob);

                ptr += 16;
            }
        }
        else
        {
            __m512 _half512 = _mm512_set1_ps(0.5f);
            __m512 _one512 = _mm512_set1_ps(1.f);
            __m512 _inv_sqrt2_512 = _mm512_set1_ps(0.70710678f);
            for (; i + 15 < size; i += 16)
            {
                __m512 _pLoad = _mm512_loadu_ps(ptr);

                __m512 _erf = erf512_ps(_mm512_mul_ps(_pLoad, _inv_sqrt2_512));
                __m512 _blob = _mm512_add_ps(_one512, _erf);
                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm512_storeu_ps(ptr, _blob);

                ptr += 16;
            }
        }
#endif // __AVX512F__
        if (fast_gelu)
        {
            __m256 _half256 = _mm256_set1_ps(0.5f);
            __m256 _one256 = _mm256_set1_ps(1.f);
            __m256 _fast1c256 = _mm256_set1_ps(0.79788452f);
            __m256 _fast2c256 = _mm256_set1_ps(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _pLoad = _mm256_loadu_ps(ptr);

                __m256 _cube = _mm256_mul_ps(_pLoad, _pLoad);
                _cube = _mm256_mul_ps(_pLoad, _cube);

                __m256 _blob = _mm256_mul_ps(_fast2c256, _cube);
                _blob = _mm256_add_ps(_pLoad, _blob);
                _blob = _mm256_mul_ps(_fast1c256, _blob);
                _blob = tanh256_ps(_blob);
                _blob = _mm256_add_ps(_one256, _blob);

                _blob = _mm256_mul_ps(_half256, _mm256_mul_ps(_blob, _pLoad));

                _mm256_storeu_ps(ptr, _blob);

                ptr += 8;
            }
        }
        else
        {
            __m256 _half256 = _mm256_set1_ps(0.5f);
            __m256 _one256 = _mm256_set1_ps(1.f);
            __m256 _inv_sqrt2_256 = _mm256_set1_ps(0.70710678f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _pLoad = _mm256_loadu_ps(ptr);

                __m256 _erf = erf256_ps(_mm256_mul_ps(_pLoad, _inv_sqrt2_256));
                __m256 _blob = _mm256_add_ps(_one256, _erf);
                _blob = _mm256_mul_ps(_half256, _mm256_mul_ps(_blob, _pLoad));

                _mm256_storeu_ps(ptr, _blob);

                ptr += 8;
            }
        }
#endif // __AVX__
        if (fast_gelu)
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _fast1c128 = _mm_set1_ps(0.79788452f);
            __m128 _fast2c128 = _mm_set1_ps(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = _mm_loadu_ps(ptr);

                __m128 _cube = _mm_mul_ps(_pLoad, _pLoad);
                _cube = _mm_mul_ps(_pLoad, _cube);

                __m128 _blob = _mm_mul_ps(_fast2c128, _cube);
                _blob = _mm_add_ps(_pLoad, _blob);
                _blob = _mm_mul_ps(_fast1c128, _blob);
                _blob = tanh_ps(_blob);
                _blob = _mm_add_ps(_one128, _blob);

                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storeu_ps(ptr, _blob);

                ptr += 4;
            }
        }
        else
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _inv_sqrt2_128 = _mm_set1_ps(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = _mm_loadu_ps(ptr);

                __m128 _erf = erf_ps(_mm_mul_ps(_pLoad, _inv_sqrt2_128));
                __m128 _blob = _mm_add_ps(_one128, _erf);
                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storeu_ps(ptr, _blob);

                ptr += 4;
            }
        }
#endif // __SSE2__
        if (fast_gelu)
        {
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));

                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * (1.0f + erff(0.70710678f * *ptr));

                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int GELU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    gelu_bf16s(bottom_top_blob, fast_gelu, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
