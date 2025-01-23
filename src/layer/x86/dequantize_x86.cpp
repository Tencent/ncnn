// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "dequantize_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Dequantize_x86::Dequantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __SSE2__
    __m128 _scale = _mm_set1_ps(scale);
#if __AVX__
    __m256 _scale_avx = _mm256_set1_ps(scale);
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
#endif // __AVX512F__
#endif // __AVX__
    if (scale_data_size > 1)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_avx512 = _mm512_loadu_ps((const float*)scale_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            _scale_avx = _mm256_loadu_ps((const float*)scale_data);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            _scale = _mm_loadu_ps((const float*)scale_data);
#if __AVX__
            _scale_avx = combine4x2_ps(_scale, _scale);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
#endif // __AVX__
        }
    }
#endif // __SSE2__

    if (bias_data_size == 0)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_mul_ps(_v, _scale_avx512);
            _mm512_storeu_ps(ptr, _v);
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_mul_ps(_v, _scale_avx);
            _mm256_storeu_ps(ptr, _v);
            intptr += 8;
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_mul_ps(_v, _scale);
            _mm_storeu_ps(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __SSE2__
        __m128 _bias = _mm_set1_ps(bias);
#if __AVX__
        __m256 _bias_avx = _mm256_set1_ps(bias);
#if __AVX512F__
        __m512 _bias_avx512 = _mm512_set1_ps(bias);
#endif // __AVX512F__
#endif // __AVX__
        if (bias_data_size > 1)
        {
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                _bias_avx512 = _mm512_loadu_ps((const float*)bias_data);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                _bias_avx = _mm256_loadu_ps((const float*)bias_data);
#if __AVX512F__
                _bias_avx512 = combine8x2_ps(_bias_avx, _bias_avx);
#endif // __AVX512F__
            }
#endif // __AVX__
            if (elempack == 4)
            {
                _bias = _mm_loadu_ps((const float*)bias_data);
#if __AVX__
                _bias_avx = combine4x2_ps(_bias, _bias);
#if __AVX512F__
                _bias_avx512 = combine8x2_ps(_bias_avx, _bias_avx);
#endif // __AVX512F__
#endif // __AVX__
            }
        }
#endif // __SSE2__

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_fmadd_ps(_v, _scale_avx512, _bias_avx512);
            _mm512_storeu_ps(ptr, _v);
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_comp_fmadd_ps(_v, _scale_avx, _bias_avx);
            _mm256_storeu_ps(ptr, _v);
            intptr += 8;
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_comp_fmadd_ps(_v, _scale, _bias);
            _mm_storeu_ps(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
}

int Dequantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            float* ptr = (float*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
