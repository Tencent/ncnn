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

#include "quantize_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Quantize_x86::Quantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

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

    int i = 0;
#if __SSE2__
#if __AVX__
    for (; i + 15 < size; i += 16)
    {
#if __AVX512F__
        __m512 _v = _mm512_loadu_ps(ptr);
        _v = _mm512_mul_ps(_v, _scale_avx512);
        _mm_storeu_si128((__m128i*)s8ptr, float2int8_avx512(_v));
#else  // __AVX512F__
        __m256 _v0 = _mm256_loadu_ps(ptr);
        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
        _v0 = _mm256_mul_ps(_v0, _scale_avx);
        _v1 = _mm256_mul_ps(_v1, _scale_avx);
        _mm_storeu_si128((__m128i*)s8ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
        ptr += 16;
        s8ptr += 16;
    }
#endif // __AVX__
    for (; i + 7 < size; i += 8)
    {
#if __AVX__
        __m256 _v = _mm256_loadu_ps(ptr);
        _v = _mm256_mul_ps(_v, _scale_avx);
        *(int64_t*)s8ptr = float2int8_avx(_v);
#else  // __AVX__
        __m128 _v0 = _mm_loadu_ps(ptr);
        __m128 _v1 = _mm_loadu_ps(ptr + 4);
        _v0 = _mm_mul_ps(_v0, _scale);
        _v1 = _mm_mul_ps(_v1, _scale);
        *(int64_t*)s8ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _v = _mm_loadu_ps(ptr);
        _v = _mm_mul_ps(_v, _scale);
        int32_t v = float2int8_sse(_v);
        s8ptr[0] = (v >> 0) & 0xff;
        s8ptr[1] = (v >> 8) & 0xff;
        s8ptr[2] = (v >> 16) & 0xff;
        s8ptr[3] = (v >> 24) & 0xff;
        ptr += 4;
        s8ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 1u;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_blob.row(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

            quantize(ptr, s8ptr, scale_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

            quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
