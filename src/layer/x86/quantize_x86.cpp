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

#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __SSE2__
static inline int64_t float2int8(__m128 _v0, __m128 _v1)
{
    float v0[4];
    float v1[4];
    _mm_storeu_ps(v0, _v0);
    _mm_storeu_ps(v1, _v1);

    int v0_i[4];
    int v1_i[4];
    v0_i[0] = round(v0[0]);
    v0_i[1] = round(v0[1]);
    v0_i[2] = round(v0[2]);
    v0_i[3] = round(v0[3]);
    v1_i[0] = round(v1[0]);
    v1_i[1] = round(v1[1]);
    v1_i[2] = round(v1[2]);
    v1_i[3] = round(v1[3]);

    __m128i _v0_i = _mm_loadu_si128((const __m128i*)v0_i);
    __m128i _v1_i = _mm_loadu_si128((const __m128i*)v1_i);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v01_s16);

    return _mm_cvtsi128_si64(_v8);
}

static inline __m128i float2int8(__m128 _v0, __m128 _v1, __m128 _v2, __m128 _v3)
{
    float v0[4];
    float v1[4];
    float v2[4];
    float v3[4];
    _mm_storeu_ps(v0, _v0);
    _mm_storeu_ps(v1, _v1);
    _mm_storeu_ps(v2, _v2);
    _mm_storeu_ps(v3, _v3);

    int v0_i[4];
    int v1_i[4];
    int v2_i[4];
    int v3_i[4];
    v0_i[0] = round(v0[0]);
    v0_i[1] = round(v0[1]);
    v0_i[2] = round(v0[2]);
    v0_i[3] = round(v0[3]);
    v1_i[0] = round(v1[0]);
    v1_i[1] = round(v1[1]);
    v1_i[2] = round(v1[2]);
    v1_i[3] = round(v1[3]);
    v2_i[0] = round(v2[0]);
    v2_i[1] = round(v2[1]);
    v2_i[2] = round(v2[2]);
    v2_i[3] = round(v2[3]);
    v3_i[0] = round(v3[0]);
    v3_i[1] = round(v3[1]);
    v3_i[2] = round(v3[2]);
    v3_i[3] = round(v3[3]);

    __m128i _v0_i = _mm_loadu_si128((const __m128i*)v0_i);
    __m128i _v1_i = _mm_loadu_si128((const __m128i*)v1_i);
    __m128i _v2_i = _mm_loadu_si128((const __m128i*)v2_i);
    __m128i _v3_i = _mm_loadu_si128((const __m128i*)v3_i);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);
    __m128i _v23_s16 = _mm_packs_epi32(_v2_i, _v3_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v23_s16 = _mm_min_epi16(_v23_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));
    _v23_s16 = _mm_max_epi16(_v23_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v23_s16);

    return _v8;
}
#if __AVX__
static inline int64_t float2int8(__m256 _v0)
{
    __m256i _v0_i = _mm256_cvtps_epi32(_mm256_round_ps(_v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    __m256i _v01_s16 = _mm256_packs_epi32(_v0_i, _v0_i);
    _v01_s16 = _mm256_permute4x64_epi64(_v01_s16, 0xd8);

    __m128i _v01_s16low = _mm256_extractf128_si256(_v01_s16, 0);

    _v01_s16low = _mm_min_epi16(_v01_s16low, _mm_set1_epi16(127));
    _v01_s16low = _mm_max_epi16(_v01_s16low, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16low, _v01_s16low);

    return _mm_cvtsi128_si64(_v8);
}

static inline __m128i float2int8(__m256 _v0, __m256 _v1)
{
    __m256i _v0_i = _mm256_cvtps_epi32(_mm256_round_ps(_v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i _v1_i = _mm256_cvtps_epi32(_mm256_round_ps(_v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    __m256i _v01_s16 = _mm256_packs_epi32(_v0_i, _v1_i);
    _v01_s16 = _mm256_permute4x64_epi64(_v01_s16, 0xd8);

    _v01_s16 = _mm256_min_epi16(_v01_s16, _mm256_set1_epi16(127));
    _v01_s16 = _mm256_max_epi16(_v01_s16, _mm256_set1_epi16(-127));

    __m256i _v8 = _mm256_packs_epi16(_v01_s16, _v01_s16);
    _v8 = _mm256_permute4x64_epi64(_v8, 0xd8);

    return _mm256_extractf128_si256(_v8, 0);
}
#endif // __AVX__
#endif // __SSE2__

Quantize_x86::Quantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Quantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m256 _scale = _mm256_set1_ps(scale_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr = (const float*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    __m256 _v = _mm256_loadu_ps(ptr);
                    _v = _mm256_mul_ps(_v, _scale);
                    *(int64_t*)outptr = float2int8(_v);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr = (const float*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    __m256 _v = _mm256_loadu_ps(ptr);
                    __m256 _scale = _mm256_loadu_ps((const float*)scale_data + i * 8);
                    _v = _mm256_mul_ps(_v, _scale);
                    *(int64_t*)outptr = float2int8(_v);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m256 _scale = _mm256_set1_ps(scale_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.row(i);
                    signed char* outptr = top_blob.row<signed char>(i);

                    int j = 0;
                    for (; j + 1 < w; j += 2)
                    {
                        __m256 _v0 = _mm256_loadu_ps(ptr);
                        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
                        _v0 = _mm256_mul_ps(_v0, _scale);
                        _v1 = _mm256_mul_ps(_v1, _scale);
                        __m128i _v = float2int8(_v0, _v1);
                        _mm_storeu_si128((__m128i*)outptr, _v);

                        ptr += 16;
                        outptr += 16;
                    }
                    for (; j < w; j++)
                    {
                        __m256 _v = _mm256_loadu_ps(ptr);
                        _v = _mm256_mul_ps(_v, _scale);
                        *(int64_t*)outptr = float2int8(_v);

                        ptr += 8;
                        outptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.row(i);
                    signed char* outptr = top_blob.row<signed char>(i);

                    __m256 _scale = _mm256_loadu_ps((const float*)scale_data + i * 8);

                    int j = 0;
                    for (; j + 1 < w; j += 2)
                    {
                        __m256 _v0 = _mm256_loadu_ps(ptr);
                        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
                        _v0 = _mm256_mul_ps(_v0, _scale);
                        _v1 = _mm256_mul_ps(_v1, _scale);
                        __m128i _v = float2int8(_v0, _v1);
                        _mm_storeu_si128((__m128i*)outptr, _v);

                        ptr += 16;
                        outptr += 16;
                    }
                    for (; j < w; j++)
                    {
                        __m256 _v = _mm256_loadu_ps(ptr);
                        _v = _mm256_mul_ps(_v, _scale);
                        *(int64_t*)outptr = float2int8(_v);

                        ptr += 8;
                        outptr += 8;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m256 _scale = _mm256_set1_ps(scale_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    signed char* outptr = top_blob.channel(q);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __m256 _v0 = _mm256_loadu_ps(ptr);
                        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
                        _v0 = _mm256_mul_ps(_v0, _scale);
                        _v1 = _mm256_mul_ps(_v1, _scale);
                        __m128i _v = float2int8(_v0, _v1);
                        _mm_storeu_si128((__m128i*)outptr, _v);

                        ptr += 16;
                        outptr += 16;
                    }
                    for (; i < size; i++)
                    {
                        __m256 _v = _mm256_loadu_ps(ptr);
                        _v = _mm256_mul_ps(_v, _scale);
                        *(int64_t*)outptr = float2int8(_v);

                        ptr += 8;
                        outptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    signed char* outptr = top_blob.channel(q);

                    __m256 _scale = _mm256_loadu_ps((const float*)scale_data + q * 8);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        __m256 _v0 = _mm256_loadu_ps(ptr);
                        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
                        _v0 = _mm256_mul_ps(_v0, _scale);
                        _v1 = _mm256_mul_ps(_v1, _scale);
                        __m128i _v = float2int8(_v0, _v1);
                        _mm_storeu_si128((__m128i*)outptr, _v);

                        ptr += 16;
                        outptr += 16;
                    }
                    for (; i < size; i++)
                    {
                        __m256 _v = _mm256_loadu_ps(ptr);
                        _v = _mm256_mul_ps(_v, _scale);
                        *(int64_t*)outptr = float2int8(_v);

                        ptr += 8;
                        outptr += 8;
                    }
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    __m128 _scale = _mm_set1_ps(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        int j = 0;
                        for (; j + 1 < w; j += 2)
                        {
                            __m128 _v0 = _mm_loadu_ps(ptr0);
                            __m128 _v1 = _mm_loadu_ps(ptr1);
                            __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                            __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                            _v0 = _mm_mul_ps(_v0, _scale);
                            _v1 = _mm_mul_ps(_v1, _scale);
                            _v2 = _mm_mul_ps(_v2, _scale);
                            _v3 = _mm_mul_ps(_v3, _scale);
                            __m128i _v = float2int8(_v0, _v1, _v2, _v3);
                            _mm_storeu_si128((__m128i*)outptr, _v);

                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; j < w; j++)
                        {
                            __m128 _vlow = _mm_loadu_ps(ptr0);
                            __m128 _vhigh = _mm_loadu_ps(ptr1);
                            _vlow = _mm_mul_ps(_vlow, _scale);
                            _vhigh = _mm_mul_ps(_vhigh, _scale);
                            *(int64_t*)outptr = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        __m128 _scale0 = _mm_loadu_ps((const float*)scale_data + i * 8);
                        __m128 _scale1 = _mm_loadu_ps((const float*)scale_data + i * 8 + 4);

                        int j = 0;
                        for (; j + 1 < w; j += 2)
                        {
                            __m128 _v0 = _mm_loadu_ps(ptr0);
                            __m128 _v1 = _mm_loadu_ps(ptr1);
                            __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                            __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                            _v0 = _mm_mul_ps(_v0, _scale0);
                            _v1 = _mm_mul_ps(_v1, _scale1);
                            _v2 = _mm_mul_ps(_v2, _scale0);
                            _v3 = _mm_mul_ps(_v3, _scale1);
                            __m128i _v = float2int8(_v0, _v1, _v2, _v3);
                            _mm_storeu_si128((__m128i*)outptr, _v);

                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; j < w; j++)
                        {
                            __m128 _vlow = _mm_loadu_ps(ptr0);
                            __m128 _vhigh = _mm_loadu_ps(ptr1);
                            _vlow = _mm_mul_ps(_vlow, _scale0);
                            _vhigh = _mm_mul_ps(_vhigh, _scale1);
                            *(int64_t*)outptr = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        const float s0 = scale_data[i * 4];
                        const float s1 = scale_data[i * 4 + 1];
                        const float s2 = scale_data[i * 4 + 2];
                        const float s3 = scale_data[i * 4 + 3];

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    __m128 _scale = _mm_set1_ps(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        int i = 0;
                        for (; i + 1 < size; i += 2)
                        {
                            __m128 _v0 = _mm_loadu_ps(ptr0);
                            __m128 _v1 = _mm_loadu_ps(ptr1);
                            __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                            __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                            _v0 = _mm_mul_ps(_v0, _scale);
                            _v1 = _mm_mul_ps(_v1, _scale);
                            _v2 = _mm_mul_ps(_v2, _scale);
                            _v3 = _mm_mul_ps(_v3, _scale);
                            __m128i _v = float2int8(_v0, _v1, _v2, _v3);
                            _mm_storeu_si128((__m128i*)outptr, _v);

                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; i < size; i++)
                        {
                            __m128 _vlow = _mm_loadu_ps(ptr0);
                            __m128 _vhigh = _mm_loadu_ps(ptr1);
                            _vlow = _mm_mul_ps(_vlow, _scale);
                            _vhigh = _mm_mul_ps(_vhigh, _scale);
                            *(int64_t*)outptr = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        __m128 _scale0 = _mm_loadu_ps((const float*)scale_data + q * 8);
                        __m128 _scale1 = _mm_loadu_ps((const float*)scale_data + q * 8 + 4);

                        int i = 0;
                        for (; i + 1 < size; i += 2)
                        {
                            __m128 _v0 = _mm_loadu_ps(ptr0);
                            __m128 _v1 = _mm_loadu_ps(ptr1);
                            __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                            __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                            _v0 = _mm_mul_ps(_v0, _scale0);
                            _v1 = _mm_mul_ps(_v1, _scale1);
                            _v2 = _mm_mul_ps(_v2, _scale0);
                            _v3 = _mm_mul_ps(_v3, _scale1);
                            __m128i _v = float2int8(_v0, _v1, _v2, _v3);
                            _mm_storeu_si128((__m128i*)outptr, _v);

                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; i < size; i++)
                        {
                            __m128 _vlow = _mm_loadu_ps(ptr0);
                            __m128 _vhigh = _mm_loadu_ps(ptr1);
                            _vlow = _mm_mul_ps(_vlow, _scale0);
                            _vhigh = _mm_mul_ps(_vhigh, _scale1);
                            *(int64_t*)outptr = float2int8(_vlow, _vhigh);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        const float s0 = scale_data[q * 4];
                        const float s1 = scale_data[q * 4 + 1];
                        const float s2 = scale_data[q * 4 + 2];
                        const float s3 = scale_data[q * 4 + 3];

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr0 = bottom_blob.row(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8(*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}

} // namespace ncnn
