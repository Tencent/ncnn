// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cast_x86.h"

#if __SSE2__
#include <emmintrin.h>
#endif // __SSE2__
#if __AVX__
#include <immintrin.h>
#endif // __AVX__

#if __AVX__
#include <stdint.h>
typedef union m128i
{
    __m128i vec;
    uint16_t m128i_u16[8];
} m128;

typedef union m256i
{
    __m256i vec;
    uint32_t m256i_u32[8];
} m256;
static inline __m256 bfloat2float_avx(__m128i v0)
{
    __m128i zero = _mm_set1_epi32(0);
    __m128i a = _mm_slli_epi32(_mm_unpacklo_epi16(v0, zero), 16);
    __m128i b = _mm_slli_epi32(_mm_unpackhi_epi16(v0, zero), 16);
    __m256i ab = _mm256_set1_epi32(0);
    ab = _mm256_insertf128_si256(ab, a, 0); // insert in low 128-bit lane
    ab = _mm256_insertf128_si256(ab, b, 1); // insert in high 128-bit lane
    return _mm256_castsi256_ps(ab);
}
static inline __m256i float2bfloat_avx(__m256 v0, __m256 v1)
{
    __m256i a = _mm256_castps_si256(v0);
    a = _mm256_srli_epi32(a, 16);
    __m256i b = _mm256_castps_si256(v1);
    b = _mm256_srli_epi32(b, 16);
    __m256i abab = _mm256_packus_epi32(a, b);
    return _mm256_permutevar8x32_epi32(abab, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}
static inline __m128i float2bfloat_avx(__m256 v0)
{
    __m256i a = _mm256_castps_si256(v0);
    a = _mm256_srli_epi32(a, 16);
    __m256i aaaa = _mm256_packus_epi32(a, a);
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(aaaa, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7)));
}
#endif // __AVX__

namespace ncnn {

Cast_x86::Cast_x86()
{
    support_packing = true;
}

int Cast_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __AVX__
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        if (type_from == 3)
        {
            Cast::forward(bottom_blob, top_blob, opt);
        }

        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int size = w * h * elempack;

    if (type_from == 1 && type_to == 2)
    {
        int nn = size >> 3;
        int remain = size - (nn << 3);
        m256i mask = {_mm256_setzero_si256()};
        for (int i = 0; i < remain; i++)
            mask.m256i_u32[i] = 0x80000000;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < nn; i++)
            {
                __m256 fp32 = _mm256_loadu_ps(ptr);
                __m128i fp16 = _mm256_cvtps_ph(fp32, _MM_FROUND_TRUNC);
                _mm_store_si128((__m128i*)outptr, fp16);
                ptr += 8;
                outptr += 8;
            }

            if (remain > 0)
            {
                __m256 fp32 = _mm256_maskload_ps(ptr, mask.vec);
                m128i fp16 = {_mm256_cvtps_ph(fp32, _MM_FROUND_TRUNC)};
                memcpy(outptr, fp16.m128i_u16, remain * sizeof(unsigned short));
            }
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        int nn = size >> 3;
        int remain = size - (nn << 3);
        m256i mask = {_mm256_setzero_si256()};
        for (int i = 0; i < remain; i++)
            mask.m256i_u32[i] = 0x80000000;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < nn; i++)
            {
                __m128i fp16 = _mm_lddqu_si128((__m128i const*)ptr);
                __m256 fp32 = _mm256_cvtph_ps(fp16);
                _mm256_storeu_ps(outptr, fp32);
                ptr += 8;
                outptr += 8;
            }

            if (remain > 0)
            {
                m128i fp16 = {_mm_setzero_si128()};
                memcpy(fp16.m128i_u16, ptr, remain * sizeof(unsigned short));
                __m256 fp32 = _mm256_cvtph_ps(fp16.vec);
                _mm256_maskstore_ps(outptr, mask.vec, fp32);
            }
        }
    }
    if (type_from == 4 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int nn = size >> 3;
            int remain = size & 7;
            for (; nn > 0; nn--)
            {
                _mm256_storeu_ps(outptr, bfloat2float_avx(_mm_lddqu_si128((__m128i const*)ptr)));
                ptr += 8;
                outptr += 8;
            }

            for (; remain > 0; remain--)
            {
                *outptr = bfloat16_to_float32(*ptr);
                outptr++;
                ptr++;
            }
        }
    }
    if (type_from == 1 && type_to == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);
            int nn = size >> 4;
            int remain = size & 15;
            for (; nn > 0; nn--)
            {
                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr + 8)));
                ptr += 16;
                outptr += 16;
            }
            if (remain >= 8)
            {
                remain -= 8;
                _mm_store_si128((__m128i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr)));
                ptr += 8;
                outptr += 8;
            }
            for (; remain > 0; remain--)
            {
                *outptr = float32_to_bfloat16(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    return 0;
#else // __AVX__

    return Cast::forward(bottom_blob, top_blob, opt);

#endif // __AVX__
}

} // namespace ncnn
