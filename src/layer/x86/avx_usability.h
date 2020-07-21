// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
#ifndef AVX_USABILITY
#define AVX_USABILITY
#include <immintrin.h>

static inline __m256 loadfp16(const unsigned short* ptr)
{
    return _mm256_cvtph_ps(_mm_lddqu_si128((__m128i*)(ptr)));
}
static inline __m256 _mm256_fmadd_1_ps(__m256 a, __m256 b, float c)
{
    return _mm256_fmadd_ps(b, _mm256_set1_ps(c), a);
}

static inline __m256 _mm256_fmrsub_1_ps(__m256 a, __m256 b, float c)
{
    return _mm256_sub_ps(a, _mm256_mul_ps(b, _mm256_set1_ps(c)));
}
// From: https://stackoverflow.com/a/25627536
static inline void transpose8_ps(__m256& row0, __m256& row1, __m256& row2, __m256& row3, __m256& row4, __m256& row5, __m256& row6, __m256& row7)
{
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
    __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
    __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
    __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
    __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
    __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
    __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
    __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

static inline __m256 HorizontalSums(__m256 v0, __m256 v1, __m256 v2, __m256 v3, __m256 v4,
                                    __m256 v5, __m256 v6, __m256 v7)
{
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s45 = _mm256_hadd_ps(v4, v5);
    const __m256 s67 = _mm256_hadd_ps(v6, v7);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);
    const __m256 s4556 = _mm256_hadd_ps(s45, s67);

    // inter-lane shuffle
    v0 = _mm256_blend_ps(s0123, s4556, 0xF0);
    v1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);

    return _mm256_add_ps(v0, v1);
}

static inline __m128 HorizontalSums(__m256 v0, __m256 v1, __m256 v2, __m256 v3)
{
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);

    return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                      _mm256_castps256_ps128(s0123));
}

static inline __m128 HorizontalSums(__m256 v0, __m256 v1, __m256 v2)
{
    const __m256 v3 = _mm256_set1_ps(0.0f);
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);

    return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                      _mm256_castps256_ps128(s0123));
}

static inline float _mm256_reduce_add_ps(__m256 x)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

static inline float _mm_reduce_add_ps(__m128 x128)
{
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}
#endif