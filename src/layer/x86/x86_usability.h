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

#ifndef X86_USABILITY_H
#define X86_USABILITY_H

#include <math.h>
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#if __XOP__
#ifdef _MSC_VER
#include <ammintrin.h>
#else
#include <x86intrin.h>
#endif
#endif
#endif
#endif // __SSE2__

static NCNN_FORCEINLINE signed char float2int8(float v)
{
    int int32 = (int)round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __SSE2__
static NCNN_FORCEINLINE float _mm_reduce_add_ps(__m128 x128)
{
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static NCNN_FORCEINLINE float _mm_reduce_max_ps(__m128 x128)
{
    const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static NCNN_FORCEINLINE int _mm_reduce_add_epi32(__m128i x)
{
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static NCNN_FORCEINLINE int32_t float2int8_sse(const __m128& _v0)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);

    __m128i _v0_s16 = _mm_packs_epi32(_v0_i, _v0_i);

    _v0_s16 = _mm_min_epi16(_v0_s16, _mm_set1_epi16(127));
    _v0_s16 = _mm_max_epi16(_v0_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v0_s16, _v0_s16);

#if defined(__x86_64__) || defined(_M_X64)
    return (int32_t)_mm_cvtsi128_si64(_v8);
#else
    return _mm_cvtsi128_si32(_v8);
#endif
}

static NCNN_FORCEINLINE int64_t float2int8_sse(const __m128& _v0, const __m128& _v1)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _sign1 = _mm_and_ps(_v1, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v1_p5 = _mm_or_ps(_p5, _sign1);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128 _v1_adj = _mm_add_ps(_v1, _v1_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);
    __m128i _v1_i = _mm_cvttps_epi32(_v1_adj);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v01_s16);

#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(_v8);
#else
    int64_t v8[2];
    _mm_storeu_si128((__m128i*)v8, _v8);
    return v8[0];
#endif
}

static NCNN_FORCEINLINE __m128i float2int8_sse(const __m128& _v0, const __m128& _v1, const __m128& _v2, const __m128& _v3)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _sign1 = _mm_and_ps(_v1, _signmask);
    __m128 _sign2 = _mm_and_ps(_v2, _signmask);
    __m128 _sign3 = _mm_and_ps(_v3, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v1_p5 = _mm_or_ps(_p5, _sign1);
    __m128 _v2_p5 = _mm_or_ps(_p5, _sign2);
    __m128 _v3_p5 = _mm_or_ps(_p5, _sign3);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128 _v1_adj = _mm_add_ps(_v1, _v1_p5);
    __m128 _v2_adj = _mm_add_ps(_v2, _v2_p5);
    __m128 _v3_adj = _mm_add_ps(_v3, _v3_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);
    __m128i _v1_i = _mm_cvttps_epi32(_v1_adj);
    __m128i _v2_i = _mm_cvttps_epi32(_v2_adj);
    __m128i _v3_i = _mm_cvttps_epi32(_v3_adj);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);
    __m128i _v23_s16 = _mm_packs_epi32(_v2_i, _v3_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v23_s16 = _mm_min_epi16(_v23_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));
    _v23_s16 = _mm_max_epi16(_v23_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v23_s16);

    return _v8;
}

#ifndef __FMA__
static NCNN_FORCEINLINE __m128 _mm_comp_fmadd_ps(__m128 _a, const __m128 _b, const __m128 _c)
{
    return _mm_add_ps(_mm_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmadd_ps(__m128 _a, const __m128 _b, const __m128 _c)
{
    return _mm_sub_ps(_c, _mm_mul_ps(_a, _b));
}
#else
static NCNN_FORCEINLINE __m128 _mm_comp_fmadd_ps(__m128 _a, const __m128 _b, const __m128 _c)
{
    return _mm_fmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmadd_ps(__m128 _a, const __m128 _b, const __m128 _c)
{
    // return -a * b + c
    return _mm_fnmadd_ps(_a, _b, _c);
}
#endif // !__FMA__

#if __AVX__
#ifndef __FMA__
static NCNN_FORCEINLINE __m256 _mm256_comp_fmadd_ps(__m256 _a, const __m256 _b, const __m256 _c)
{
    return _mm256_add_ps(_mm256_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmadd_ps(__m256 _a, const __m256 _b, const __m256 _c)
{
    return _mm256_sub_ps(_c, _mm256_mul_ps(_a, _b));
}
#else
static NCNN_FORCEINLINE __m256 _mm256_comp_fmadd_ps(__m256 _a, const __m256 _b, const __m256 _c)
{
    return _mm256_fmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmadd_ps(__m256 _a, const __m256 _b, const __m256 _c)
{
    // return -a * b + c
    return _mm256_fnmadd_ps(_a, _b, _c);
}
#endif

#if __AVX2__
static NCNN_FORCEINLINE __m256 loadfp16(const unsigned short* ptr)
{
    return _mm256_cvtph_ps(_mm_lddqu_si128((__m128i*)(ptr)));
}
#endif

static NCNN_FORCEINLINE __m256 _mm256_fmadd_1_ps(__m256 a, __m256 b, float c)
{
    return _mm256_comp_fmadd_ps(b, _mm256_set1_ps(c), a);
}

static NCNN_FORCEINLINE __m256 _mm256_fmrsub_1_ps(__m256 a, __m256 b, float c)
{
    // return a - b * c
    return _mm256_comp_fnmadd_ps(b, _mm256_set1_ps(c), a);
}
// From: https://stackoverflow.com/a/25627536
static NCNN_FORCEINLINE void transpose8_ps(__m256& row0, __m256& row1, __m256& row2, __m256& row3, __m256& row4, __m256& row5, __m256& row6, __m256& row7)
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

static NCNN_FORCEINLINE __m256 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3, __m256& v4, __m256& v5, __m256& v6, __m256& v7)
{
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s45 = _mm256_hadd_ps(v4, v5);
    const __m256 s67 = _mm256_hadd_ps(v6, v7);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);
    const __m256 s4556 = _mm256_hadd_ps(s45, s67);

    // inter-lane shuffle
    const __m256 vb0 = _mm256_blend_ps(s0123, s4556, 0xF0);
    const __m256 vb1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);

    return _mm256_add_ps(vb0, vb1);
}

static NCNN_FORCEINLINE __m128 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3)
{
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);

    return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                      _mm256_castps256_ps128(s0123));
}

static NCNN_FORCEINLINE __m128 HorizontalSums(__m256& v0, __m256& v1, __m256& v2)
{
    const __m256 v3 = _mm256_set1_ps(0.0f);
    const __m256 s01 = _mm256_hadd_ps(v0, v1);
    const __m256 s23 = _mm256_hadd_ps(v2, v3);
    const __m256 s0123 = _mm256_hadd_ps(s01, s23);

    return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                      _mm256_castps256_ps128(s0123));
}

static NCNN_FORCEINLINE float _mm256_reduce_add_ps(__m256 x)
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

static NCNN_FORCEINLINE float _mm256_reduce_max_ps(__m256 x)
{
    const __m128 x128 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static NCNN_FORCEINLINE int64_t float2int8_avx(const __m256& _v0)
{
    // _MM_FROUND_TO_NEAREST_INT round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m256 _p5 = _mm256_set1_ps(0.5f);
    __m256 _signmask = _mm256_castsi256_ps(_mm256_set1_epi32(1 << 31));
    __m256 _sign = _mm256_and_ps(_v0, _signmask);
    __m256 _v0_p5 = _mm256_or_ps(_p5, _sign);
    __m256 _v0_adj = _mm256_add_ps(_v0, _v0_p5);
    __m256i _v0_i = _mm256_cvttps_epi32(_v0_adj);

#if __AVX2__
    __m256i _v01_s16 = _mm256_packs_epi32(_v0_i, _v0_i);
    _v01_s16 = _mm256_permute4x64_epi64(_v01_s16, 0xd8);

    __m128i _v01_s16low = _mm256_extractf128_si256(_v01_s16, 0);
#else  // __AVX2__
    __m128i _v0_i_low = _mm256_extractf128_si256(_v0_i, 0);
    __m128i _v0_i_high = _mm256_extractf128_si256(_v0_i, 1);

    __m128i _v01_s16low = _mm_packs_epi32(_v0_i_low, _v0_i_high);
#endif // __AVX2__

    _v01_s16low = _mm_min_epi16(_v01_s16low, _mm_set1_epi16(127));
    _v01_s16low = _mm_max_epi16(_v01_s16low, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16low, _v01_s16low);

#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(_v8);
#else
    int64_t v8[2];
    _mm_storeu_si128((__m128i*)v8, _v8);
    return v8[0];
#endif
}

static NCNN_FORCEINLINE __m128i float2int8_avx(const __m256& _v0, const __m256& _v1)
{
    // _MM_FROUND_TO_NEAREST_INT round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m256 _p5 = _mm256_set1_ps(0.5f);
    __m256 _signmask = _mm256_castsi256_ps(_mm256_set1_epi32(1 << 31));
    __m256 _sign0 = _mm256_and_ps(_v0, _signmask);
    __m256 _sign1 = _mm256_and_ps(_v1, _signmask);
    __m256 _v0_p5 = _mm256_or_ps(_p5, _sign0);
    __m256 _v1_p5 = _mm256_or_ps(_p5, _sign1);
    __m256 _v0_adj = _mm256_add_ps(_v0, _v0_p5);
    __m256 _v1_adj = _mm256_add_ps(_v1, _v1_p5);
    __m256i _v0_i = _mm256_cvttps_epi32(_v0_adj);
    __m256i _v1_i = _mm256_cvttps_epi32(_v1_adj);

#if __AVX2__
    __m256i _v01_s16 = _mm256_packs_epi32(_v0_i, _v1_i);
    _v01_s16 = _mm256_permute4x64_epi64(_v01_s16, 0xd8);

    _v01_s16 = _mm256_min_epi16(_v01_s16, _mm256_set1_epi16(127));
    _v01_s16 = _mm256_max_epi16(_v01_s16, _mm256_set1_epi16(-127));

    __m256i _v8 = _mm256_packs_epi16(_v01_s16, _v01_s16);
    _v8 = _mm256_permute4x64_epi64(_v8, 0xd8);

    return _mm256_extractf128_si256(_v8, 0);
#else  // __AVX2__
    __m128i _v0_i_low = _mm256_extractf128_si256(_v0_i, 0);
    __m128i _v0_i_high = _mm256_extractf128_si256(_v0_i, 1);
    __m128i _v1_i_low = _mm256_extractf128_si256(_v1_i, 0);
    __m128i _v1_i_high = _mm256_extractf128_si256(_v1_i, 1);

    __m128i _v01_s16low = _mm_packs_epi32(_v0_i_low, _v0_i_high);
    __m128i _v01_s16high = _mm_packs_epi32(_v1_i_low, _v1_i_high);

    _v01_s16low = _mm_min_epi16(_v01_s16low, _mm_set1_epi16(127));
    _v01_s16high = _mm_min_epi16(_v01_s16high, _mm_set1_epi16(127));
    _v01_s16low = _mm_max_epi16(_v01_s16low, _mm_set1_epi16(-127));
    _v01_s16high = _mm_max_epi16(_v01_s16high, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16low, _v01_s16high);
    return _v8;
#endif // __AVX2__
}

static NCNN_FORCEINLINE void _mm256_comp_fmadd_ps4(__m256& _sum,
        const __m256& _w0, const __m256& _w1, const __m256& _w2, const __m256& _w3,
        const __m256& _v0, const __m256& _v1, const __m256& _v2, const __m256& _v3)
{
    __m256 _mul0 = _mm256_mul_ps(_w0, _v0);
    __m256 _mul1 = _mm256_mul_ps(_w1, _v1);
    __m256 _sum01 = _mm256_add_ps(_mul0, _mul1);
    __m256 _mul2 = _mm256_mul_ps(_w2, _v2);
    __m256 _mul3 = _mm256_mul_ps(_w3, _v3);
    __m256 _sum23 = _mm256_add_ps(_mul2, _mul3);
    __m256 _sum0123 = _mm256_add_ps(_sum01, _sum23);
    _sum = _mm256_add_ps(_sum, _sum0123);
}

static NCNN_FORCEINLINE void _mm256_comp_fmadd_ps8(__m256& _sum,
        const __m256& _w0, const __m256& _w1, const __m256& _w2, const __m256& _w3, const __m256& _w4, const __m256& _w5, const __m256& _w6, const __m256& _w7,
        const __m256& _v0, const __m256& _v1, const __m256& _v2, const __m256& _v3, const __m256& _v4, const __m256& _v5, const __m256& _v6, const __m256& _v7)
{
    _mm256_comp_fmadd_ps4(_sum, _w0, _w1, _w2, _w3, _v0, _v1, _v2, _v3);

    _mm256_comp_fmadd_ps4(_sum, _w4, _w5, _w6, _w7, _v4, _v5, _v6, _v7);
}

#if __AVX512F__
static NCNN_FORCEINLINE void transpose16_ps(__m512& _r0, __m512& _r1, __m512& _r2, __m512& _r3, __m512& _r4, __m512& _r5, __m512& _r6, __m512& _r7,
        __m512& _r8, __m512& _r9, __m512& _ra, __m512& _rb, __m512& _rc, __m512& _rd, __m512& _re, __m512& _rf)
{
    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);
    __m512 _tmp4 = _mm512_unpacklo_ps(_r4, _r5);
    __m512 _tmp5 = _mm512_unpackhi_ps(_r4, _r5);
    __m512 _tmp6 = _mm512_unpacklo_ps(_r6, _r7);
    __m512 _tmp7 = _mm512_unpackhi_ps(_r6, _r7);
    __m512 _tmp8 = _mm512_unpacklo_ps(_r8, _r9);
    __m512 _tmp9 = _mm512_unpackhi_ps(_r8, _r9);
    __m512 _tmpa = _mm512_unpacklo_ps(_ra, _rb);
    __m512 _tmpb = _mm512_unpackhi_ps(_ra, _rb);
    __m512 _tmpc = _mm512_unpacklo_ps(_rc, _rd);
    __m512 _tmpd = _mm512_unpackhi_ps(_rc, _rd);
    __m512 _tmpe = _mm512_unpacklo_ps(_re, _rf);
    __m512 _tmpf = _mm512_unpackhi_ps(_re, _rf);

    __m512 _tmpg = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmph = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpi = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpj = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpk = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpl = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpm = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpn = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpo = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpp = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpq = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpr = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmps = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpt = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpu = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpv = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp5 = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp6 = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp7 = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp8 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp9 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpa = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpb = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpc = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpd = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpe = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpf = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(3, 1, 3, 1));

    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
    _r2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
    _r3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
    _r4 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
    _r5 = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
    _r6 = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
    _r7 = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
    _r8 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
    _r9 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
    _ra = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
    _rb = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
    _rc = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
    _rd = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
    _re = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
    _rf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE float _mm512_comp_reduce_add_ps(__m512 x)
{
    const __m256 x256 = _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
    const __m128 x128 = _mm_add_ps(_mm256_castps256_ps128(x256), _mm256_extractf128_ps(x256, 1));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static NCNN_FORCEINLINE float _mm512_comp_reduce_max_ps(__m512 x)
{
    const __m256 x256 = _mm256_max_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
    const __m128 x128 = _mm_max_ps(_mm256_castps256_ps128(x256), _mm256_extractf128_ps(x256, 1));
    const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#endif // X86_USABILITY_H
