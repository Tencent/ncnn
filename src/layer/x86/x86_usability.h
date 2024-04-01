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

#include <stdint.h>
#if __SSE2__
#include <emmintrin.h>
#if __SSE4_1__
#include <smmintrin.h>
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
static NCNN_FORCEINLINE void transpose4x8_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = _mm_unpacklo_epi32(_r0, _r1);
    __m128i _tmp1 = _mm_unpackhi_epi32(_r0, _r1);
    __m128i _tmp2 = _mm_unpacklo_epi32(_r2, _r3);
    __m128i _tmp3 = _mm_unpackhi_epi32(_r2, _r3);
    __m128i _tmp4 = _mm_unpacklo_epi32(_r4, _r5);
    __m128i _tmp5 = _mm_unpackhi_epi32(_r4, _r5);
    __m128i _tmp6 = _mm_unpacklo_epi32(_r6, _r7);
    __m128i _tmp7 = _mm_unpackhi_epi32(_r6, _r7);

    _r0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
    _r1 = _mm_unpacklo_epi64(_tmp4, _tmp6);
    _r2 = _mm_unpackhi_epi64(_tmp0, _tmp2);
    _r3 = _mm_unpackhi_epi64(_tmp4, _tmp6);
    _r4 = _mm_unpacklo_epi64(_tmp1, _tmp3);
    _r5 = _mm_unpacklo_epi64(_tmp5, _tmp7);
    _r6 = _mm_unpackhi_epi64(_tmp1, _tmp3);
    _r7 = _mm_unpackhi_epi64(_tmp5, _tmp7);
}

static NCNN_FORCEINLINE void transpose4x4_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = _mm_unpacklo_epi32(_r0, _r1);
    __m128i _tmp1 = _mm_unpackhi_epi32(_r0, _r1);
    __m128i _tmp2 = _mm_unpacklo_epi32(_r2, _r3);
    __m128i _tmp3 = _mm_unpackhi_epi32(_r2, _r3);

    _r0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
    _r1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
    _r2 = _mm_unpacklo_epi64(_tmp1, _tmp3);
    _r3 = _mm_unpackhi_epi64(_tmp1, _tmp3);
}

static NCNN_FORCEINLINE void transpose8x8_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _tmp1 = _mm_unpackhi_epi16(_r0, _r1);
    __m128i _tmp2 = _mm_unpacklo_epi16(_r2, _r3);
    __m128i _tmp3 = _mm_unpackhi_epi16(_r2, _r3);
    __m128i _tmp4 = _mm_unpacklo_epi16(_r4, _r5);
    __m128i _tmp5 = _mm_unpackhi_epi16(_r4, _r5);
    __m128i _tmp6 = _mm_unpacklo_epi16(_r6, _r7);
    __m128i _tmp7 = _mm_unpackhi_epi16(_r6, _r7);

    __m128i _tmp8 = _mm_unpacklo_epi32(_tmp0, _tmp2);
    __m128i _tmp9 = _mm_unpackhi_epi32(_tmp0, _tmp2);
    __m128i _tmpa = _mm_unpacklo_epi32(_tmp1, _tmp3);
    __m128i _tmpb = _mm_unpackhi_epi32(_tmp1, _tmp3);
    __m128i _tmpc = _mm_unpacklo_epi32(_tmp4, _tmp6);
    __m128i _tmpd = _mm_unpackhi_epi32(_tmp4, _tmp6);
    __m128i _tmpe = _mm_unpacklo_epi32(_tmp5, _tmp7);
    __m128i _tmpf = _mm_unpackhi_epi32(_tmp5, _tmp7);

    _r0 = _mm_unpacklo_epi64(_tmp8, _tmpc);
    _r1 = _mm_unpackhi_epi64(_tmp8, _tmpc);
    _r2 = _mm_unpacklo_epi64(_tmp9, _tmpd);
    _r3 = _mm_unpackhi_epi64(_tmp9, _tmpd);
    _r4 = _mm_unpacklo_epi64(_tmpa, _tmpe);
    _r5 = _mm_unpackhi_epi64(_tmpa, _tmpe);
    _r6 = _mm_unpacklo_epi64(_tmpb, _tmpf);
    _r7 = _mm_unpackhi_epi64(_tmpb, _tmpf);
}

static NCNN_FORCEINLINE void transpose8x4_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _tmp1 = _mm_unpackhi_epi16(_r0, _r1);
    __m128i _tmp2 = _mm_unpacklo_epi16(_r2, _r3);
    __m128i _tmp3 = _mm_unpackhi_epi16(_r2, _r3);

    _r0 = _mm_unpacklo_epi32(_tmp0, _tmp2);
    _r1 = _mm_unpackhi_epi32(_tmp0, _tmp2);
    _r2 = _mm_unpacklo_epi32(_tmp1, _tmp3);
    _r3 = _mm_unpackhi_epi32(_tmp1, _tmp3);
}

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

static NCNN_FORCEINLINE __m128 bfloat2float_sse(const __m128i& v0)
{
    __m128i _zero = _mm_setzero_si128();
    __m128i _a = _mm_unpacklo_epi16(_zero, v0);
    __m128 _v = _mm_castsi128_ps(_a);
    return _v;
}

static NCNN_FORCEINLINE __m128i float2bfloat_sse(const __m128& v0, const __m128& v1)
{
#if __AVX512BF16__
    __m128i _v = (__m128i)_mm256_cvtneps_pbh(_mm256_insertf128_ps(_mm256_castps128_ps256(v0), v1, 1));
#else
    __m128i _a = _mm_castps_si128(v0);
    __m128i _b = _mm_castps_si128(v1);
#if __SSE4_1__
    _a = _mm_srli_epi32(_a, 16);
    _b = _mm_srli_epi32(_b, 16);
    __m128i _v = _mm_packus_epi32(_a, _b);
#else
    _a = _mm_shufflelo_epi16(_a, _MM_SHUFFLE(2, 0, 3, 1));
    _b = _mm_shufflelo_epi16(_b, _MM_SHUFFLE(2, 0, 3, 1));
    _a = _mm_shufflehi_epi16(_a, _MM_SHUFFLE(2, 0, 3, 1));
    _b = _mm_shufflehi_epi16(_b, _MM_SHUFFLE(2, 0, 3, 1));
    __m128i _v = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_a), _mm_castsi128_ps(_b), _MM_SHUFFLE(2, 0, 2, 0)));
#endif
#endif
    return _v;
}

#ifndef __FMA__
static NCNN_FORCEINLINE __m128 _mm_comp_fmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_add_ps(_mm_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_sub_ps(_c, _mm_mul_ps(_a, _b));
}
static NCNN_FORCEINLINE __m128 _mm_comp_fmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_sub_ps(_mm_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_sub_ps(_c, _mm_mul_ps(_mm_mul_ps(_a, _b), _mm_set1_ps(-1)));
}
#else
static NCNN_FORCEINLINE __m128 _mm_comp_fmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_fmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    // return -a * b + c
    return _mm_fnmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_fmsub_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m128 _mm_comp_fnmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return _mm_fnmsub_ps(_a, _b, _c);
}
#endif // !__FMA__

#if __AVX__
#ifndef __FMA__
static NCNN_FORCEINLINE __m256 _mm256_comp_fmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return _mm256_add_ps(_mm256_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return _mm256_sub_ps(_c, _mm256_mul_ps(_a, _b));
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return _mm256_sub_ps(_mm256_mul_ps(_a, _b), _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return _mm256_sub_ps(_c, _mm256_mul_ps(_mm256_mul_ps(_a, _b), _mm256_set1_ps(-1)));
}
#else
static NCNN_FORCEINLINE __m256 _mm256_comp_fmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    // return a * b + c
    return _mm256_fmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    // return -a * b + c
    return _mm256_fnmadd_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    // return a * b - c
    return _mm256_fmsub_ps(_a, _b, _c);
}
static NCNN_FORCEINLINE __m256 _mm256_comp_fnmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    // return -(a * b) - c
    return _mm256_fnmsub_ps(_a, _b, _c);
}
#endif

static NCNN_FORCEINLINE __m256 _mm256_fmadd_1_ps(const __m256& a, const __m256& b, float c)
{
    return _mm256_comp_fmadd_ps(b, _mm256_set1_ps(c), a);
}

static NCNN_FORCEINLINE __m256 _mm256_fmrsub_1_ps(const __m256& a, const __m256& b, float c)
{
    // return a - b * c
    return _mm256_comp_fnmadd_ps(b, _mm256_set1_ps(c), a);
}

static NCNN_FORCEINLINE void transpose8x12_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7,
        __m256& _r8, __m256& _r9, __m256& _ra, __m256& _rb)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);

    __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
    _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
    _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose8x8_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);

    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose8x4_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);

    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
    _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose8x2_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);

    _r0 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose2x8_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = _mm256_permute2f128_ps(_r0, _r1, _MM_SHUFFLE(0, 2, 0, 0));
    __m256 _tmp1 = _mm256_permute2f128_ps(_r0, _r1, _MM_SHUFFLE(0, 3, 0, 1));

    _r0 = _mm256_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm256_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE void transpose3x8_ps(__m256& _r0, __m256& _r1, __m256& _r2)
{
    __m256 _tmp0 = _mm256_permute2f128_ps(_r0, _r1, _MM_SHUFFLE(0, 3, 0, 0));
    __m256 _tmp1 = _mm256_permute2f128_ps(_r0, _r2, _MM_SHUFFLE(0, 2, 0, 1));
    __m256 _tmp2 = _mm256_permute2f128_ps(_r1, _r2, _MM_SHUFFLE(0, 3, 0, 0));

    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(1, 0, 2, 1));
    __m256 _tmp5 = _mm256_shuffle_ps(_tmp1, _tmp2, _MM_SHUFFLE(2, 1, 3, 2));

    _r0 = _mm256_shuffle_ps(_tmp0, _tmp5, _MM_SHUFFLE(2, 0, 3, 0));
    _r1 = _mm256_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
    _r2 = _mm256_shuffle_ps(_tmp4, _tmp2, _MM_SHUFFLE(3, 0, 3, 1));
}

static NCNN_FORCEINLINE void transpose8x6_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);

    __m256 _tmp6 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp7 = _mm256_shuffle_ps(_tmp4, _tmp0, _MM_SHUFFLE(3, 2, 1, 0));
    __m256 _tmp8 = _mm256_shuffle_ps(_tmp2, _tmp4, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmp9 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpa = _mm256_shuffle_ps(_tmp5, _tmp1, _MM_SHUFFLE(3, 2, 1, 0));
    __m256 _tmpb = _mm256_shuffle_ps(_tmp3, _tmp5, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmp8, _tmp9, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpb, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
    _r4 = _mm256_permute2f128_ps(_tmp8, _tmp9, _MM_SHUFFLE(0, 3, 0, 1));
    _r5 = _mm256_permute2f128_ps(_tmpa, _tmpb, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose8x11_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7, __m256& _r8, __m256& _r9, __m256& _ra)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _r0);
    __m256 _tmpb = _mm256_shuffle_ps(_ra, _tmp1, _MM_SHUFFLE(3, 2, 1, 2));
    __m256 _tmpc = _mm256_unpacklo_ps(_r1, _r2);
    __m256 _tmpd = _mm256_unpackhi_ps(_r1, _r2);
    __m256 _tmpe = _mm256_unpacklo_ps(_r3, _r4);
    __m256 _tmpf = _mm256_unpackhi_ps(_r3, _r4);
    __m256 _tmpg = _mm256_unpacklo_ps(_r5, _r6);
    __m256 _tmph = _mm256_unpackhi_ps(_r5, _r6);
    __m256 _tmpi = _mm256_unpacklo_ps(_r7, _r8);
    __m256 _tmpj = _mm256_unpackhi_ps(_r7, _r8);
    __m256 _tmpk = _mm256_unpacklo_ps(_r9, _ra);
    __m256 _tmpl = _mm256_unpackhi_ps(_r9, _ra);

    __m256 _tmpm = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpo = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 0, 1, 0));
    __m256 _tmpp = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpq = _mm256_shuffle_ps(_tmpg, _tmpi, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpr = _mm256_shuffle_ps(_tmpk, _tmp1, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 _tmps = _mm256_shuffle_ps(_tmp3, _tmp5, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpt = _mm256_shuffle_ps(_tmp7, _tmp9, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpu = _mm256_shuffle_ps(_tmpb, _tmpd, _MM_SHUFFLE(3, 2, 2, 0));
    __m256 _tmpv = _mm256_shuffle_ps(_tmpf, _tmph, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpw = _mm256_shuffle_ps(_tmpj, _tmpl, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpm, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpo, _tmpp, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmpq, _tmpr, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmps, _tmpt, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpu, _tmpv, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmpw, _tmpm, _MM_SHUFFLE(0, 3, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpn, _tmpo, _MM_SHUFFLE(0, 3, 0, 1));
    _r7 = _mm256_permute2f128_ps(_tmpp, _tmpq, _MM_SHUFFLE(0, 3, 0, 1));
    _r8 = _mm256_permute2f128_ps(_tmpr, _tmps, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2f128_ps(_tmpt, _tmpu, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmpv, _tmpw, _MM_SHUFFLE(0, 3, 0, 1));
}

static void transpose8x18_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7, __m256& _r8, __m256& _r9, __m256& _ra, __m256& _rb, __m256& _rc, __m256& _rd, __m256& _re, __m256& _rf, __m256& _rg, __m256& _rh)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
    __m256 _tmpc = _mm256_unpacklo_ps(_rc, _rd);
    __m256 _tmpd = _mm256_unpackhi_ps(_rc, _rd);
    __m256 _tmpe = _mm256_unpacklo_ps(_re, _rf);
    __m256 _tmpf = _mm256_unpackhi_ps(_re, _rf);
    __m256 _tmpg = _mm256_unpacklo_ps(_rg, _rh);
    __m256 _tmph = _mm256_unpackhi_ps(_rg, _rh);

    __m256 _tmpi = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpj = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpl = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpm = _mm256_shuffle_ps(_tmpg, _tmp0, _MM_SHUFFLE(3, 2, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp2, _tmp4, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpo = _mm256_shuffle_ps(_tmp6, _tmp8, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpp = _mm256_shuffle_ps(_tmpa, _tmpc, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpq = _mm256_shuffle_ps(_tmpe, _tmpg, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpr = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmps = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpt = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpu = _mm256_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpv = _mm256_shuffle_ps(_tmph, _tmp1, _MM_SHUFFLE(3, 2, 1, 0));
    __m256 _tmpw = _mm256_shuffle_ps(_tmp3, _tmp5, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpx = _mm256_shuffle_ps(_tmp7, _tmp9, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpy = _mm256_shuffle_ps(_tmpb, _tmpd, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpz = _mm256_shuffle_ps(_tmpf, _tmph, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpi, _tmpj, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmpm, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpo, _tmpp, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpq, _tmpr, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmps, _tmpt, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpu, _tmpv, _MM_SHUFFLE(0, 2, 0, 0));
    _r7 = _mm256_permute2f128_ps(_tmpw, _tmpx, _MM_SHUFFLE(0, 2, 0, 0));
    _r8 = _mm256_permute2f128_ps(_tmpy, _tmpz, _MM_SHUFFLE(0, 2, 0, 0));
    _r9 = _mm256_permute2f128_ps(_tmpi, _tmpj, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmpk, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2f128_ps(_tmpm, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));
    _rc = _mm256_permute2f128_ps(_tmpo, _tmpp, _MM_SHUFFLE(0, 3, 0, 1));
    _rd = _mm256_permute2f128_ps(_tmpq, _tmpr, _MM_SHUFFLE(0, 3, 0, 1));
    _re = _mm256_permute2f128_ps(_tmps, _tmpt, _MM_SHUFFLE(0, 3, 0, 1));
    _rf = _mm256_permute2f128_ps(_tmpu, _tmpv, _MM_SHUFFLE(0, 3, 0, 1));
    _rg = _mm256_permute2f128_ps(_tmpw, _tmpx, _MM_SHUFFLE(0, 3, 0, 1));
    _rh = _mm256_permute2f128_ps(_tmpy, _tmpz, _MM_SHUFFLE(0, 3, 0, 1));
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

static NCNN_FORCEINLINE __m256 bfloat2float_avx(const __m128i& v0)
{
#if __AVX512BF16__
    __m256 _v = _mm256_cvtpbh_ps((__m128bh)v0);
#else
    __m128i _zero = _mm_setzero_si128();
    __m128i _a = _mm_unpacklo_epi16(_zero, v0);
    __m128i _b = _mm_unpackhi_epi16(_zero, v0);
    __m256 _v = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_a), _b, 1));
#endif
    return _v;
}

static NCNN_FORCEINLINE __m128i float2bfloat_avx(const __m256& v0)
{
#if __AVX512BF16__
    __m128i _v = (__m128i)_mm256_cvtneps_pbh(v0);
#else
    __m256i _ab = _mm256_castps_si256(v0);
#if __AVX2__
    _ab = _mm256_srli_epi32(_ab, 16);
    __m128i _a = _mm256_extractf128_si256(_ab, 0);
    __m128i _b = _mm256_extractf128_si256(_ab, 1);
#else
    __m128i _a = _mm256_extractf128_si256(_ab, 0);
    __m128i _b = _mm256_extractf128_si256(_ab, 1);
    _a = _mm_srli_epi32(_a, 16);
    _b = _mm_srli_epi32(_b, 16);
#endif
    __m128i _v = _mm_packus_epi32(_a, _b);
#endif
    return _v;
}

static NCNN_FORCEINLINE __m256i float2bfloat_avx(const __m256& v0, const __m256& v1)
{
#if __AVX512BF16__
    __m128i _v0 = (__m128i)_mm256_cvtneps_pbh(v0);
    __m128i _v1 = (__m128i)_mm256_cvtneps_pbh(v1);
    __m256i _v = _mm256_insertf128_si256(_mm256_castsi128_si256(_v0), _v1, 1);
#else
    __m256i _a = _mm256_castps_si256(v0);
    __m256i _b = _mm256_castps_si256(v1);
#if __AVX2__
    _a = _mm256_srli_epi32(_a, 16);
    _b = _mm256_srli_epi32(_b, 16);
    __m256i _v = _mm256_packus_epi32(_a, _b);
    _v = _mm256_permute4x64_epi64(_v, _MM_SHUFFLE(3, 1, 2, 0));
#else
    __m128i _a0 = _mm256_extractf128_si256(_a, 0);
    __m128i _a1 = _mm256_extractf128_si256(_a, 1);
    __m128i _b0 = _mm256_extractf128_si256(_b, 0);
    __m128i _b1 = _mm256_extractf128_si256(_b, 1);
    _a0 = _mm_srli_epi32(_a0, 16);
    _a1 = _mm_srli_epi32(_a1, 16);
    _b0 = _mm_srli_epi32(_b0, 16);
    _b1 = _mm_srli_epi32(_b1, 16);
    __m128i _v0 = _mm_packus_epi32(_a0, _a1);
    __m128i _v1 = _mm_packus_epi32(_b0, _b1);
    __m256i _v = _mm256_insertf128_si256(_mm256_castsi128_si256(_v0), _v1, 1);
#endif
#endif
    return _v;
}

#if __AVX2__
static NCNN_FORCEINLINE void transpose8x2_epi32(__m256i& _r0, __m256i& _r1)
{
    __m256i _tmp0 = _mm256_unpacklo_epi32(_r0, _r1);
    __m256i _tmp1 = _mm256_unpackhi_epi32(_r0, _r1);

    _r0 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose16x8_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7)
{
    __m256i _tmp0 = _mm256_unpacklo_epi16(_r0, _r1);
    __m256i _tmp1 = _mm256_unpackhi_epi16(_r0, _r1);
    __m256i _tmp2 = _mm256_unpacklo_epi16(_r2, _r3);
    __m256i _tmp3 = _mm256_unpackhi_epi16(_r2, _r3);
    __m256i _tmp4 = _mm256_unpacklo_epi16(_r4, _r5);
    __m256i _tmp5 = _mm256_unpackhi_epi16(_r4, _r5);
    __m256i _tmp6 = _mm256_unpacklo_epi16(_r6, _r7);
    __m256i _tmp7 = _mm256_unpackhi_epi16(_r6, _r7);

    __m256i _tmpg = _mm256_unpacklo_epi32(_tmp0, _tmp2);
    __m256i _tmph = _mm256_unpackhi_epi32(_tmp0, _tmp2);
    __m256i _tmpi = _mm256_unpacklo_epi32(_tmp1, _tmp3);
    __m256i _tmpj = _mm256_unpackhi_epi32(_tmp1, _tmp3);
    __m256i _tmpk = _mm256_unpacklo_epi32(_tmp4, _tmp6);
    __m256i _tmpl = _mm256_unpackhi_epi32(_tmp4, _tmp6);
    __m256i _tmpm = _mm256_unpacklo_epi32(_tmp5, _tmp7);
    __m256i _tmpn = _mm256_unpackhi_epi32(_tmp5, _tmp7);

    _tmp0 = _mm256_unpacklo_epi64(_tmpg, _tmpk);
    _tmp1 = _mm256_unpackhi_epi64(_tmpg, _tmpk);
    _tmp2 = _mm256_unpacklo_epi64(_tmph, _tmpl);
    _tmp3 = _mm256_unpackhi_epi64(_tmph, _tmpl);
    _tmp4 = _mm256_unpacklo_epi64(_tmpi, _tmpm);
    _tmp5 = _mm256_unpackhi_epi64(_tmpi, _tmpm);
    _tmp6 = _mm256_unpacklo_epi64(_tmpj, _tmpn);
    _tmp7 = _mm256_unpackhi_epi64(_tmpj, _tmpn);

    _r0 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2x128_si256(_tmp2, _tmp3, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2x128_si256(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2x128_si256(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2x128_si256(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));
    _r5 = _mm256_permute2x128_si256(_tmp2, _tmp3, _MM_SHUFFLE(0, 3, 0, 1));
    _r6 = _mm256_permute2x128_si256(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
    _r7 = _mm256_permute2x128_si256(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
}

#if __AVX512F__
static NCNN_FORCEINLINE void transpose16x16_ps(__m512& _r0, __m512& _r1, __m512& _r2, __m512& _r3, __m512& _r4, __m512& _r5, __m512& _r6, __m512& _r7,
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

static NCNN_FORCEINLINE void transpose16x12_ps(__m512& _r0, __m512& _r1, __m512& _r2, __m512& _r3, __m512& _r4, __m512& _r5, __m512& _r6, __m512& _r7,
        __m512& _r8, __m512& _r9, __m512& _ra, __m512& _rb)
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

    __m512 _tmpc = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpd = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpe = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpf = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpg = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmph = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpi = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpj = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpk = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpl = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpm = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpn = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmpc, _tmpg, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmpk, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_f32x4(_tmpe, _tmpi, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_f32x4(_tmpm, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp5 = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp6 = _mm512_shuffle_f32x4(_tmpc, _tmpg, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp7 = _mm512_shuffle_f32x4(_tmpk, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp8 = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp9 = _mm512_shuffle_f32x4(_tmpe, _tmpi, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpa = _mm512_shuffle_f32x4(_tmpm, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpb = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(3, 1, 3, 1));

    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
    _r2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
    _r3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
    _r4 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
    _r5 = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
    _r6 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
    _r7 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
    _r8 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
    _r9 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
    _ra = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
    _rb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE void transpose16x8_ps(__m512& _r0, __m512& _r1, __m512& _r2, __m512& _r3, __m512& _r4, __m512& _r5, __m512& _r6, __m512& _r7)
{
    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);
    __m512 _tmp4 = _mm512_unpacklo_ps(_r4, _r5);
    __m512 _tmp5 = _mm512_unpackhi_ps(_r4, _r5);
    __m512 _tmp6 = _mm512_unpacklo_ps(_r6, _r7);
    __m512 _tmp7 = _mm512_unpackhi_ps(_r6, _r7);

    __m512 _tmp8 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmp9 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpa = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpb = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpc = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpd = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpe = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpf = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp5 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp6 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp7 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
    _r2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
    _r3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
    _r4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
    _r5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
    _r6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
    _r7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE void transpose16x4_ps(__m512& _r0, __m512& _r1, __m512& _r2, __m512& _r3)
{
    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);

    __m512 _tmp4 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmp5 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmp6 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmp7 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
    _r2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
    _r3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE void transpose16x2_ps(__m512& _r0, __m512& _r1)
{
    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);

    __m512 _tmp2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
    __m512 _tmp3 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));

    _r0 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
}

static NCNN_FORCEINLINE void transpose8x16_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7,
        __m256& _r8, __m256& _r9, __m256& _ra, __m256& _rb, __m256& _rc, __m256& _rd, __m256& _re, __m256& _rf)
{
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
    __m256 _tmpc = _mm256_unpacklo_ps(_rc, _rd);
    __m256 _tmpd = _mm256_unpackhi_ps(_rc, _rd);
    __m256 _tmpe = _mm256_unpacklo_ps(_re, _rf);
    __m256 _tmpf = _mm256_unpackhi_ps(_re, _rf);

    __m256 _tmpg = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmph = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpi = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpj = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpk = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpl = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpm = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpo = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpp = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpq = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpr = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmps = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpt = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpu = _mm256_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpv = _mm256_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpg, _tmpk, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpo, _tmps, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpp, _tmpt, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpi, _tmpm, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmpq, _tmpu, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r7 = _mm256_permute2f128_ps(_tmpr, _tmpv, _MM_SHUFFLE(0, 2, 0, 0));
    _r8 = _mm256_permute2f128_ps(_tmpg, _tmpk, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2f128_ps(_tmpo, _tmps, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2f128_ps(_tmpp, _tmpt, _MM_SHUFFLE(0, 3, 0, 1));
    _rc = _mm256_permute2f128_ps(_tmpi, _tmpm, _MM_SHUFFLE(0, 3, 0, 1));
    _rd = _mm256_permute2f128_ps(_tmpq, _tmpu, _MM_SHUFFLE(0, 3, 0, 1));
    _re = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));
    _rf = _mm256_permute2f128_ps(_tmpr, _tmpv, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose16x16_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7,
        __m256i& _r8, __m256i& _r9, __m256i& _ra, __m256i& _rb, __m256i& _rc, __m256i& _rd, __m256i& _re, __m256i& _rf)
{
    __m256i _tmp0 = _mm256_unpacklo_epi16(_r0, _r1);
    __m256i _tmp1 = _mm256_unpackhi_epi16(_r0, _r1);
    __m256i _tmp2 = _mm256_unpacklo_epi16(_r2, _r3);
    __m256i _tmp3 = _mm256_unpackhi_epi16(_r2, _r3);
    __m256i _tmp4 = _mm256_unpacklo_epi16(_r4, _r5);
    __m256i _tmp5 = _mm256_unpackhi_epi16(_r4, _r5);
    __m256i _tmp6 = _mm256_unpacklo_epi16(_r6, _r7);
    __m256i _tmp7 = _mm256_unpackhi_epi16(_r6, _r7);
    __m256i _tmp8 = _mm256_unpacklo_epi16(_r8, _r9);
    __m256i _tmp9 = _mm256_unpackhi_epi16(_r8, _r9);
    __m256i _tmpa = _mm256_unpacklo_epi16(_ra, _rb);
    __m256i _tmpb = _mm256_unpackhi_epi16(_ra, _rb);
    __m256i _tmpc = _mm256_unpacklo_epi16(_rc, _rd);
    __m256i _tmpd = _mm256_unpackhi_epi16(_rc, _rd);
    __m256i _tmpe = _mm256_unpacklo_epi16(_re, _rf);
    __m256i _tmpf = _mm256_unpackhi_epi16(_re, _rf);

    __m256i _tmpg = _mm256_unpacklo_epi32(_tmp0, _tmp2);
    __m256i _tmph = _mm256_unpackhi_epi32(_tmp0, _tmp2);
    __m256i _tmpi = _mm256_unpacklo_epi32(_tmp1, _tmp3);
    __m256i _tmpj = _mm256_unpackhi_epi32(_tmp1, _tmp3);
    __m256i _tmpk = _mm256_unpacklo_epi32(_tmp4, _tmp6);
    __m256i _tmpl = _mm256_unpackhi_epi32(_tmp4, _tmp6);
    __m256i _tmpm = _mm256_unpacklo_epi32(_tmp5, _tmp7);
    __m256i _tmpn = _mm256_unpackhi_epi32(_tmp5, _tmp7);
    __m256i _tmpo = _mm256_unpacklo_epi32(_tmp8, _tmpa);
    __m256i _tmpp = _mm256_unpackhi_epi32(_tmp8, _tmpa);
    __m256i _tmpq = _mm256_unpacklo_epi32(_tmp9, _tmpb);
    __m256i _tmpr = _mm256_unpackhi_epi32(_tmp9, _tmpb);
    __m256i _tmps = _mm256_unpacklo_epi32(_tmpc, _tmpe);
    __m256i _tmpt = _mm256_unpackhi_epi32(_tmpc, _tmpe);
    __m256i _tmpu = _mm256_unpacklo_epi32(_tmpd, _tmpf);
    __m256i _tmpv = _mm256_unpackhi_epi32(_tmpd, _tmpf);

    _tmp0 = _mm256_unpacklo_epi64(_tmpg, _tmpk);
    _tmp1 = _mm256_unpackhi_epi64(_tmpg, _tmpk);
    _tmp2 = _mm256_unpacklo_epi64(_tmph, _tmpl);
    _tmp3 = _mm256_unpackhi_epi64(_tmph, _tmpl);
    _tmp4 = _mm256_unpacklo_epi64(_tmpi, _tmpm);
    _tmp5 = _mm256_unpackhi_epi64(_tmpi, _tmpm);
    _tmp6 = _mm256_unpacklo_epi64(_tmpj, _tmpn);
    _tmp7 = _mm256_unpackhi_epi64(_tmpj, _tmpn);
    _tmp8 = _mm256_unpacklo_epi64(_tmpo, _tmps);
    _tmp9 = _mm256_unpackhi_epi64(_tmpo, _tmps);
    _tmpa = _mm256_unpacklo_epi64(_tmpp, _tmpt);
    _tmpb = _mm256_unpackhi_epi64(_tmpp, _tmpt);
    _tmpc = _mm256_unpacklo_epi64(_tmpq, _tmpu);
    _tmpd = _mm256_unpackhi_epi64(_tmpq, _tmpu);
    _tmpe = _mm256_unpacklo_epi64(_tmpr, _tmpv);
    _tmpf = _mm256_unpackhi_epi64(_tmpr, _tmpv);

    _r0 = _mm256_permute2x128_si256(_tmp0, _tmp8, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2x128_si256(_tmp1, _tmp9, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2x128_si256(_tmp2, _tmpa, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2x128_si256(_tmp3, _tmpb, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2x128_si256(_tmp4, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2x128_si256(_tmp5, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2x128_si256(_tmp6, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
    _r7 = _mm256_permute2x128_si256(_tmp7, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
    _r8 = _mm256_permute2x128_si256(_tmp0, _tmp8, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2x128_si256(_tmp1, _tmp9, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2x128_si256(_tmp2, _tmpa, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2x128_si256(_tmp3, _tmpb, _MM_SHUFFLE(0, 3, 0, 1));
    _rc = _mm256_permute2x128_si256(_tmp4, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
    _rd = _mm256_permute2x128_si256(_tmp5, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
    _re = _mm256_permute2x128_si256(_tmp6, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
    _rf = _mm256_permute2x128_si256(_tmp7, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
}

static NCNN_FORCEINLINE void transpose8x16_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7, __m128i& _r8, __m128i& _r9, __m128i& _ra, __m128i& _rb, __m128i& _rc, __m128i& _rd, __m128i& _re, __m128i& _rf)
{
    __m128i _tmp0 = _mm_unpacklo_epi16(_r0, _r1);
    __m128i _tmp1 = _mm_unpackhi_epi16(_r0, _r1);
    __m128i _tmp2 = _mm_unpacklo_epi16(_r2, _r3);
    __m128i _tmp3 = _mm_unpackhi_epi16(_r2, _r3);
    __m128i _tmp4 = _mm_unpacklo_epi16(_r4, _r5);
    __m128i _tmp5 = _mm_unpackhi_epi16(_r4, _r5);
    __m128i _tmp6 = _mm_unpacklo_epi16(_r6, _r7);
    __m128i _tmp7 = _mm_unpackhi_epi16(_r6, _r7);
    __m128i _tmp8 = _mm_unpacklo_epi16(_r8, _r9);
    __m128i _tmp9 = _mm_unpackhi_epi16(_r8, _r9);
    __m128i _tmpa = _mm_unpacklo_epi16(_ra, _rb);
    __m128i _tmpb = _mm_unpackhi_epi16(_ra, _rb);
    __m128i _tmpc = _mm_unpacklo_epi16(_rc, _rd);
    __m128i _tmpd = _mm_unpackhi_epi16(_rc, _rd);
    __m128i _tmpe = _mm_unpacklo_epi16(_re, _rf);
    __m128i _tmpf = _mm_unpackhi_epi16(_re, _rf);

    __m128i _tmpg = _mm_unpacklo_epi32(_tmp0, _tmp2);
    __m128i _tmph = _mm_unpackhi_epi32(_tmp0, _tmp2);
    __m128i _tmpi = _mm_unpacklo_epi32(_tmp1, _tmp3);
    __m128i _tmpj = _mm_unpackhi_epi32(_tmp1, _tmp3);
    __m128i _tmpk = _mm_unpacklo_epi32(_tmp4, _tmp6);
    __m128i _tmpl = _mm_unpackhi_epi32(_tmp4, _tmp6);
    __m128i _tmpm = _mm_unpacklo_epi32(_tmp5, _tmp7);
    __m128i _tmpn = _mm_unpackhi_epi32(_tmp5, _tmp7);
    __m128i _tmpo = _mm_unpacklo_epi32(_tmp8, _tmpa);
    __m128i _tmpp = _mm_unpackhi_epi32(_tmp8, _tmpa);
    __m128i _tmpq = _mm_unpacklo_epi32(_tmp9, _tmpb);
    __m128i _tmpr = _mm_unpackhi_epi32(_tmp9, _tmpb);
    __m128i _tmps = _mm_unpacklo_epi32(_tmpc, _tmpe);
    __m128i _tmpt = _mm_unpackhi_epi32(_tmpc, _tmpe);
    __m128i _tmpu = _mm_unpacklo_epi32(_tmpd, _tmpf);
    __m128i _tmpv = _mm_unpackhi_epi32(_tmpd, _tmpf);

    _r0 = _mm_unpacklo_epi64(_tmpg, _tmpk);
    _r1 = _mm_unpacklo_epi64(_tmpo, _tmps);
    _r2 = _mm_unpackhi_epi64(_tmpg, _tmpk);
    _r3 = _mm_unpackhi_epi64(_tmpo, _tmps);
    _r4 = _mm_unpacklo_epi64(_tmph, _tmpl);
    _r5 = _mm_unpacklo_epi64(_tmpp, _tmpt);
    _r6 = _mm_unpackhi_epi64(_tmph, _tmpl);
    _r7 = _mm_unpackhi_epi64(_tmpp, _tmpt);
    _r8 = _mm_unpacklo_epi64(_tmpi, _tmpm);
    _r9 = _mm_unpacklo_epi64(_tmpq, _tmpu);
    _ra = _mm_unpackhi_epi64(_tmpi, _tmpm);
    _rb = _mm_unpackhi_epi64(_tmpq, _tmpu);
    _rc = _mm_unpacklo_epi64(_tmpj, _tmpn);
    _rd = _mm_unpacklo_epi64(_tmpr, _tmpv);
    _re = _mm_unpackhi_epi64(_tmpj, _tmpn);
    _rf = _mm_unpackhi_epi64(_tmpr, _tmpv);
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

static NCNN_FORCEINLINE __m512 bfloat2float_avx512(const __m256i& v0)
{
#if __AVX512BF16__
    __m512 _v = _mm512_cvtpbh_ps((__m256bh)v0);
#else
    __m256i _zero = _mm256_setzero_si256();
    __m256i _a = _mm256_unpacklo_epi16(_zero, v0);
    __m256i _b = _mm256_unpackhi_epi16(_zero, v0);
    __m256i _c = _mm256_permute2x128_si256(_a, _b, _MM_SHUFFLE(0, 2, 0, 0));
    __m256i _d = _mm256_permute2x128_si256(_a, _b, _MM_SHUFFLE(0, 3, 0, 1));
    __m512 _v = _mm512_castsi512_ps(_mm512_inserti32x8(_mm512_castsi256_si512(_c), _d, 1));
#endif
    return _v;
}

static NCNN_FORCEINLINE __m256i float2bfloat_avx512(const __m512& v0)
{
#if __AVX512BF16__
    __m256i _v = (__m256i)_mm512_cvtneps_pbh(v0);
#else
    __m512i _ab = _mm512_castps_si512(v0);
    _ab = _mm512_srli_epi32(_ab, 16);
    __m256i _a = _mm512_extracti32x8_epi32(_ab, 0);
    __m256i _b = _mm512_extracti32x8_epi32(_ab, 1);
    __m256i _v = _mm256_packus_epi32(_a, _b);
    _v = _mm256_permute4x64_epi64(_v, _MM_SHUFFLE(3, 1, 2, 0));
#endif
    return _v;
}

static NCNN_FORCEINLINE __m512i float2bfloat_avx512(const __m512& v0, const __m512& v1)
{
#if __AVX512BF16__
    __m256bh _v0 = _mm512_cvtneps_pbh(v0);
    __m256bh _v1 = _mm512_cvtneps_pbh(v1);
    __m512i _v = _mm512_inserti32x8(_mm512_castsi256_si512((__m256i)_v0), (__m256i)_v1, 1);
#else
    __m512i _a = _mm512_castps_si512(v0);
    __m512i _b = _mm512_castps_si512(v1);
    _a = _mm512_srli_epi32(_a, 16);
    _b = _mm512_srli_epi32(_b, 16);
    __m512i _v = _mm512_packus_epi32(_a, _b);
    _v = _mm512_permutex_epi64(_v, _MM_SHUFFLE(3, 1, 2, 0));
    _v = _mm512_shuffle_i32x4(_v, _v, _MM_SHUFFLE(3, 1, 2, 0));
#endif
    return _v;
}

#endif // __AVX512F__
#endif // __AVX2__
#endif // __AVX__
#endif // __SSE2__

#endif // X86_USABILITY_H
