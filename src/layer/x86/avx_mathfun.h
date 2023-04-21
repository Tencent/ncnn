/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef AVX_MATHFUN_H
#define AVX_MATHFUN_H

#include <immintrin.h>
#include <emmintrin.h>
#include <x86_usability.h>

/* yes I know, the top of this file is quite ugly */

#ifdef _MSC_VER /* visual c++ */
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#else /* gcc or icc */
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#endif

#define _PI32AVX_CONST(Name, Val) \
    static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = {Val, Val, Val, Val}

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val) \
    static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST256(Name, Val) \
    static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS256_CONST_TYPE(Name, Type, Val) \
    static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = {Val, Val, Val, Val, Val, Val, Val, Val}

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS256_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS256_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS256_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS256_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS256_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS256_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS256_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS256_CONST(cephes_log_q1, -2.12194440e-4f);
_PS256_CONST(cephes_log_q2, 0.693359375f);

#ifndef __AVX2__
typedef union imm_xmm_union
{
    __m256i imm;
    __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)      \
    {                                            \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
        u.imm = imm_;                            \
        xmm0_ = u.xmm[0];                        \
        xmm1_ = u.xmm[1];                        \
    }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)      \
    {                                            \
        ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
        u.xmm[0] = xmm0_;                        \
        u.xmm[1] = xmm1_;                        \
        imm_ = u.imm;                            \
    }

#define AVX2_BITOP_USING_SSE2(fn)                                      \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, int a) \
    {                                                                  \
        /* use SSE2 instruction to perform the bitop AVX2 */           \
        __m128i x1, x2;                                                \
        __m256i ret;                                                   \
        COPY_IMM_TO_XMM(x, x1, x2);                                    \
        x1 = _mm_##fn(x1, a);                                          \
        x2 = _mm_##fn(x2, a);                                          \
        COPY_XMM_TO_IMM(x1, x2, ret);                                  \
        return (ret);                                                  \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                          \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, __m256i y) \
    {                                                                      \
        /* use SSE2 instructions to perform the AVX2 integer operation */  \
        __m128i x1, x2;                                                    \
        __m128i y1, y2;                                                    \
        __m256i ret;                                                       \
        COPY_IMM_TO_XMM(x, x1, x2);                                        \
        COPY_IMM_TO_XMM(y, y1, y2);                                        \
        x1 = _mm_##fn(x1, y1);                                             \
        x2 = _mm_##fn(x2, y2);                                             \
        COPY_XMM_TO_IMM(x1, x2, ret);                                      \
        return (ret);                                                      \
    }
#else
#define AVX2_BITOP_USING_SSE2(fn)                                      \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, int a) \
    {                                                                  \
        return _mm256_##fn(x, a);                                      \
    }
#define AVX2_INTOP_USING_SSE2(fn)                                          \
    static NCNN_FORCEINLINE __m256i _mm256_comp_##fn(__m256i x, __m256i y) \
    {                                                                      \
        return _mm256_##fn(x, y);                                          \
    }
#endif

AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)

// Replace 256 bit operations with 128 bit ones when AVX2 is disabled
#ifndef __AVX2__
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
#endif

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
static NCNN_FORCEINLINE __m256 log256_ps(__m256 x)
{
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    //__m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
    __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_comp_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
    x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_comp_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    __m256 e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    //__m256 mask = _mm256_cmplt_ps(x, *(__m256*)_ps256_cephes_SQRTHF);
    __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    __m256 z = _mm256_mul_ps(x, x);

    __m256 y = *(__m256*)_ps256_cephes_log_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p5);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p6);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p7);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    y = _mm256_comp_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q1, y);

    //y = -z * 0.5 + y
    y = _mm256_comp_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);

    x = _mm256_add_ps(x, y);
    x = _mm256_comp_fmadd_ps(e, *(__m256*)_ps256_cephes_log_q2, x);
    y = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    return y;
}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

static NCNN_FORCEINLINE __m256 exp256_ps(__m256 x)
{
    __m256 tmp = _mm256_setzero_ps(), fx;
    __m256i imm0;
    __m256 one = *(__m256*)_ps256_1;

    x = _mm256_min_ps(x, *(__m256*)_ps256_exp_hi);
    x = _mm256_max_ps(x, *(__m256*)_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_comp_fmadd_ps(x, *(__m256*)_ps256_cephes_LOG2EF, *(__m256*)_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    //imm0 = _mm256_cvttps_epi32(fx);
    //tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, subtract 1 */
    //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    // x = x - fx * exp_C1
    x = _mm256_comp_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C1, x);
    // x = x - fx * exp_C2
    x = _mm256_comp_fnmadd_ps(fx, *(__m256*)_ps256_cephes_exp_C2, x);

    tmp = _mm256_mul_ps(x, x);

    __m256 y = *(__m256*)_ps256_cephes_exp_p0;
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p1);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p2);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p3);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p4);
    y = _mm256_comp_fmadd_ps(y, x, *(__m256*)_ps256_cephes_exp_p5);
    y = _mm256_comp_fmadd_ps(y, tmp, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_comp_add_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
    imm0 = _mm256_comp_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

_PS256_CONST(tanh_hi, 9.0f);
_PS256_CONST(tanh_lo, -9.0f);

_PS256_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS256_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS256_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS256_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS256_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS256_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS256_CONST(cephes_tanh_p6, 4.89352455891786E-03f);

_PS256_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS256_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS256_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

// an approximation of tanh
static inline __m256 tanh256_ps(const __m256 x)
{
    __m256 value = x;
    value = _mm256_max_ps(*(__m256*)_ps256_tanh_lo, value);
    value = _mm256_min_ps(*(__m256*)_ps256_tanh_hi, value);

    __m256 value_squared = _mm256_mul_ps(value, value);

    __m256 p;
    p = _mm256_comp_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p0, *(__m256*)_ps256_cephes_tanh_p1);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p2);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p3);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p4);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p5);
    p = _mm256_comp_fmadd_ps(p, value_squared, *(__m256*)_ps256_cephes_tanh_p6);
    p = _mm256_mul_ps(p, value);

    __m256 q;
    q = _mm256_comp_fmadd_ps(value_squared, *(__m256*)_ps256_cephes_tanh_p7, *(__m256*)_ps256_cephes_tanh_p8);
    q = _mm256_comp_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p9);
    q = _mm256_comp_fmadd_ps(q, value_squared, *(__m256*)_ps256_cephes_tanh_p6);

    __m256 dst = _mm256_div_ps(p, q);
    return dst;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625f);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS256_CONST(sincof_p0, -1.9515295891E-4f);
_PS256_CONST(sincof_p1, 8.3321608736E-3f);
_PS256_CONST(sincof_p2, -1.6666654611E-1f);
_PS256_CONST(coscof_p0, 2.443315711809948E-005f);
_PS256_CONST(coscof_p1, -1.388731625493765E-003f);
_PS256_CONST(coscof_p2, 4.166664568298827E-002f);
_PS256_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
static NCNN_FORCEINLINE __m256 sin256_ps(__m256 x)
{   // any x
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    __m256i imm0, imm2;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
#endif

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, *(__m256*)_ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);

    /*
      Here we start a series of integer operations, which are in the
      realm of AVX2.
      If we don't have AVX, let's perform them using SSE2 directives
    */

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i*)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i*)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm0_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    __m256 swap_sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m256*)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);

    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_comp_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static NCNN_FORCEINLINE __m256 cos256_ps(__m256 x)
{   // any x
    __m256 xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    __m256i imm0, imm2;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
#endif

    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_comp_sub_epi32(imm2, *(__m256i*)_pi32_256_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
#else

    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i*)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i*)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm2_1 = _mm_sub_epi32(imm2_1, *(__m128i*)_pi32avx_2);
    imm2_2 = _mm_sub_epi32(imm2_2, *(__m128i*)_pi32avx_2);

    imm0_1 = _mm_andnot_si128(imm2_1, *(__m128i*)_pi32avx_4);
    imm0_2 = _mm_andnot_si128(imm2_2, *(__m128i*)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

    __m256 sign_bit = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m256*)_ps256_coscof_p0;
    __m256 z = _mm256_mul_ps(x, x);

    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y, y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static NCNN_FORCEINLINE void sincos256_ps(__m256 x, __m256* s, __m256* c)
{
    __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    __m256i imm0, imm2, imm4;

#ifndef __AVX2__
    __m128i imm0_1, imm0_2;
    __m128i imm2_1, imm2_2;
    __m128i imm4_1, imm4_2;
#endif

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)_ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);

#ifdef __AVX2__
    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_comp_add_epi32(imm2, *(__m256i*)_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
    imm0 = _mm256_comp_slli_epi32(imm0, 29);
    //__m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
    //__m256 poly_mask = _mm256_castsi256_ps(imm2);
#else
    /* we use SSE2 routines to perform the integer ops */
    COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);

    imm2_1 = _mm_add_epi32(imm2_1, *(__m128i*)_pi32avx_1);
    imm2_2 = _mm_add_epi32(imm2_2, *(__m128i*)_pi32avx_1);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_inv1);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_inv1);

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
    y = _mm256_cvtepi32_ps(imm2);

    imm4_1 = imm2_1;
    imm4_2 = imm2_2;

    imm0_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_4);
    imm0_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_4);

    imm0_1 = _mm_slli_epi32(imm0_1, 29);
    imm0_2 = _mm_slli_epi32(imm0_2, 29);

    COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

    imm2_1 = _mm_and_si128(imm2_1, *(__m128i*)_pi32avx_2);
    imm2_2 = _mm_and_si128(imm2_2, *(__m128i*)_pi32avx_2);

    imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
    imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

    COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    __m256 poly_mask = _mm256_castsi256_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
    x = _mm256_comp_fmadd_ps(y, xmm1, x);
    x = _mm256_comp_fmadd_ps(y, xmm2, x);
    x = _mm256_comp_fmadd_ps(y, xmm3, x);

#ifdef __AVX2__
    imm4 = _mm256_comp_sub_epi32(imm4, *(__m256i*)_pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(__m256i*)_pi32_256_4);
    imm4 = _mm256_comp_slli_epi32(imm4, 29);
#else
    imm4_1 = _mm_sub_epi32(imm4_1, *(__m128i*)_pi32avx_2);
    imm4_2 = _mm_sub_epi32(imm4_2, *(__m128i*)_pi32avx_2);

    imm4_1 = _mm_andnot_si128(imm4_1, *(__m128i*)_pi32avx_4);
    imm4_2 = _mm_andnot_si128(imm4_2, *(__m128i*)_pi32avx_4);

    imm4_1 = _mm_slli_epi32(imm4_1, 29);
    imm4_2 = _mm_slli_epi32(imm4_2, 29);

    COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
#endif

    __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m256 z = _mm256_mul_ps(x, x);
    y = *(__m256*)_ps256_coscof_p0;

    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p1);
    y = _mm256_comp_fmadd_ps(y, z, *(__m256*)_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm256_comp_fnmadd_ps(z, *(__m256*)_ps256_0p5, y);
    y = _mm256_add_ps(y, *(__m256*)_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256*)_ps256_sincof_p0;
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p1);
    y2 = _mm256_comp_fmadd_ps(y2, z, *(__m256*)_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_comp_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2, ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1, ysin2);
    xmm2 = _mm256_add_ps(y, y2);

    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

static NCNN_FORCEINLINE __m256 tan256_ps(__m256 x)
{
    __m256 ysin, ycos;
    __m256 eps = _mm256_set1_ps(1E-8f);
    sincos256_ps(x, &ysin, &ycos);
    __m256 mask = _mm256_cmp_ps(ycos, _mm256_setzero_ps(), _CMP_EQ_OS);
    __m256 _tmp = _mm256_and_ps(eps, mask);
    ycos = _mm256_add_ps(ycos, _tmp);
    __m256 ytan = _mm256_div_ps(ysin, ycos);
    return ytan;
}

static NCNN_FORCEINLINE __m256 pow256_ps(__m256 a, __m256 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp256_ps(_mm256_mul_ps(b, log256_ps(a)));
}

static NCNN_FORCEINLINE __m256 asin256_ps(__m256 x)
{
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_half_one = _mm256_set1_ps(0.5f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_a4 = _mm256_set1_ps(0.023994016f);
    const __m256 magic_a5 = _mm256_set1_ps(0.042417344f);
    const __m256 magic_a2 = _mm256_set1_ps(0.07494697f);
    const __m256 magic_a3 = _mm256_set1_ps(0.045520633f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(0.166667819f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);
    const __m256 magic_three = _mm256_set1_ps(3.0f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __m256 is_small_input = _mm256_cmp_ps(absolute, magic_half_one, _CMP_LE_OQ);

    // is_big_input = (is_small_input ? 0.0f : 1.0f);
    __m256 is_big_input = _mm256_andnot_ps(is_small_input, magic_one);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m256 big_input_approx = _mm256_sqrt_ps(_mm256_mul_ps(
                                  magic_half_one,
                                  _mm256_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m256 input_approx = _mm256_or_ps(
                              _mm256_and_ps(is_small_input, absolute),
                              _mm256_andnot_ps(is_small_input, big_input_approx));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx = _mm256_mul_ps(
            square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m256 output_approx = _mm256_comp_fmadd_ps(
                               square_of_input_approx,
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a5,
                                       magic_a3),
                                   magic_a1),
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a4,
                                       magic_a2),
                                   magic_a0));

    // TODO: Need more explanations.
    // x1 = ((0.5 * PI) * is_big_input);
    // x2 = (output_approx * input_approx);
    // x3 = (-(3.0f * is_big_input) + 1.0f);
    // final_approx = ((x2 * x3) + x1);
    __m256 final_approx = _mm256_comp_fmadd_ps(
                              _mm256_mul_ps(output_approx, input_approx),
                              _mm256_comp_fnmadd_ps(magic_three, is_big_input, magic_one),
                              _mm256_mul_ps(magic_half_pi, is_big_input));

    // return (final_approx || negative_mask);
    return _mm256_or_ps(final_approx, negative_mask);
}

static NCNN_FORCEINLINE __m256 acos256_ps(__m256 x)
{
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_zero = _mm256_set1_ps(0.0f);
    const __m256 magic_half_one = _mm256_set1_ps(0.5f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_a4 = _mm256_set1_ps(0.023994016f);
    const __m256 magic_a5 = _mm256_set1_ps(0.042417344f);
    const __m256 magic_a2 = _mm256_set1_ps(0.07494697f);
    const __m256 magic_a3 = _mm256_set1_ps(0.045520633f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(0.166667819f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);
    const __m256 magic_pi = _mm256_set1_ps(3.1415927f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __m256 is_small_input = _mm256_cmp_ps(absolute, magic_half_one, _CMP_LE_OQ);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m256 big_input_approx = _mm256_sqrt_ps(_mm256_mul_ps(
                                  magic_half_one,
                                  _mm256_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m256 input_approx = _mm256_or_ps(
                              _mm256_and_ps(is_small_input, absolute),
                              _mm256_andnot_ps(is_small_input, big_input_approx));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx = _mm256_mul_ps(
            square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m256 output_approx = _mm256_comp_fmadd_ps(
                               square_of_input_approx,
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a5,
                                       magic_a3),
                                   magic_a1),
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a4,
                                       magic_a2),
                                   magic_a0));

    // TODO: Need more explanations.
    // x1 = (output_approx * input_approx);
    __m256 x1 = _mm256_mul_ps(output_approx, input_approx);

    // TODO: Need more explanations.
    // small_final_approx = ((0.5 * PI) - (x1 | negative_mask));
    __m256 small_final_approx = _mm256_sub_ps(
                                    magic_half_pi,
                                    _mm256_or_ps(x1, negative_mask));

    // TODO: Need more explanations.
    // big_final_approx = (((x < 0.0f) & PI) + ((x1 * 2) | negative_mask));
    __m256 big_final_approx = _mm256_add_ps(
                                  _mm256_and_ps(_mm256_cmp_ps(x, magic_zero, _CMP_LT_OQ), magic_pi),
                                  _mm256_or_ps(_mm256_add_ps(x1, x1), negative_mask));

    // return (is_small_input ? small_final_approx : big_final_approx);
    return _mm256_or_ps(
               _mm256_and_ps(is_small_input, small_final_approx),
               _mm256_andnot_ps(is_small_input, big_final_approx));
}

static NCNN_FORCEINLINE __m256 atan256_ps(__m256 x)
{
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_one = _mm256_set1_ps(1.0f);
    const __m256 magic_negative_one = _mm256_set1_ps(-1.0f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);
    const __m256 magic_a0 = _mm256_set1_ps(1.0f);
    const __m256 magic_a1 = _mm256_set1_ps(-0.33333072f);
    const __m256 magic_a2 = _mm256_set1_ps(0.1999262f);
    const __m256 magic_a3 = _mm256_set1_ps(-0.14203644f);
    const __m256 magic_a4 = _mm256_set1_ps(0.10640934f);
    const __m256 magic_a5 = _mm256_set1_ps(-0.07504295f);
    const __m256 magic_a6 = _mm256_set1_ps(0.04269152f);
    const __m256 magic_a7 = _mm256_set1_ps(-0.01606863f);
    const __m256 magic_a8 = _mm256_set1_ps(0.0028498897f);

    // negative_mask = magic_negative_zero && x;
    __m256 negative_mask = _mm256_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m256 absolute = _mm256_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (1.0f < absolute);
    __m256 is_small_input = _mm256_cmp_ps(magic_one, absolute, _CMP_LT_OQ);

    // x1 = (is_small_input ? -1.0f : absolute);
    // x2 = (is_small_input ? absolute : 1.0f)
    // input_approx = x1 / x2;
    __m256 input_approx = _mm256_div_ps(
                              _mm256_or_ps(
                                  _mm256_and_ps(is_small_input, magic_negative_one),
                                  _mm256_andnot_ps(is_small_input, absolute)),
                              _mm256_or_ps(
                                  _mm256_and_ps(is_small_input, absolute),
                                  _mm256_andnot_ps(is_small_input, magic_one)));

    // square_of_input_approx = input_approx * input_approx;
    __m256 square_of_input_approx = _mm256_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m256 fourth_power_of_input_approx = _mm256_mul_ps(
            square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a7) + magic_a5);
    // x2 = ((fourth_power_of_input_approx * magic_a8) + magic_a6);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a3);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a4);
    // x5 = ((fourth_power_of_input_approx * x3) + magic_a1);
    // x6 = ((fourth_power_of_input_approx * x4) + magic_a2);
    // x7 = ((fourth_power_of_input_approx * x6) + magic_a0);
    // output_approx = ((square_of_input_approx * x5) + x7);
    __m256 output_approx = _mm256_comp_fmadd_ps(
                               square_of_input_approx,
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       _mm256_comp_fmadd_ps(
                                           fourth_power_of_input_approx,
                                           magic_a7,
                                           magic_a5),
                                       magic_a3),
                                   magic_a1),
                               _mm256_comp_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm256_comp_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       _mm256_comp_fmadd_ps(
                                           fourth_power_of_input_approx,
                                           _mm256_comp_fmadd_ps(
                                                   fourth_power_of_input_approx,
                                                   magic_a8,
                                                   magic_a6),
                                           magic_a4),
                                       magic_a2),
                                   magic_a0));

    // TODO: Need more explanations.
    // x1 = (output_approx * input_approx);
    // if (is_small_input) x1 += (0.5 * PI);
    // return (negative_mask ? -x1 : x1);
    return _mm256_or_ps(
               _mm256_add_ps(
                   _mm256_mul_ps(output_approx, input_approx),
                   _mm256_and_ps(is_small_input, magic_half_pi)),
               negative_mask);
}

static NCNN_FORCEINLINE __m256 atan2256_ps(__m256 y, __m256 x)
{
    // Reference: https://mazzo.li/posts/vectorized-atan2.html

    const __m256 magic_zero = _mm256_set1_ps(0.0f);
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);
    const __m256 magic_pi = _mm256_set1_ps(3.1415927f);
    const __m256 magic_half_pi = _mm256_set1_ps(1.5707964f);

    // not_equal_zero_x = (x != 0.0f);
    __m256 not_equal_zero_x = _mm256_cmp_ps(x, magic_zero, _CMP_NEQ_OQ);

    // not_equal_zero_y = (y != 0.0f);
    __m256 not_equal_zero_y = _mm256_cmp_ps(y, magic_zero, _CMP_NEQ_OQ);

    // normal_mode = ((x != 0.0f) & (y != 0.0f));
    __m256 normal_mode = _mm256_and_ps(not_equal_zero_x, not_equal_zero_y);

    // negative_mask_x = magic_negative_zero && x;
    __m256 negative_mask_x = _mm256_and_ps(magic_negative_zero, x);

    // negative_mask_y = magic_negative_zero && y;
    __m256 negative_mask_y = _mm256_and_ps(magic_negative_zero, y);

    // pi_additions = ((x < 0.0f) ? ((y < 0.0f) ? -PI : PI) : 0.0f);
    __m256 pi_additions = _mm256_and_ps(
                              _mm256_cmp_ps(x, magic_zero, _CMP_LT_OQ),
                              _mm256_or_ps(
                                  _mm256_and_ps(
                                      _mm256_cmp_ps(y, magic_zero, _CMP_LT_OQ),
                                      magic_negative_zero),
                                  magic_pi));

    // normal_result = (atan(y / x) + pi_additions);
    __m256 normal_result = _mm256_add_ps(
                               atan256_ps(_mm256_div_ps(y, x)),
                               pi_additions);

    // negative_mask_full_x = ((negative_mask_x | PI) < 0.0f);
    __m256 negative_mask_full_x = _mm256_cmp_ps(
                                      _mm256_or_ps(negative_mask_x, magic_pi),
                                      magic_zero,
                                      _CMP_LT_OQ);

    // x1 = (negative_mask_y ? -(0.5 * PI) : (0.5 * PI));
    // x2 = (negative_mask_full_x ? PI : 0.0f);
    // special_result = ((y != 0.0f) ? x1 : x2);
    __m256 special_result = _mm256_or_ps(
                                _mm256_and_ps(
                                    not_equal_zero_y,
                                    _mm256_or_ps(negative_mask_y, magic_half_pi)),
                                _mm256_andnot_ps(
                                    not_equal_zero_y,
                                    _mm256_or_ps(
                                        _mm256_and_ps(negative_mask_full_x, magic_pi),
                                        _mm256_andnot_ps(negative_mask_full_x, magic_zero))));

    // return (normal_mode ? normal_result : special_result);
    return _mm256_or_ps(
               _mm256_and_ps(normal_mode, normal_result),
               _mm256_andnot_ps(normal_mode, special_result));
}

static NCNN_FORCEINLINE __m256 abs256_ps(__m256 x)
{
    // Use negative zero as the sign bit mask.
    const __m256 magic_negative_zero = _mm256_set1_ps(-0.0f);

    // return (!magic_negative_zero && x);
    return _mm256_andnot_ps(magic_negative_zero, x);
}

#endif // AVX_MATHFUN_H
