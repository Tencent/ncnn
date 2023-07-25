// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef AVX512_MATHFUN_H
#define AVX512_MATHFUN_H

#include <immintrin.h>
#include <emmintrin.h>

/* yes I know, the top of this file is quite ugly */

#ifdef _MSC_VER /* visual c++ */
#define ALIGN64_BEG __declspec(align(64))
#define ALIGN64_END
#else /* gcc or icc */
#define ALIGN64_BEG
#define ALIGN64_END __attribute__((aligned(64)))
#endif

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS512_CONST(Name, Val) \
    static const ALIGN64_BEG float _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST512(Name, Val) \
    static const ALIGN64_BEG int _pi32_512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS512_CONST_TYPE(Name, Type, Val) \
    static const ALIGN64_BEG Type _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}

_PS512_CONST(1, 1.0f);
_PS512_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS512_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS512_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS512_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS512_CONST_TYPE(sign_mask, int, (int)0x80000000);

_PI32_CONST512(0, 0);
_PI32_CONST512(1, 1);
_PI32_CONST512(inv1, ~1);
_PI32_CONST512(2, 2);
_PI32_CONST512(4, 4);
_PI32_CONST512(0x7f, 0x7f);

_PS512_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS512_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS512_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS512_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS512_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS512_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS512_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS512_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS512_CONST(cephes_log_q1, -2.12194440e-4f);
_PS512_CONST(cephes_log_q2, 0.693359375f);

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
static NCNN_FORCEINLINE __m512 log512_ps(__m512 x)
{
    __m512i imm0;
    __m512 one = *(__m512*)_ps512_1;

    __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);

    x = _mm512_max_ps(x, *(__m512*)_ps512_min_norm_pos); /* cut off denormalized stuff */

    imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

    /* keep only the fractional part */
    x = _mm512_and_ps(x, *(__m512*)_ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(__m512*)_ps512_0p5);

    imm0 = _mm512_sub_epi32(imm0, *(__m512i*)_pi32_512_0x7f);
    __m512 e = _mm512_cvtepi32_ps(imm0);

    e = _mm512_add_ps(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    __mmask16 mask = _mm512_cmp_ps_mask(x, *(__m512*)_ps512_cephes_SQRTHF, _CMP_LT_OQ);
    __m512 tmp = _mm512_sub_ps(x, one);
    e = _mm512_mask_sub_ps(e, mask, e, one);
    x = _mm512_mask_add_ps(tmp, mask, tmp, x);

    __m512 z = _mm512_mul_ps(x, x);

    __m512 y = *(__m512*)_ps512_cephes_log_p0;
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p1);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p2);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p3);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p4);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p5);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p6);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p7);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_log_p8);
    y = _mm512_mul_ps(y, x);

    y = _mm512_mul_ps(y, z);

    y = _mm512_fmadd_ps(e, *(__m512*)_ps512_cephes_log_q1, y);

    //y = -z * 0.5 + y
    y = _mm512_fnmadd_ps(z, *(__m512*)_ps512_0p5, y);

    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, *(__m512*)_ps512_cephes_log_q2, x);
    y = _mm512_or_ps(x, _mm512_castsi512_ps(_mm512_movm_epi32(invalid_mask))); // negative arg will be NAN
    return y;
}

_PS512_CONST(exp_hi, 88.3762626647949f);
_PS512_CONST(exp_lo, -88.3762626647949f);

_PS512_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS512_CONST(cephes_exp_C1, 0.693359375f);
_PS512_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS512_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1f);

static NCNN_FORCEINLINE __m512 exp512_ps(__m512 x)
{
    __m512 tmp = _mm512_setzero_ps(), fx;
    __m512i imm0;
    __m512 one = *(__m512*)_ps512_1;

    x = _mm512_min_ps(x, *(__m512*)_ps512_exp_hi);
    x = _mm512_max_ps(x, *(__m512*)_ps512_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm512_fmadd_ps(x, *(__m512*)_ps512_cephes_LOG2EF, *(__m512*)_ps512_0p5);

    tmp = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF);

    /* if greater, subtract 1 */
    __mmask16 mask = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OQ);
    fx = _mm512_mask_sub_ps(tmp, mask, tmp, one);

    // x = x - fx * exp_C1
    x = _mm512_fnmadd_ps(fx, *(__m512*)_ps512_cephes_exp_C1, x);
    // x = x - fx * exp_C2
    x = _mm512_fnmadd_ps(fx, *(__m512*)_ps512_cephes_exp_C2, x);

    tmp = _mm512_mul_ps(x, x);

    __m512 y = *(__m512*)_ps512_cephes_exp_p0;
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p1);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p2);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p3);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p4);
    y = _mm512_fmadd_ps(y, x, *(__m512*)_ps512_cephes_exp_p5);
    y = _mm512_fmadd_ps(y, tmp, x);
    y = _mm512_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm512_cvttps_epi32(fx);
    imm0 = _mm512_add_epi32(imm0, *(__m512i*)_pi32_512_0x7f);
    imm0 = _mm512_slli_epi32(imm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(imm0);
    y = _mm512_mul_ps(y, pow2n);
    return y;
}

_PS512_CONST(tanh_hi, 9.0f);
_PS512_CONST(tanh_lo, -9.0f);

_PS512_CONST(cephes_tanh_p0, -2.76076847742355E-16f);
_PS512_CONST(cephes_tanh_p1, 2.00018790482477E-13f);
_PS512_CONST(cephes_tanh_p2, -8.60467152213735E-11f);
_PS512_CONST(cephes_tanh_p3, 5.12229709037114E-08f);
_PS512_CONST(cephes_tanh_p4, 1.48572235717979E-05f);
_PS512_CONST(cephes_tanh_p5, 6.37261928875436E-04f);
_PS512_CONST(cephes_tanh_p6, 4.89352455891786E-03f);

_PS512_CONST(cephes_tanh_p7, 1.19825839466702e-06f);
_PS512_CONST(cephes_tanh_p8, 1.18534705686654e-04f);
_PS512_CONST(cephes_tanh_p9, 2.26843463243900e-03f);

// an approximation of tanh
static inline __m512 tanh512_ps(const __m512 x)
{
    __m512 value = x;
    value = _mm512_max_ps(*(__m512*)_ps512_tanh_lo, value);
    value = _mm512_min_ps(*(__m512*)_ps512_tanh_hi, value);

    __m512 value_squared = _mm512_mul_ps(value, value);

    __m512 p;
    p = _mm512_fmadd_ps(value_squared, *(__m512*)_ps512_cephes_tanh_p0, *(__m512*)_ps512_cephes_tanh_p1);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p2);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p3);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p4);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p5);
    p = _mm512_fmadd_ps(p, value_squared, *(__m512*)_ps512_cephes_tanh_p6);
    p = _mm512_mul_ps(p, value);

    __m512 q;
    q = _mm512_fmadd_ps(value_squared, *(__m512*)_ps512_cephes_tanh_p7, *(__m512*)_ps512_cephes_tanh_p8);
    q = _mm512_fmadd_ps(q, value_squared, *(__m512*)_ps512_cephes_tanh_p9);
    q = _mm512_fmadd_ps(q, value_squared, *(__m512*)_ps512_cephes_tanh_p6);

    __m512 dst = _mm512_div_ps(p, q);
    return dst;
}

_PS512_CONST(minus_cephes_DP1, -0.78515625f);
_PS512_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS512_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS512_CONST(sincof_p0, -1.9515295891E-4f);
_PS512_CONST(sincof_p1, 8.3321608736E-3f);
_PS512_CONST(sincof_p2, -1.6666654611E-1f);
_PS512_CONST(coscof_p0, 2.443315711809948E-005f);
_PS512_CONST(coscof_p1, -1.388731625493765E-003f);
_PS512_CONST(coscof_p2, 4.166664568298827E-002f);
_PS512_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
static NCNN_FORCEINLINE __m512 sin512_ps(__m512 x)
{   // any x
    __m512 xmm1, xmm2, xmm3, sign_bit, y;
    __m512i imm0, imm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
    /* extract the sign bit (upper one) */
    sign_bit = _mm512_and_ps(sign_bit, *(__m512*)_ps512_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(__m512*)_ps512_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = _mm512_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm512_add_epi32(imm2, *(__m512i*)_pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_inv1);
    y = _mm512_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_2);
    imm2 = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(imm2, *(__m512i*)_pi32_512_0));

    __m512 swap_sign_bit = _mm512_castsi512_ps(imm0);
    __m512 poly_mask = _mm512_castsi512_ps(imm2);
    sign_bit = _mm512_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m512*)_ps512_minus_cephes_DP1;
    xmm2 = *(__m512*)_ps512_minus_cephes_DP2;
    xmm3 = *(__m512*)_ps512_minus_cephes_DP3;
    x = _mm512_fmadd_ps(y, xmm1, x);
    x = _mm512_fmadd_ps(y, xmm2, x);
    x = _mm512_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m512*)_ps512_coscof_p0;
    __m512 z = _mm512_mul_ps(x, x);

    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm512_fnmadd_ps(z, *(__m512*)_ps512_0p5, y);
    y = _mm512_add_ps(y, *(__m512*)_ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m512 y2 = *(__m512*)_ps512_sincof_p0;
    y2 = _mm512_fmadd_ps(y2, z, *(__m512*)_ps512_sincof_p1);
    y2 = _mm512_fmadd_ps(y2, z, *(__m512*)_ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm512_and_ps(xmm3, y2); //, xmm3);
    y = _mm512_andnot_ps(xmm3, y);
    y = _mm512_add_ps(y, y2);
    /* update the sign */
    y = _mm512_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static NCNN_FORCEINLINE __m512 cos512_ps(__m512 x)
{   // any x
    __m512 xmm1, xmm2, xmm3, y;
    __m512i imm0, imm2;

    /* take the absolute value */
    x = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(__m512*)_ps512_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = _mm512_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm512_add_epi32(imm2, *(__m512i*)_pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_inv1);
    y = _mm512_cvtepi32_ps(imm2);
    imm2 = _mm512_sub_epi32(imm2, *(__m512i*)_pi32_512_2);

    /* get the swap sign flag */
    imm0 = _mm512_andnot_si512(imm2, *(__m512i*)_pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_2);
    imm2 = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(imm2, *(__m512i*)_pi32_512_0));

    __m512 sign_bit = _mm512_castsi512_ps(imm0);
    __m512 poly_mask = _mm512_castsi512_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m512*)_ps512_minus_cephes_DP1;
    xmm2 = *(__m512*)_ps512_minus_cephes_DP2;
    xmm3 = *(__m512*)_ps512_minus_cephes_DP3;
    x = _mm512_fmadd_ps(y, xmm1, x);
    x = _mm512_fmadd_ps(y, xmm2, x);
    x = _mm512_fmadd_ps(y, xmm3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(__m512*)_ps512_coscof_p0;
    __m512 z = _mm512_mul_ps(x, x);

    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm512_fnmadd_ps(z, *(__m512*)_ps512_0p5, y);
    y = _mm512_add_ps(y, *(__m512*)_ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m512 y2 = *(__m512*)_ps512_sincof_p0;
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_add_ps(y2, *(__m512*)_ps512_sincof_p1);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_add_ps(y2, *(__m512*)_ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_mul_ps(y2, x);
    y2 = _mm512_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm512_and_ps(xmm3, y2); //, xmm3);
    y = _mm512_andnot_ps(xmm3, y);
    y = _mm512_add_ps(y, y2);
    /* update the sign */
    y = _mm512_xor_ps(y, sign_bit);

    return y;
}

/* since sin512_ps and cos512_ps are almost identical, sincos512_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
static NCNN_FORCEINLINE void sincos512_ps(__m512 x, __m512* s, __m512* c)
{
    __m512 xmm1, xmm2, xmm3, sign_bit_sin, y;
    __m512i imm0, imm2, imm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm512_and_ps(sign_bit_sin, *(__m512*)_ps512_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(__m512*)_ps512_cephes_FOPI);

    /* store the integer part of y in imm2 */
    imm2 = _mm512_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm512_add_epi32(imm2, *(__m512i*)_pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_inv1);

    y = _mm512_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    //__m512 swap_sign_bit_sin = _mm512_castsi512_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm512_and_si512(imm2, *(__m512i*)_pi32_512_2);
    imm2 = _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(imm2, *(__m512i*)_pi32_512_0));
    //__m512 poly_mask = _mm512_castsi512_ps(imm2);

    __m512 swap_sign_bit_sin = _mm512_castsi512_ps(imm0);
    __m512 poly_mask = _mm512_castsi512_ps(imm2);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m512*)_ps512_minus_cephes_DP1;
    xmm2 = *(__m512*)_ps512_minus_cephes_DP2;
    xmm3 = *(__m512*)_ps512_minus_cephes_DP3;
    x = _mm512_fmadd_ps(y, xmm1, x);
    x = _mm512_fmadd_ps(y, xmm2, x);
    x = _mm512_fmadd_ps(y, xmm3, x);

    imm4 = _mm512_sub_epi32(imm4, *(__m512i*)_pi32_512_2);
    imm4 = _mm512_andnot_si512(imm4, *(__m512i*)_pi32_512_4);
    imm4 = _mm512_slli_epi32(imm4, 29);

    __m512 sign_bit_cos = _mm512_castsi512_ps(imm4);

    sign_bit_sin = _mm512_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m512 z = _mm512_mul_ps(x, x);
    y = *(__m512*)_ps512_coscof_p0;

    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(__m512*)_ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    // y = y - z * 0.5
    y = _mm512_fnmadd_ps(z, *(__m512*)_ps512_0p5, y);
    y = _mm512_add_ps(y, *(__m512*)_ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m512 y2 = *(__m512*)_ps512_sincof_p0;
    y2 = _mm512_fmadd_ps(y2, z, *(__m512*)_ps512_sincof_p1);
    y2 = _mm512_fmadd_ps(y2, z, *(__m512*)_ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    __m512 ysin2 = _mm512_and_ps(xmm3, y2);
    __m512 ysin1 = _mm512_andnot_ps(xmm3, y);
    y2 = _mm512_sub_ps(y2, ysin2);
    y = _mm512_sub_ps(y, ysin1);

    xmm1 = _mm512_add_ps(ysin1, ysin2);
    xmm2 = _mm512_add_ps(y, y2);

    /* update the sign */
    *s = _mm512_xor_ps(xmm1, sign_bit_sin);
    *c = _mm512_xor_ps(xmm2, sign_bit_cos);
}

static NCNN_FORCEINLINE __m512 tan512_ps(__m512 x)
{
    __m512 ysin, ycos;
    __m512 eps = _mm512_set1_ps(1E-8f);
    sincos512_ps(x, &ysin, &ycos);
    __mmask16 mask = _mm512_cmp_ps_mask(ycos, _mm512_setzero_ps(), _CMP_EQ_OS);
    ycos = _mm512_mask_add_ps(ycos, mask, ycos, eps);
    __m512 ytan = _mm512_div_ps(ysin, ycos);
    return ytan;
}

static NCNN_FORCEINLINE __m512 pow512_ps(__m512 a, __m512 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp512_ps(_mm512_mul_ps(b, log512_ps(a)));
}

static NCNN_FORCEINLINE __m512 asin512_ps(__m512 x)
{
    const __m512 magic_negative_zero = _mm512_set1_ps(-0.0f);
    const __m512 magic_half_one = _mm512_set1_ps(0.5f);
    const __m512 magic_one = _mm512_set1_ps(1.0f);
    const __m512 magic_a4 = _mm512_set1_ps(0.023994016f);
    const __m512 magic_a5 = _mm512_set1_ps(0.042417344f);
    const __m512 magic_a2 = _mm512_set1_ps(0.07494697f);
    const __m512 magic_a3 = _mm512_set1_ps(0.045520633f);
    const __m512 magic_a0 = _mm512_set1_ps(1.0f);
    const __m512 magic_a1 = _mm512_set1_ps(0.166667819f);
    const __m512 magic_half_pi = _mm512_set1_ps(1.5707964f);
    const __m512 magic_three = _mm512_set1_ps(3.0f);

    // negative_mask = magic_negative_zero && x;
    __m512 negative_mask = _mm512_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m512 absolute = _mm512_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __mmask16 is_small_input = _mm512_cmp_ps_mask(
                                   absolute,
                                   magic_half_one,
                                   _CMP_LE_OQ);

    // is_big_input = (is_small_input ? 0.0f : 1.0f);
    __m512 is_big_input = _mm512_maskz_mov_ps(~is_small_input, magic_one);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m512 big_input_approx = _mm512_sqrt_ps(_mm512_mul_ps(
                                  magic_half_one,
                                  _mm512_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m512 input_approx = _mm512_mask_mov_ps(
                              big_input_approx,
                              is_small_input,
                              absolute);

    // square_of_input_approx = input_approx * input_approx;
    __m512 square_of_input_approx = _mm512_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m512 fourth_power_of_input_approx = _mm512_mul_ps(
            square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m512 output_approx = _mm512_fmadd_ps(
                               square_of_input_approx,
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a5,
                                       magic_a3),
                                   magic_a1),
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a4,
                                       magic_a2),
                                   magic_a0));

    // TODO: Need more explanations.
    // x1 = ((0.5 * PI) * is_big_input);
    // x2 = (output_approx * input_approx);
    // x3 = (-(3.0f * is_big_input) + 1.0f);
    // final_approx = ((x2 * x3) + x1);
    __m512 final_approx = _mm512_fmadd_ps(
                              _mm512_mul_ps(output_approx, input_approx),
                              _mm512_fnmadd_ps(magic_three, is_big_input, magic_one),
                              _mm512_mul_ps(magic_half_pi, is_big_input));

    // return (final_approx || negative_mask);
    return _mm512_or_ps(final_approx, negative_mask);
}

static NCNN_FORCEINLINE __m512 acos512_ps(__m512 x)
{
    const __m512 magic_negative_zero = _mm512_set1_ps(-0.0f);
    const __m512 magic_zero = _mm512_set1_ps(0.0f);
    const __m512 magic_half_one = _mm512_set1_ps(0.5f);
    const __m512 magic_one = _mm512_set1_ps(1.0f);
    const __m512 magic_a4 = _mm512_set1_ps(0.023994016f);
    const __m512 magic_a5 = _mm512_set1_ps(0.042417344f);
    const __m512 magic_a2 = _mm512_set1_ps(0.07494697f);
    const __m512 magic_a3 = _mm512_set1_ps(0.045520633f);
    const __m512 magic_a0 = _mm512_set1_ps(1.0f);
    const __m512 magic_a1 = _mm512_set1_ps(0.166667819f);
    const __m512 magic_half_pi = _mm512_set1_ps(1.5707964f);
    const __m512 magic_pi = _mm512_set1_ps(3.1415927f);

    // negative_mask = magic_negative_zero && x;
    __m512 negative_mask = _mm512_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m512 absolute = _mm512_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (absolute <= 0.5f);
    __mmask16 is_small_input = _mm512_cmp_ps_mask(
                                   absolute,
                                   magic_half_one,
                                   _CMP_LE_OQ);

    // big_input_approx = sqrt(0.5f * (1 - absolute));
    __m512 big_input_approx = _mm512_sqrt_ps(_mm512_mul_ps(
                                  magic_half_one,
                                  _mm512_sub_ps(magic_one, absolute)));

    // input_approx = (is_small_input ? absolute : big_input_approx);
    __m512 input_approx = _mm512_mask_mov_ps(
                              big_input_approx,
                              is_small_input,
                              absolute);

    // square_of_input_approx = input_approx * input_approx;
    __m512 square_of_input_approx = _mm512_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m512 fourth_power_of_input_approx = _mm512_mul_ps(
            square_of_input_approx, square_of_input_approx);

    // TODO: Need more explanations.
    // x1 = ((fourth_power_of_input_approx * magic_a4) + magic_a2);
    // x2 = ((fourth_power_of_input_approx * magic_a5) + magic_a3);
    // x3 = ((fourth_power_of_input_approx * x1) + magic_a0);
    // x4 = ((fourth_power_of_input_approx * x2) + magic_a1);
    // output_approx = ((square_of_input_approx * x4) + x3);
    __m512 output_approx = _mm512_fmadd_ps(
                               square_of_input_approx,
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a5,
                                       magic_a3),
                                   magic_a1),
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       magic_a4,
                                       magic_a2),
                                   magic_a0));

    // TODO: Need more explanations.
    // x1 = (output_approx * input_approx);
    __m512 x1 = _mm512_mul_ps(output_approx, input_approx);

    // TODO: Need more explanations.
    // small_final_approx = ((0.5 * PI) - (x1 | negative_mask));
    __m512 small_final_approx = _mm512_sub_ps(
                                    magic_half_pi,
                                    _mm512_or_ps(x1, negative_mask));

    // TODO: Need more explanations.
    // big_final_approx = (((x < 0.0f) & PI) + ((x1 * 2) | negative_mask));
    __m512 big_final_approx = _mm512_add_ps(
                                  _mm512_maskz_mov_ps(
                                      _mm512_cmp_ps_mask(x, magic_zero, _CMP_LT_OQ),
                                      magic_pi),
                                  _mm512_or_ps(_mm512_add_ps(x1, x1), negative_mask));

    // return (is_small_input ? small_final_approx : big_final_approx);
    return _mm512_mask_mov_ps(
               big_final_approx,
               is_small_input,
               small_final_approx);
}

static NCNN_FORCEINLINE __m512 atan512_ps(__m512 x)
{
    const __m512 magic_negative_zero = _mm512_set1_ps(-0.0f);
    const __m512 magic_one = _mm512_set1_ps(1.0f);
    const __m512 magic_negative_one = _mm512_set1_ps(-1.0f);
    const __m512 magic_half_pi = _mm512_set1_ps(1.5707964f);
    const __m512 magic_a0 = _mm512_set1_ps(1.0f);
    const __m512 magic_a1 = _mm512_set1_ps(-0.33333072f);
    const __m512 magic_a2 = _mm512_set1_ps(0.1999262f);
    const __m512 magic_a3 = _mm512_set1_ps(-0.14203644f);
    const __m512 magic_a4 = _mm512_set1_ps(0.10640934f);
    const __m512 magic_a5 = _mm512_set1_ps(-0.07504295f);
    const __m512 magic_a6 = _mm512_set1_ps(0.04269152f);
    const __m512 magic_a7 = _mm512_set1_ps(-0.01606863f);
    const __m512 magic_a8 = _mm512_set1_ps(0.0028498897f);

    // negative_mask = magic_negative_zero && x;
    __m512 negative_mask = _mm512_and_ps(magic_negative_zero, x);

    // absolute = abs(x);
    __m512 absolute = _mm512_andnot_ps(magic_negative_zero, x);

    // Reference: https://en.wikipedia.org/wiki/Small-angle_approximation

    // is_small_input = (1.0f < absolute);
    __mmask16 is_small_input = _mm512_cmp_ps_mask(
                                   magic_one,
                                   absolute,
                                   _CMP_LT_OQ);

    // x1 = (is_small_input ? -1.0f : absolute);
    // x2 = (is_small_input ? absolute : 1.0f)
    // input_approx = x1 / x2;
    __m512 input_approx = _mm512_div_ps(
                              _mm512_mask_mov_ps(absolute, is_small_input, magic_negative_one),
                              _mm512_mask_mov_ps(magic_one, is_small_input, absolute));

    // square_of_input_approx = input_approx * input_approx;
    __m512 square_of_input_approx = _mm512_mul_ps(input_approx, input_approx);

    // fourth_power_of_input_approx =
    //     square_of_input_approx * square_of_input_approx;
    __m512 fourth_power_of_input_approx = _mm512_mul_ps(
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
    __m512 output_approx = _mm512_fmadd_ps(
                               square_of_input_approx,
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       _mm512_fmadd_ps(
                                           fourth_power_of_input_approx,
                                           magic_a7,
                                           magic_a5),
                                       magic_a3),
                                   magic_a1),
                               _mm512_fmadd_ps(
                                   fourth_power_of_input_approx,
                                   _mm512_fmadd_ps(
                                       fourth_power_of_input_approx,
                                       _mm512_fmadd_ps(
                                           fourth_power_of_input_approx,
                                           _mm512_fmadd_ps(
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
    return _mm512_or_ps(
               _mm512_add_ps(
                   _mm512_mul_ps(output_approx, input_approx),
                   _mm512_maskz_mov_ps(is_small_input, magic_half_pi)),
               negative_mask);
}

// MSVC 2017 x86 CI will be broken if use NCNN_FORCEINLINE for atan2512_ps.
// This function still be inlined compiled by MSVC 2017 even without that.
#if _MSC_VER < 1920
static __m512 atan2512_ps(__m512 y, __m512 x)
#else
static NCNN_FORCEINLINE __m512 atan2512_ps(__m512 y, __m512 x)
#endif
{
    // Reference: https://mazzo.li/posts/vectorized-atan2.html

    const __m512 magic_zero = _mm512_set1_ps(0.0f);
    const __m512 magic_negative_zero = _mm512_set1_ps(-0.0f);
    const __m512 magic_pi = _mm512_set1_ps(3.1415927f);
    const __m512 magic_half_pi = _mm512_set1_ps(1.5707964f);

    // not_equal_zero_x = (x != 0.0f);
    __mmask16 not_equal_zero_x = _mm512_cmp_ps_mask(
                                     x,
                                     magic_zero,
                                     _CMP_NEQ_OQ);

    // not_equal_zero_y = (y != 0.0f);
    __mmask16 not_equal_zero_y = _mm512_cmp_ps_mask(
                                     y,
                                     magic_zero,
                                     _CMP_NEQ_OQ);

    // normal_mode = ((x != 0.0f) & (y != 0.0f));
    __mmask16 normal_mode = (not_equal_zero_x & not_equal_zero_y);

    // negative_mask_x = magic_negative_zero && x;
    __m512 negative_mask_x = _mm512_and_ps(magic_negative_zero, x);

    // negative_mask_y = magic_negative_zero && y;
    __m512 negative_mask_y = _mm512_and_ps(magic_negative_zero, y);

    // pi_additions = ((x < 0.0f) ? ((y < 0.0f) ? -PI : PI) : 0.0f);
    __m512 pi_additions = _mm512_mask_mov_ps(
                              magic_zero,
                              _mm512_cmp_ps_mask(x, magic_zero, _CMP_LT_OQ),
                              _mm512_mask_mov_ps(
                                  magic_pi,
                                  _mm512_cmp_ps_mask(y, magic_zero, _CMP_LT_OQ),
                                  _mm512_or_ps(magic_negative_zero, magic_pi)));

    // normal_result = (atan(y / x) + pi_additions);
    __m512 normal_result = _mm512_add_ps(
                               atan512_ps(_mm512_div_ps(y, x)),
                               pi_additions);

    // negative_mask_full_x = ((negative_mask_x | PI) < 0.0f);
    __mmask16 negative_mask_full_x = _mm512_cmp_ps_mask(
                                         _mm512_or_ps(negative_mask_x, magic_pi),
                                         magic_zero,
                                         _CMP_LT_OQ);

    // x1 = (negative_mask_y ? -(0.5 * PI) : (0.5 * PI));
    // x2 = (negative_mask_full_x ? PI : 0.0f);
    // special_result = ((y != 0.0f) ? x1 : x2);
    __m512 special_result = _mm512_mask_mov_ps(
                                _mm512_mask_mov_ps(magic_zero, negative_mask_full_x, magic_pi),
                                not_equal_zero_y,
                                _mm512_or_ps(negative_mask_y, magic_half_pi));

    // return (normal_mode ? normal_result : special_result);
    return _mm512_mask_mov_ps(special_result, normal_mode, normal_result);
}

static NCNN_FORCEINLINE __m512 abs512_ps(__m512 x)
{
    // Use negative zero as the sign bit mask.
    const __m512 magic_negative_zero = _mm512_set1_ps(-0.0f);

    // return (!magic_negative_zero && x);
    return _mm512_andnot_ps(magic_negative_zero, x);
}

#endif // AVX512_MATHFUN_H
