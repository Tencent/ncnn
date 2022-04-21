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

static NCNN_FORCEINLINE __m512 pow512_ps(__m512 a, __m512 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp512_ps(_mm512_mul_ps(b, log512_ps(a)));
}

#endif // AVX512_MATHFUN_H
