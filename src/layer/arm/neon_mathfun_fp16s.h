// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

/* NEON implementation of sin, cos, exp and log
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 */

/* Copyright (C) 2011  Julien Pommier
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty.  In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original software.
 *  3. This notice may not be removed or altered from any source distribution.
 *
 *  (this is the zlib license)
 */

#ifndef NEON_MATHFUN_FP16S_H
#define NEON_MATHFUN_FP16S_H

#include <arm_neon.h>

#define c_inv_mant_mask_f16 -31745 // ~0x7c00u
#define c_cephes_SQRTHF     0.707106781186547524
#define c_cephes_log_p0     7.0376836292E-2
#define c_cephes_log_p1     -1.1514610310E-1
#define c_cephes_log_p2     1.1676998740E-1
#define c_cephes_log_p3     -1.2420140846E-1
#define c_cephes_log_p4     +1.4249322787E-1
#define c_cephes_log_p5     -1.6668057665E-1
#define c_cephes_log_p6     +2.0000714765E-1
#define c_cephes_log_p7     -2.4999993993E-1
#define c_cephes_log_p8     +3.3333331174E-1
#define c_cephes_log_q1     -2.12194440e-4
#define c_cephes_log_q2     0.693359375

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline float16x4_t log_ps_f16(float16x4_t x)
{
    float16x4_t one = vdup_n_f16(1);

    x = vmax_f16(x, vdup_n_f16(0)); /* force flush to zero on denormal values */
    uint16x4_t invalid_mask = vcle_f16(x, vdup_n_f16(0));

    int16x4_t ux = vreinterpret_s16_f16(x);

    int16x4_t emm0 = vshr_n_s16(ux, 10);

    /* keep only the fractional part */
    ux = vand_s16(ux, vdup_n_s16(c_inv_mant_mask_f16));
    ux = vorr_s16(ux, vreinterpret_s16_f16(vdup_n_f16(0.5f)));
    x = vreinterpret_f16_s16(ux);

    emm0 = vsub_s16(emm0, vdup_n_s16(0xf));
    float16x4_t e = vcvt_f16_s16(emm0);

    e = vadd_f16(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    uint16x4_t mask = vclt_f16(x, vdup_n_f16(c_cephes_SQRTHF));
    float16x4_t tmp = (float16x4_t)(vand_u16((uint16x4_t)(x), mask));
    x = vsub_f16(x, one);
    e = vsub_f16(e, (float16x4_t)(vand_u16((uint16x4_t)(one), mask)));
    x = vadd_f16(x, tmp);

    float16x4_t z = vmul_f16(x, x);

    float16x4_t y = vdup_n_f16(c_cephes_log_p0);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p1), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p2), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p3), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p4), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p5), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p6), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p7), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_log_p8), y, x);
    y = vmul_f16(y, x);

    y = vmul_f16(y, z);

    y = vfma_f16(y, e, vdup_n_f16(c_cephes_log_q1));

    y = vfms_f16(y, z, vdup_n_f16(0.5f));

    x = vadd_f16(x, y);
    x = vfma_f16(x, e, vdup_n_f16(c_cephes_log_q2));
    x = (float16x4_t)(vorr_u16((uint16x4_t)(x), invalid_mask)); // negative arg will be NAN
    return x;
}

static inline float16x8_t log_ps_f16(float16x8_t x)
{
    float16x8_t one = vdupq_n_f16(1);

    x = vmaxq_f16(x, vdupq_n_f16(0)); /* force flush to zero on denormal values */
    uint16x8_t invalid_mask = vcleq_f16(x, vdupq_n_f16(0));

    int16x8_t ux = vreinterpretq_s16_f16(x);

    int16x8_t emm0 = vshrq_n_s16(ux, 10);

    /* keep only the fractional part */
    ux = vandq_s16(ux, vdupq_n_s16(c_inv_mant_mask_f16));
    ux = vorrq_s16(ux, vreinterpretq_s16_f16(vdupq_n_f16(0.5f)));
    x = vreinterpretq_f16_s16(ux);

    emm0 = vsubq_s16(emm0, vdupq_n_s16(0xf));
    float16x8_t e = vcvtq_f16_s16(emm0);

    e = vaddq_f16(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    uint16x8_t mask = vcltq_f16(x, vdupq_n_f16(c_cephes_SQRTHF));
    float16x8_t tmp = vreinterpretq_f16_u16(vandq_u16(vreinterpretq_u16_f16(x), mask));
    x = vsubq_f16(x, one);
    e = vsubq_f16(e, vreinterpretq_f16_u16(vandq_u16(vreinterpretq_u16_f16(one), mask)));
    x = vaddq_f16(x, tmp);

    float16x8_t z = vmulq_f16(x, x);

    float16x8_t y = vdupq_n_f16(c_cephes_log_p0);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p1), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p2), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p3), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p4), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p5), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p6), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p7), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_log_p8), y, x);
    y = vmulq_f16(y, x);

    y = vmulq_f16(y, z);

    y = vfmaq_f16(y, e, vdupq_n_f16(c_cephes_log_q1));

    y = vfmsq_f16(y, z, vdupq_n_f16(0.5f));

    x = vaddq_f16(x, y);
    x = vfmaq_f16(x, e, vdupq_n_f16(c_cephes_log_q2));
    x = vreinterpretq_f16_u16(vorrq_u16(vreinterpretq_u16_f16(x), invalid_mask)); // negative arg will be NAN
    return x;
}

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float16x4_t exp_ps_f16(float16x4_t x)
{
    float16x4_t tmp, fx;

    float16x4_t one = vdup_n_f16(1);
    x = vmin_f16(x, vdup_n_f16(c_exp_hi_f16));
    x = vmax_f16(x, vdup_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
#if _MSC_VER
    fx = vfma_f16(vdup_n_f16(0.5f), x, vcvt_f16_f32(vdupq_n_f32(c_cephes_LOG2EF)));
#else
    fx = vfma_f16(vdup_n_f16(0.5f), x, vdup_n_f16(c_cephes_LOG2EF));
#endif

    /* perform a floorf */
    tmp = vcvt_f16_s16(vcvt_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x4_t mask = vcgt_f16(tmp, fx);
    mask = vand_u16(mask, (uint16x4_t)(one));

    fx = vsub_f16(tmp, (float16x4_t)(mask));

#if _MSC_VER
    tmp = vmul_f16(fx, vcvt_f16_f32(vdupq_n_f32(c_cephes_exp_C1)));
    float16x4_t z = vmul_f16(fx, vcvt_f16_f32(vdupq_n_f32(c_cephes_exp_C2)));
#else
    tmp = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C1));
    float16x4_t z = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C2));
#endif
    x = vsub_f16(x, tmp);
    x = vsub_f16(x, z);

    z = vmul_f16(x, x);

    float16x4_t y = vdup_n_f16(c_cephes_exp_p0);
    y = vfma_f16(vdup_n_f16(c_cephes_exp_p1), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_exp_p2), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_exp_p3), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_exp_p4), y, x);
    y = vfma_f16(vdup_n_f16(c_cephes_exp_p5), y, x);

    y = vfma_f16(x, y, z);
    y = vadd_f16(y, one);

    /* build 2^n */
    int16x4_t mm;
    mm = vcvt_s16_f16(fx);
    mm = vadd_s16(mm, vdup_n_s16(0xf));
    mm = vshl_n_s16(mm, 10);
    float16x4_t pow2n = vreinterpret_f16_s16(mm);

    y = vmul_f16(y, pow2n);
    return y;
}

static inline float16x8_t exp_ps_f16(float16x8_t x)
{
    float16x8_t tmp, fx;

    float16x8_t one = vdupq_n_f16(1);
    x = vminq_f16(x, vdupq_n_f16(c_exp_hi_f16));
    x = vmaxq_f16(x, vdupq_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
#if _MSC_VER
    float16x4_t _c_cephes_LOG2EF = vcvt_f16_f32(vdupq_n_f32(c_cephes_LOG2EF));
    fx = vfmaq_f16(vdupq_n_f16(0.5f), x, vcombine_f16(_c_cephes_LOG2EF, _c_cephes_LOG2EF));
#else
    fx = vfmaq_f16(vdupq_n_f16(0.5f), x, vdupq_n_f16(c_cephes_LOG2EF));
#endif

    /* perform a floorf */
    tmp = vcvtq_f16_s16(vcvtq_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x8_t mask = vcgtq_f16(tmp, fx);
    mask = vandq_u16(mask, vreinterpretq_u16_f16(one));

    fx = vsubq_f16(tmp, vreinterpretq_f16_u16(mask));

#if _MSC_VER
    float16x4_t _c_cephes_exp_C1 = vcvt_f16_f32(vdupq_n_f32(c_cephes_exp_C1));
    tmp = vmulq_f16(fx, vcombine_f16(_c_cephes_exp_C1, _c_cephes_exp_C1));
    float16x4_t _c_cephes_exp_C2 = vcvt_f16_f32(vdupq_n_f32(c_cephes_exp_C2));
    float16x8_t z = vmulq_f16(fx, vcombine_f16(_c_cephes_exp_C2, _c_cephes_exp_C2));
#else
    tmp = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C1));
    float16x8_t z = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C2));
#endif
    x = vsubq_f16(x, tmp);
    x = vsubq_f16(x, z);

    z = vmulq_f16(x, x);

    float16x8_t y = vdupq_n_f16(c_cephes_exp_p0);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p1), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p2), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p3), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p4), y, x);
    y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p5), y, x);

    y = vfmaq_f16(x, y, z);
    y = vaddq_f16(y, one);

    /* build 2^n */
    int16x8_t mm;
    mm = vcvtq_s16_f16(fx);
    mm = vaddq_s16(mm, vdupq_n_s16(0xf));
    mm = vshlq_n_s16(mm, 10);
    float16x8_t pow2n = vreinterpretq_f16_s16(mm);

    y = vmulq_f16(y, pow2n);
    return y;
}

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0        -1.9515295891E-4
#define c_sincof_p1        8.3321608736E-3
#define c_sincof_p2        -1.6666654611E-1
#define c_coscof_p0        2.443315711809948E-005
#define c_coscof_p1        -1.388731625493765E-003
#define c_coscof_p2        4.166664568298827E-002
#define c_cephes_FOPI      1.27323954473516 // 4 / M_PI

/* evaluation of 4 sines & cosines at once.
 *
 *   The code is the exact rewriting of the cephes sinf function.
 *   Precision is excellent as long as x < 8192 (I did not bother to
 *   take into account the special handling they have for greater values
 *   -- it does not return garbage for arguments over 8192, though, but
 *   the extra precision is missing).
 *
 *   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
 *   surprising but correct result.
 *
 *   Note also that when you compute sin(x), cos(x) is available at
 *   almost no extra price so both sin_ps and cos_ps make use of
 *   sincos_ps..
 */
static inline void sincos_ps_f16(float16x4_t x, float16x4_t* ysin, float16x4_t* ycos)
{
    // any x
    float16x4_t y;

    uint16x4_t emm2;

    uint16x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vclt_f16(x, vdup_n_f16(0));
    x = vabs_f16(x);

    /* scale by 4/Pi */
#if _MSC_VER
    float16x4_t _c_cephes_FOPI = vcvt_f16_f32(vdupq_n_f32(c_cephes_FOPI));
    y = vmul_f16(x, _c_cephes_FOPI);
#else
    y = vmul_f16(x, vdup_n_f16(c_cephes_FOPI));
#endif

    /* store the integer part of y in mm0 */
    emm2 = vcvt_u16_f16(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vadd_u16(emm2, vdup_n_u16(1));
    emm2 = vand_u16(emm2, vdup_n_u16(~1));
    y = vcvt_f16_u16(emm2);

    /* get the polynom selection mask
     *     there is one polynom for 0 <= x <= Pi/4
     *     and another one for Pi/4<x<=Pi/2
     *
     *     Both branches will be computed.
     */
    uint16x4_t poly_mask = vtst_u16(emm2, vdup_n_u16(2));

    /* The magic pass: "Extended precision modular arithmetic"
     *     x = ((x - y * DP1) - y * DP2) - y * DP3; */
#if _MSC_VER
    float16x4_t _c_minus_cephes_DP1 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP1));
    float16x4_t _c_minus_cephes_DP2 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP2));
    float16x4_t _c_minus_cephes_DP3 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP3));
    x = vfma_f16(x, y, _c_minus_cephes_DP1);
    x = vfma_f16(x, y, _c_minus_cephes_DP2);
    x = vfma_f16(x, y, _c_minus_cephes_DP3);
#else
    x = vfma_f16(x, y, vdup_n_f16(c_minus_cephes_DP1));
    x = vfma_f16(x, y, vdup_n_f16(c_minus_cephes_DP2));
    x = vfma_f16(x, y, vdup_n_f16(c_minus_cephes_DP3));
#endif

    sign_mask_sin = veor_u16(sign_mask_sin, vtst_u16(emm2, vdup_n_u16(4)));
    sign_mask_cos = vtst_u16(vsub_u16(emm2, vdup_n_u16(2)), vdup_n_u16(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    float16x4_t z = vmul_f16(x, x);
    float16x4_t y1, y2;

    y1 = vfma_f16(vdup_n_f16(c_coscof_p1), z, vdup_n_f16(c_coscof_p0));
    y2 = vfma_f16(vdup_n_f16(c_sincof_p1), z, vdup_n_f16(c_sincof_p0));
    y1 = vfma_f16(vdup_n_f16(c_coscof_p2), y1, z);
    y2 = vfma_f16(vdup_n_f16(c_sincof_p2), y2, z);
    y1 = vmul_f16(y1, z);
    y2 = vmul_f16(y2, z);
    y1 = vmul_f16(y1, z);
    y1 = vfms_f16(y1, z, vdup_n_f16(0.5f));
    y2 = vfma_f16(x, y2, x);
    y1 = vadd_f16(y1, vdup_n_f16(1));

    /* select the correct result from the two polynoms */
    float16x4_t ys = vbsl_f16(poly_mask, y1, y2);
    float16x4_t yc = vbsl_f16(poly_mask, y2, y1);
    *ysin = vbsl_f16(sign_mask_sin, vneg_f16(ys), ys);
    *ycos = vbsl_f16(sign_mask_cos, yc, vneg_f16(yc));
}

static inline void sincos_ps_f16(float16x8_t x, float16x8_t* ysin, float16x8_t* ycos)
{
    // any x
    float16x8_t y;

    uint16x8_t emm2;

    uint16x8_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f16(x, vdupq_n_f16(0));
    x = vabsq_f16(x);

    /* scale by 4/Pi */
#if _MSC_VER
    float16x4_t _c_cephes_FOPI = vcvt_f16_f32(vdupq_n_f32(c_cephes_FOPI));
    y = vmulq_f16(x, vcombine_f16(_c_cephes_FOPI, _c_cephes_FOPI));
#else
    y = vmulq_f16(x, vdupq_n_f16(c_cephes_FOPI));
#endif

    /* store the integer part of y in mm0 */
    emm2 = vcvtq_u16_f16(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u16(emm2, vdupq_n_u16(1));
    emm2 = vandq_u16(emm2, vdupq_n_u16(~1));
    y = vcvtq_f16_u16(emm2);

    /* get the polynom selection mask
     *     there is one polynom for 0 <= x <= Pi/4
     *     and another one for Pi/4<x<=Pi/2
     *
     *     Both branches will be computed.
     */
    uint16x8_t poly_mask = vtstq_u16(emm2, vdupq_n_u16(2));

    /* The magic pass: "Extended precision modular arithmetic"
     *     x = ((x - y * DP1) - y * DP2) - y * DP3; */
#if _MSC_VER
    float16x4_t _c_minus_cephes_DP1 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP1));
    float16x4_t _c_minus_cephes_DP2 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP2));
    float16x4_t _c_minus_cephes_DP3 = vcvt_f16_f32(vdupq_n_f32(c_minus_cephes_DP3));
    x = vfmaq_f16(x, y, vcombine_f16(_c_minus_cephes_DP1, _c_minus_cephes_DP1));
    x = vfmaq_f16(x, y, vcombine_f16(_c_minus_cephes_DP2, _c_minus_cephes_DP2));
    x = vfmaq_f16(x, y, vcombine_f16(_c_minus_cephes_DP3, _c_minus_cephes_DP3));
#else
    x = vfmaq_f16(x, y, vdupq_n_f16(c_minus_cephes_DP1));
    x = vfmaq_f16(x, y, vdupq_n_f16(c_minus_cephes_DP2));
    x = vfmaq_f16(x, y, vdupq_n_f16(c_minus_cephes_DP3));
#endif

    sign_mask_sin = veorq_u16(sign_mask_sin, vtstq_u16(emm2, vdupq_n_u16(4)));
    sign_mask_cos = vtstq_u16(vsubq_u16(emm2, vdupq_n_u16(2)), vdupq_n_u16(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    float16x8_t z = vmulq_f16(x, x);
    float16x8_t y1, y2;

    y1 = vfmaq_f16(vdupq_n_f16(c_coscof_p1), z, vdupq_n_f16(c_coscof_p0));
    y2 = vfmaq_f16(vdupq_n_f16(c_sincof_p1), z, vdupq_n_f16(c_sincof_p0));
    y1 = vfmaq_f16(vdupq_n_f16(c_coscof_p2), y1, z);
    y2 = vfmaq_f16(vdupq_n_f16(c_sincof_p2), y2, z);
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, z);
    y1 = vmulq_f16(y1, z);
    y1 = vfmsq_f16(y1, z, vdupq_n_f16(0.5f));
    y2 = vfmaq_f16(x, y2, x);
    y1 = vaddq_f16(y1, vdupq_n_f16(1));

    /* select the correct result from the two polynoms */
    float16x8_t ys = vbslq_f16(poly_mask, y1, y2);
    float16x8_t yc = vbslq_f16(poly_mask, y2, y1);
    *ysin = vbslq_f16(sign_mask_sin, vnegq_f16(ys), ys);
    *ycos = vbslq_f16(sign_mask_cos, yc, vnegq_f16(yc));
}

static inline float16x4_t sin_ps_f16(float16x4_t x)
{
    float16x4_t ysin, ycos;
    sincos_ps_f16(x, &ysin, &ycos);
    return ysin;
}

static inline float16x8_t sin_ps_f16(float16x8_t x)
{
    float16x8_t ysin, ycos;
    sincos_ps_f16(x, &ysin, &ycos);
    return ysin;
}

static inline float16x4_t cos_ps_f16(float16x4_t x)
{
    float16x4_t ysin, ycos;
    sincos_ps_f16(x, &ysin, &ycos);
    return ycos;
}

static inline float16x8_t cos_ps_f16(float16x8_t x)
{
    float16x8_t ysin, ycos;
    sincos_ps_f16(x, &ysin, &ycos);
    return ycos;
}

#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float16x4_t tanh_ps_f16(float16x4_t x)
{
    float16x4_t x2 = vabs_f16(x);

    uint16x4_t tiny_mask = vcge_f16(x2, vdup_n_f16(c_tanh_tiny));

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = (float16x4_t)(vbsl_u16(vcge_f16(vdup_n_f16(c_tanh_hi), x2), (uint16x4_t)(x2), (uint16x4_t)(vdup_n_f16(c_tanh_hi))));

    // since the polynomials are odd/even, we need x**2.
    float16x4_t z = vmul_f16(x2, x2);

    // evaluate the numerator polynomial y.
    float16x4_t y = vdup_n_f16(c_tanh_alpha_13);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_11), y, z);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_9), y, z);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_7), y, z);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_5), y, z);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_3), y, z);
    y = vfma_f16(vdup_n_f16(c_tanh_alpha_1), y, z);
    y = vmul_f16(y, x2);

    // evaluate the denominator polynomial w.
    float16x4_t w = vdup_n_f16(c_tanh_beta_6);
    w = vfma_f16(vdup_n_f16(c_tanh_beta_4), w, z);
    w = vfma_f16(vdup_n_f16(c_tanh_beta_2), w, z);
    w = vfma_f16(vdup_n_f16(c_tanh_beta_0), w, z);

    // divide the numerator by the denominator.
    y = vdiv_f16(y, w);

    // reinstate the sign.
    y = (float16x4_t)(vbsl_u16(vdup_n_u16(1u << 15), (uint16x4_t)(x), (uint16x4_t)(y)));

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = (float16x4_t)(vbsl_u16(tiny_mask, (uint16x4_t)(y), (uint16x4_t)(x)));

    return y;
}

static inline float16x8_t tanh_ps_f16(float16x8_t x)
{
    float16x8_t x2 = vabsq_f16(x);

    uint16x8_t tiny_mask = vcgeq_f16(x2, vdupq_n_f16(c_tanh_tiny));

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = vreinterpretq_f16_u16(vbslq_u16(vcgeq_f16(vdupq_n_f16(c_tanh_hi), x2), vreinterpretq_u16_f16(x2), vreinterpretq_u16_f16(vdupq_n_f16(c_tanh_hi))));

    // since the polynomials are odd/even, we need x**2.
    float16x8_t z = vmulq_f16(x2, x2);

    // evaluate the numerator polynomial y.
    float16x8_t y = vdupq_n_f16(c_tanh_alpha_13);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_11), y, z);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_9), y, z);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_7), y, z);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_5), y, z);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_3), y, z);
    y = vfmaq_f16(vdupq_n_f16(c_tanh_alpha_1), y, z);
    y = vmulq_f16(y, x2);

    // evaluate the denominator polynomial w.
    float16x8_t w = vdupq_n_f16(c_tanh_beta_6);
    w = vfmaq_f16(vdupq_n_f16(c_tanh_beta_4), w, z);
    w = vfmaq_f16(vdupq_n_f16(c_tanh_beta_2), w, z);
    w = vfmaq_f16(vdupq_n_f16(c_tanh_beta_0), w, z);

    // divide the numerator by the denominator.
    y = vdivq_f16(y, w);

    // reinstate the sign.
    y = vreinterpretq_f16_u16(vbslq_u16(vdupq_n_u16(1u << 15), vreinterpretq_u16_f16(x), vreinterpretq_u16_f16(y)));

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = vreinterpretq_f16_u16(vbslq_u16(tiny_mask, vreinterpretq_u16_f16(y), vreinterpretq_u16_f16(x)));

    return y;
}

static inline float16x4_t sigmoid_ps_f16(float16x4_t _v)
{
    float16x4_t _one = vdup_n_f16(1.f);
    _v = vneg_f16(_v);
    _v = exp_ps_f16(_v);
    _v = vadd_f16(_v, _one);
    return vdiv_f16(_one, _v);
}

static inline float16x8_t sigmoid_ps_f16(float16x8_t _v)
{
    float16x8_t _one = vdupq_n_f16(1.f);
    _v = vnegq_f16(_v);
    _v = exp_ps_f16(_v);
    _v = vaddq_f16(_v, _one);
    return vdivq_f16(_one, _v);
}

#endif // NEON_MATHFUN_FP16S_H
