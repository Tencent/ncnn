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
static inline float16x4_t log_ps(float16x4_t x)
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
    float16x4_t tmp = vreinterpret_f16_u16(vand_u16(vreinterpret_u16_f16(x), mask));
    x = vsub_f16(x, one);
    e = vsub_f16(e, vreinterpret_f16_u16(vand_u16(vreinterpret_u16_f16(one), mask)));
    x = vadd_f16(x, tmp);

    float16x4_t z = vmul_f16(x, x);

    float16x4_t y = vdup_n_f16(c_cephes_log_p0);
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p1));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p2));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p3));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p4));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p5));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p6));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p7));
    y = vmul_f16(y, x);
    y = vadd_f16(y, vdup_n_f16(c_cephes_log_p8));
    y = vmul_f16(y, x);

    y = vmul_f16(y, z);

    tmp = vmul_f16(e, vdup_n_f16(c_cephes_log_q1));
    y = vadd_f16(y, tmp);

    tmp = vmul_f16(z, vdup_n_f16(0.5f));
    y = vsub_f16(y, tmp);

    tmp = vmul_f16(e, vdup_n_f16(c_cephes_log_q2));
    x = vadd_f16(x, y);
    x = vadd_f16(x, tmp);
    x = vreinterpret_f16_u16(vorr_u16(vreinterpret_u16_f16(x), invalid_mask)); // negative arg will be NAN
    return x;
}

static inline float16x8_t log_ps(float16x8_t x)
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
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p1));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p2));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p3));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p4));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p5));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p6));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p7));
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, vdupq_n_f16(c_cephes_log_p8));
    y = vmulq_f16(y, x);

    y = vmulq_f16(y, z);

    tmp = vmulq_f16(e, vdupq_n_f16(c_cephes_log_q1));
    y = vaddq_f16(y, tmp);

    tmp = vmulq_f16(z, vdupq_n_f16(0.5f));
    y = vsubq_f16(y, tmp);

    tmp = vmulq_f16(e, vdupq_n_f16(c_cephes_log_q2));
    x = vaddq_f16(x, y);
    x = vaddq_f16(x, tmp);
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
static inline float16x4_t exp_ps(float16x4_t x)
{
    float16x4_t tmp, fx;

    float16x4_t one = vdup_n_f16(1);
    x = vmin_f16(x, vdup_n_f16(c_exp_hi_f16));
    x = vmax_f16(x, vdup_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfma_f16(vdup_n_f16(0.5f), x, vdup_n_f16(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvt_f16_s16(vcvt_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x4_t mask = vcgt_f16(tmp, fx);
    mask = vand_u16(mask, vreinterpret_u16_f16(one));

    fx = vsub_f16(tmp, vreinterpret_f16_u16(mask));

    tmp = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C1));
    float16x4_t z = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C2));
    x = vsub_f16(x, tmp);
    x = vsub_f16(x, z);

    static const __fp16 cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float16x4_t y = vld1_dup_f16(cephes_exp_p + 0);
    float16x4_t c1 = vld1_dup_f16(cephes_exp_p + 1);
    float16x4_t c2 = vld1_dup_f16(cephes_exp_p + 2);
    float16x4_t c3 = vld1_dup_f16(cephes_exp_p + 3);
    float16x4_t c4 = vld1_dup_f16(cephes_exp_p + 4);
    float16x4_t c5 = vld1_dup_f16(cephes_exp_p + 5);

    y = vmul_f16(y, x);
    z = vmul_f16(x, x);

    y = vadd_f16(y, c1);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c2);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c3);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c4);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c5);

    y = vmul_f16(y, z);
    y = vadd_f16(y, x);
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

static inline float16x8_t exp_ps(float16x8_t x)
{
    float16x8_t tmp, fx;

    float16x8_t one = vdupq_n_f16(1);
    x = vminq_f16(x, vdupq_n_f16(c_exp_hi_f16));
    x = vmaxq_f16(x, vdupq_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfmaq_f16(vdupq_n_f16(0.5f), x, vdupq_n_f16(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f16_s16(vcvtq_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x8_t mask = vcgtq_f16(tmp, fx);
    mask = vandq_u16(mask, vreinterpretq_u16_f16(one));

    fx = vsubq_f16(tmp, vreinterpretq_f16_u16(mask));

    tmp = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C1));
    float16x8_t z = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C2));
    x = vsubq_f16(x, tmp);
    x = vsubq_f16(x, z);

    static const __fp16 cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float16x8_t y = vld1q_dup_f16(cephes_exp_p + 0);
    float16x8_t c1 = vld1q_dup_f16(cephes_exp_p + 1);
    float16x8_t c2 = vld1q_dup_f16(cephes_exp_p + 2);
    float16x8_t c3 = vld1q_dup_f16(cephes_exp_p + 3);
    float16x8_t c4 = vld1q_dup_f16(cephes_exp_p + 4);
    float16x8_t c5 = vld1q_dup_f16(cephes_exp_p + 5);

    y = vmulq_f16(y, x);
    z = vmulq_f16(x, x);

    y = vaddq_f16(y, c1);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c2);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c3);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c4);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c5);

    y = vmulq_f16(y, z);
    y = vaddq_f16(y, x);
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
static inline void sincos_ps(float16x4_t x, float16x4_t* ysin, float16x4_t* ycos)
{
    // any x
    float16x4_t xmm1, xmm2, xmm3, y;

    uint16x4_t emm2;

    uint16x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vclt_f16(x, vdup_n_f16(0));
    x = vabs_f16(x);

    /* scale by 4/Pi */
    y = vmul_f16(x, vdup_n_f16(c_cephes_FOPI));

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
    xmm1 = vmul_n_f16(y, c_minus_cephes_DP1);
    xmm2 = vmul_n_f16(y, c_minus_cephes_DP2);
    xmm3 = vmul_n_f16(y, c_minus_cephes_DP3);
    x = vadd_f16(x, xmm1);
    x = vadd_f16(x, xmm2);
    x = vadd_f16(x, xmm3);

    sign_mask_sin = veor_u16(sign_mask_sin, vtst_u16(emm2, vdup_n_u16(4)));
    sign_mask_cos = vtst_u16(vsub_u16(emm2, vdup_n_u16(2)), vdup_n_u16(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    float16x4_t z = vmul_f16(x, x);
    float16x4_t y1, y2;

    y1 = vmul_n_f16(z, c_coscof_p0);
    y2 = vmul_n_f16(z, c_sincof_p0);
    y1 = vadd_f16(y1, vdup_n_f16(c_coscof_p1));
    y2 = vadd_f16(y2, vdup_n_f16(c_sincof_p1));
    y1 = vmul_f16(y1, z);
    y2 = vmul_f16(y2, z);
    y1 = vadd_f16(y1, vdup_n_f16(c_coscof_p2));
    y2 = vadd_f16(y2, vdup_n_f16(c_sincof_p2));
    y1 = vmul_f16(y1, z);
    y2 = vmul_f16(y2, z);
    y1 = vmul_f16(y1, z);
    y2 = vmul_f16(y2, x);
    y1 = vsub_f16(y1, vmul_f16(z, vdup_n_f16(0.5f)));
    y2 = vadd_f16(y2, x);
    y1 = vadd_f16(y1, vdup_n_f16(1));

    /* select the correct result from the two polynoms */
    float16x4_t ys = vbsl_f16(poly_mask, y1, y2);
    float16x4_t yc = vbsl_f16(poly_mask, y2, y1);
    *ysin = vbsl_f16(sign_mask_sin, vneg_f16(ys), ys);
    *ycos = vbsl_f16(sign_mask_cos, yc, vneg_f16(yc));
}

static inline void sincos_ps(float16x8_t x, float16x8_t* ysin, float16x8_t* ycos)
{
    // any x
    float16x8_t xmm1, xmm2, xmm3, y;

    uint16x8_t emm2;

    uint16x8_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f16(x, vdupq_n_f16(0));
    x = vabsq_f16(x);

    /* scale by 4/Pi */
    y = vmulq_f16(x, vdupq_n_f16(c_cephes_FOPI));

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
    xmm1 = vmulq_n_f16(y, c_minus_cephes_DP1);
    xmm2 = vmulq_n_f16(y, c_minus_cephes_DP2);
    xmm3 = vmulq_n_f16(y, c_minus_cephes_DP3);
    x = vaddq_f16(x, xmm1);
    x = vaddq_f16(x, xmm2);
    x = vaddq_f16(x, xmm3);

    sign_mask_sin = veorq_u16(sign_mask_sin, vtstq_u16(emm2, vdupq_n_u16(4)));
    sign_mask_cos = vtstq_u16(vsubq_u16(emm2, vdupq_n_u16(2)), vdupq_n_u16(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    float16x8_t z = vmulq_f16(x, x);
    float16x8_t y1, y2;

    y1 = vmulq_n_f16(z, c_coscof_p0);
    y2 = vmulq_n_f16(z, c_sincof_p0);
    y1 = vaddq_f16(y1, vdupq_n_f16(c_coscof_p1));
    y2 = vaddq_f16(y2, vdupq_n_f16(c_sincof_p1));
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, z);
    y1 = vaddq_f16(y1, vdupq_n_f16(c_coscof_p2));
    y2 = vaddq_f16(y2, vdupq_n_f16(c_sincof_p2));
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, z);
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, x);
    y1 = vsubq_f16(y1, vmulq_f16(z, vdupq_n_f16(0.5f)));
    y2 = vaddq_f16(y2, x);
    y1 = vaddq_f16(y1, vdupq_n_f16(1));

    /* select the correct result from the two polynoms */
    float16x8_t ys = vbslq_f16(poly_mask, y1, y2);
    float16x8_t yc = vbslq_f16(poly_mask, y2, y1);
    *ysin = vbslq_f16(sign_mask_sin, vnegq_f16(ys), ys);
    *ycos = vbslq_f16(sign_mask_cos, yc, vnegq_f16(yc));
}

static inline float16x4_t sin_ps(float16x4_t x)
{
    float16x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ysin;
}

static inline float16x8_t sin_ps(float16x8_t x)
{
    float16x8_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ysin;
}

static inline float16x4_t cos_ps(float16x4_t x)
{
    float16x4_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ycos;
}

static inline float16x8_t cos_ps(float16x8_t x)
{
    float16x8_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos);
    return ycos;
}

// tanh neon vector version
// refer the scalar version from Cephes Math Library

#define c_cephes_HALFMAXLOGF_f16 4.5078125f
#define c_cephes_tanh_C1         0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float16x4_t tanh_ps(float16x4_t x)
{
    float16x4_t x2 = vabs_f16(x);

    uint16x4_t mask_l = vcge_f16(x2, vdup_n_f16(c_cephes_tanh_C1));
    uint16x4_t mask_l2 = vcgt_f16(x2, vdup_n_f16(c_cephes_HALFMAXLOGF_f16));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float16x4_t _one = vdup_n_f16(1.f);
    float16x4_t _two = vdup_n_f16(2.f);
    float16x4_t exp_x_x = exp_ps(vadd_f16(x, x));
    float16x4_t y0 = vsub_f16(_one, vdiv_f16(_two, vadd_f16(exp_x_x, _one)));

    // abs(x) < 0.625
    /*
        z = x2 * x2;
        z =
        (((( -5.70498872745E-3 * z
        + 2.06390887954E-2) * z
        - 5.37397155531E-2) * z
        + 1.33314422036E-1) * z
        - 3.33332819422E-1) * z * x
        + x;
    */
    static const __fp16 cephes_tanh_p[5] = {c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4};
    float16x4_t y = vld1_dup_f16(cephes_tanh_p + 0);
    float16x4_t c1 = vld1_dup_f16(cephes_tanh_p + 1);
    float16x4_t c2 = vld1_dup_f16(cephes_tanh_p + 2);
    float16x4_t c3 = vld1_dup_f16(cephes_tanh_p + 3);
    float16x4_t c4 = vld1_dup_f16(cephes_tanh_p + 4);

    float16x4_t z = vmul_f16(x, x);

    y = vmul_f16(y, z);
    y = vadd_f16(y, c1);
    y = vmul_f16(y, z);
    y = vadd_f16(y, c2);
    y = vmul_f16(y, z);
    y = vadd_f16(y, c3);
    y = vmul_f16(y, z);
    y = vadd_f16(y, c4);

    y = vmul_f16(y, z);
    y = vmul_f16(y, x);
    y = vadd_f16(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint16x4_t mask_pos = vcgt_f16(x2, vdup_n_f16(0.f));
    float16x4_t y1 = vreinterpret_f16_u16(vbsl_u16(mask_pos, vreinterpret_u16_f16(vdup_n_f16(1.f)), vreinterpret_u16_f16(vdup_n_f16(-1.f))));

    y = vreinterpret_f16_u16(vbsl_u16(mask_l, vreinterpret_u16_f16(y0), vreinterpret_u16_f16(y)));
    y = vreinterpret_f16_u16(vbsl_u16(mask_l2, vreinterpret_u16_f16(y1), vreinterpret_u16_f16(y)));
    return y;
}

static inline float16x8_t tanh_ps(float16x8_t x)
{
    float16x8_t x2 = vabsq_f16(x);

    uint16x8_t mask_l = vcgeq_f16(x2, vdupq_n_f16(c_cephes_tanh_C1));
    uint16x8_t mask_l2 = vcgtq_f16(x2, vdupq_n_f16(c_cephes_HALFMAXLOGF_f16));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float16x8_t _one = vdupq_n_f16(1.f);
    float16x8_t _two = vdupq_n_f16(2.f);
    float16x8_t exp_x_x = exp_ps(vaddq_f16(x, x));
    float16x8_t y0 = vsubq_f16(_one, vdivq_f16(_two, vaddq_f16(exp_x_x, _one)));

    // abs(x) < 0.625
    /*
     *        z = x2 * x2;
     *        z =
     *        (((( -5.70498872745E-3 * z
     *        + 2.06390887954E-2) * z
     *        - 5.37397155531E-2) * z
     *        + 1.33314422036E-1) * z
     *        - 3.33332819422E-1) * z * x
     *        + x;
     */
    static const __fp16 cephes_tanh_p[5] = {c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4};
    float16x8_t y = vld1q_dup_f16(cephes_tanh_p + 0);
    float16x8_t c1 = vld1q_dup_f16(cephes_tanh_p + 1);
    float16x8_t c2 = vld1q_dup_f16(cephes_tanh_p + 2);
    float16x8_t c3 = vld1q_dup_f16(cephes_tanh_p + 3);
    float16x8_t c4 = vld1q_dup_f16(cephes_tanh_p + 4);

    float16x8_t z = vmulq_f16(x, x);

    y = vmulq_f16(y, z);
    y = vaddq_f16(y, c1);
    y = vmulq_f16(y, z);
    y = vaddq_f16(y, c2);
    y = vmulq_f16(y, z);
    y = vaddq_f16(y, c3);
    y = vmulq_f16(y, z);
    y = vaddq_f16(y, c4);

    y = vmulq_f16(y, z);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint16x8_t mask_pos = vcgtq_f16(x2, vdupq_n_f16(0.f));
    float16x8_t y1 = vreinterpretq_f16_u16(vbslq_u16(mask_pos, vreinterpretq_u16_f16(vdupq_n_f16(1.f)), vreinterpretq_u16_f16(vdupq_n_f16(-1.f))));

    y = vreinterpretq_f16_u16(vbslq_u16(mask_l, vreinterpretq_u16_f16(y0), vreinterpretq_u16_f16(y)));
    y = vreinterpretq_f16_u16(vbslq_u16(mask_l2, vreinterpretq_u16_f16(y1), vreinterpretq_u16_f16(y)));
    return y;
}

static inline float16x4_t sigmoid_ps(float16x4_t _v)
{
    float16x4_t _one = vdup_n_f16(1.f);
    _v = vneg_f16(_v);
    _v = exp_ps(_v);
    _v = vadd_f16(_v, _one);
    return vdiv_f16(_one, _v);
}

static inline float16x8_t sigmoid_ps(float16x8_t _v)
{
    float16x8_t _one = vdupq_n_f16(1.f);
    _v = vnegq_f16(_v);
    _v = exp_ps(_v);
    _v = vaddq_f16(_v, _one);
    return vdivq_f16(_one, _v);
}
