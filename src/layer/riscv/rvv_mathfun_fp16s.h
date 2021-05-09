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

#ifndef RVV_MATHFUN_FP16S_H
#define RVV_MATHFUN_FP16S_H

#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif

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

static inline vfloat16m8_t log_ps(vfloat16m8_t x, word_type vl)
{
    x = vfmax_vf_f16m8(x, 0.f, vl); /* force flush to zero on denormal values */
    vbool2_t invalid_mask = vmfle_vf_f16m8_b2(x, 0.f, vl);

    vint16m8_t ux = vreinterpret_v_f16m8_i16m8(x);

    vint16m8_t emm0 = vsra_vx_i16m8(ux, 10, vl);

    /* keep only the fractional part */
    ux = vand_vx_i16m8(ux, c_inv_mant_mask_f16, vl);
    ux = vor_vx_i16m8(ux, 14336 /* reinterpret_cast<short>((__fp16)0.5) */, vl);
    x = vreinterpret_v_i16m8_f16m8(ux);

    emm0 = vsub_vx_i16m8(emm0, 0xf, vl);
    vfloat16m8_t e = vfcvt_f_x_v_f16m8(emm0, vl);

    e = vfadd_vf_f16m8(e, 1.f, vl);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    vbool2_t mask = vmflt_vf_f16m8_b2(x, c_cephes_SQRTHF, vl);
    x = vfadd_vv_f16m8_m(mask, x, x, x, vl);
    x = vfsub_vf_f16m8(x, 1.f, vl);
    e = vfsub_vf_f16m8_m(mask, e, e, 1.f, vl);

    vfloat16m8_t z = vfmul_vv_f16m8(x, x, vl);

    vfloat16m8_t y = vfmul_vf_f16m8(x, c_cephes_log_p0, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p1, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p2, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p3, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p4, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p5, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p6, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p7, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_log_p8, vl);
    y = vfmul_vv_f16m8(y, x, vl);

    y = vfmul_vv_f16m8(y, z, vl);

    vfloat16m8_t tmp = vfmul_vf_f16m8(e, c_cephes_log_q1, vl);
    y = vfadd_vv_f16m8(y, tmp, vl);

    tmp = vfmul_vf_f16m8(z, 0.5f, vl);
    y = vfsub_vv_f16m8(y, tmp, vl);

    tmp = vfmul_vf_f16m8(e, c_cephes_log_q2, vl);
    x = vfadd_vv_f16m8(x, y, vl);
    x = vfadd_vv_f16m8(x, tmp, vl);
    x = vreinterpret_v_u16m8_f16m8(vor_vx_u16m8_m(invalid_mask, vreinterpret_v_f16m8_u16m8(x), vreinterpret_v_f16m8_u16m8(x), 0xffff, vl)); // negative arg will be NAN
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

static inline vfloat16m8_t exp_ps(vfloat16m8_t x, word_type vl)
{
    vfloat16m8_t tmp, fx;

    x = vfmin_vf_f16m8(x, c_exp_hi_f16, vl);
    x = vfmax_vf_f16m8(x, c_exp_lo_f16, vl);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfmacc_vf_f16m8(vfmv_v_f_f16m8(0.5f, vl), c_cephes_LOG2EF, x, vl);

    /* perform a floorf */
    tmp = vfcvt_f_x_v_f16m8(vfcvt_x_f_v_i16m8(fx, vl), vl);

    /* if greater, substract 1 */
    vbool2_t mask = vmfgt_vv_f16m8_b2(tmp, fx, vl);
    fx = vfsub_vf_f16m8_m(mask, tmp, tmp, 1.f, vl);

    tmp = vfmul_vf_f16m8(fx, c_cephes_exp_C1, vl);
    vfloat16m8_t z = vfmul_vf_f16m8(fx, c_cephes_exp_C2, vl);
    x = vfsub_vv_f16m8(x, tmp, vl);
    x = vfsub_vv_f16m8(x, z, vl);

    vfloat16m8_t y = vfmul_vf_f16m8(x, c_cephes_exp_p0, vl);
    z = vfmul_vv_f16m8(x, x, vl);

    y = vfadd_vf_f16m8(y, c_cephes_exp_p1, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_exp_p2, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_exp_p3, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_exp_p4, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, c_cephes_exp_p5, vl);

    y = vfmul_vv_f16m8(y, z, vl);
    y = vfadd_vv_f16m8(y, x, vl);
    y = vfadd_vf_f16m8(y, 1.f, vl);

    /* build 2^n */
    vint16m8_t mm = vfcvt_x_f_v_i16m8(fx, vl);
    mm = vadd_vx_i16m8(mm, 0xf, vl);
    mm = vsll_vx_i16m8(mm, 10, vl);
    vfloat16m8_t pow2n = vreinterpret_v_i16m8_f16m8(mm);

    y = vfmul_vv_f16m8(y, pow2n, vl);
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

static inline void sincos_ps(vfloat16m8_t x, vfloat16m8_t* ysin, vfloat16m8_t* ycos, word_type vl)
{
    // any x
    vfloat16m8_t xmm1, xmm2, xmm3, y;

    vuint16m8_t emm2;

    vbool2_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vmflt_vf_f16m8_b2(x, 0.f, vl);
    x = vfsgnj_vf_f16m8(x, 1.f, vl);

    /* scale by 4/Pi */
    y = vfmul_vf_f16m8(x, c_cephes_FOPI, vl);

    /* store the integer part of y in mm0 */
    emm2 = vfcvt_xu_f_v_u16m8(y, vl);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vadd_vx_u16m8(emm2, 1, vl);
    emm2 = vand_vx_u16m8(emm2, ~1, vl);
    y = vfcvt_f_xu_v_f16m8(emm2, vl);

    /* get the polynom selection mask
     *     there is one polynom for 0 <= x <= Pi/4
     *     and another one for Pi/4<x<=Pi/2
     *
     *     Both branches will be computed.
     */
    vbool2_t poly_mask = vmsne_vx_u16m8_b2(vand_vx_u16m8(emm2, 2, vl), 0, vl);

    /* The magic pass: "Extended precision modular arithmetic"
     *     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = vfmul_vf_f16m8(y, c_minus_cephes_DP1, vl);
    xmm2 = vfmul_vf_f16m8(y, c_minus_cephes_DP2, vl);
    xmm3 = vfmul_vf_f16m8(y, c_minus_cephes_DP3, vl);
    x = vfadd_vv_f16m8(x, xmm1, vl);
    x = vfadd_vv_f16m8(x, xmm2, vl);
    x = vfadd_vv_f16m8(x, xmm3, vl);

    sign_mask_sin = vmxor_mm_b2(sign_mask_sin, vmsne_vx_u16m8_b2(vand_vx_u16m8(emm2, 4, vl), 0, vl), vl);
    sign_mask_cos = vmsne_vx_u16m8_b2(vand_vx_u16m8(vsub_vx_u16m8(emm2, 2, vl), 4, vl), 0, vl);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    vfloat16m8_t z = vfmul_vv_f16m8(x, x, vl);
    vfloat16m8_t y1, y2;

    y1 = vfmul_vf_f16m8(z, c_coscof_p0, vl);
    y2 = vfmul_vf_f16m8(z, c_sincof_p0, vl);
    y1 = vfadd_vf_f16m8(y1, c_coscof_p1, vl);
    y2 = vfadd_vf_f16m8(y2, c_sincof_p1, vl);
    y1 = vfmul_vv_f16m8(y1, z, vl);
    y2 = vfmul_vv_f16m8(y2, z, vl);
    y1 = vfadd_vf_f16m8(y1, c_coscof_p2, vl);
    y2 = vfadd_vf_f16m8(y2, c_sincof_p2, vl);
    y1 = vfmul_vv_f16m8(y1, z, vl);
    y2 = vfmul_vv_f16m8(y2, z, vl);
    y1 = vfmul_vv_f16m8(y1, z, vl);
    y2 = vfmul_vv_f16m8(y2, x, vl);
    y1 = vfsub_vv_f16m8(y1, vfmul_vf_f16m8(z, 0.5f, vl), vl);
    y2 = vfadd_vv_f16m8(y2, x, vl);
    y1 = vfadd_vf_f16m8(y1, 1.f, vl);

    /* select the correct result from the two polynoms */
    vfloat16m8_t ys = vmerge_vvm_f16m8(poly_mask, y2, y1, vl);
    vfloat16m8_t yc = vmerge_vvm_f16m8(poly_mask, y1, y2, vl);
    *ysin = vmerge_vvm_f16m8(sign_mask_sin, ys, vfneg_v_f16m8(ys, vl), vl);
    *ycos = vmerge_vvm_f16m8(sign_mask_cos, vfneg_v_f16m8(yc, vl), yc, vl);
}

static inline vfloat16m8_t sin_ps(vfloat16m8_t x, word_type vl)
{
    vfloat16m8_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos, vl);
    return ysin;
}

static inline vfloat16m8_t cos_ps(vfloat16m8_t x, word_type vl)
{
    vfloat16m8_t ysin, ycos;
    sincos_ps(x, &ysin, &ycos, vl);
    return ycos;
}

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1     0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

static inline vfloat16m8_t tanh_ps(vfloat16m8_t x, word_type vl)
{
    vfloat16m8_t x2 = vfsgnj_vf_f16m8(x, 1.f, vl);

    vbool2_t mask_l = vmfge_vf_f16m8_b2(x2, c_cephes_tanh_C1, vl);
    vbool2_t mask_l2 = vmfgt_vf_f16m8_b2(x2, c_cephes_HALFMAXLOGF, vl);

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    vfloat16m8_t exp_x_x = exp_ps(vfadd_vv_f16m8(x, x, vl), vl);
    vfloat16m8_t y0 = vfrsub_vf_f16m8(vfrdiv_vf_f16m8(vfadd_vf_f16m8(exp_x_x, 1.f, vl), 2.f, vl), 1.f, vl);

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
    vfloat16m8_t z = vfmul_vv_f16m8(x, x, vl);

    vfloat16m8_t y = vfmul_vf_f16m8(z, c_cephes_tanh_p0, vl);
    y = vfadd_vf_f16m8(y, c_cephes_tanh_p1, vl);
    y = vfmul_vv_f16m8(y, z, vl);
    y = vfadd_vf_f16m8(y, c_cephes_tanh_p2, vl);
    y = vfmul_vv_f16m8(y, z, vl);
    y = vfadd_vf_f16m8(y, c_cephes_tanh_p3, vl);
    y = vfmul_vv_f16m8(y, z, vl);
    y = vfadd_vf_f16m8(y, c_cephes_tanh_p4, vl);

    y = vfmul_vv_f16m8(y, z, vl);
    y = vfmul_vv_f16m8(y, x, vl);
    y = vfadd_vv_f16m8(y, x, vl);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    vbool2_t mask_pos = vmfgt_vf_f16m8_b2(x, 0.f, vl);
    vfloat16m8_t y1 = vfmerge_vfm_f16m8(mask_pos, vfmv_v_f_f16m8(1.f, vl), -1.f, vl);

    y = vmerge_vvm_f16m8(mask_l, y, y0, vl);
    y = vmerge_vvm_f16m8(mask_l2, y, y1, vl);
    return y;
}

static inline vfloat16m8_t pow_ps(vfloat16m8_t a, vfloat16m8_t b, word_type vl)
{
    // pow(x, m) = exp(m * log(x))
    return exp_ps(vfmul_vv_f16m8(b, log_ps(a, vl), vl), vl);
}

static inline vfloat16m8_t sigmoid_ps(vfloat16m8_t _v, word_type vl)
{
    _v = vfneg_v_f16m8(_v, vl);
    _v = exp_ps(_v, vl);
    _v = vfadd_vf_f16m8(_v, 1.f, vl);
    vfloat16m8_t _reciprocal = vfrec7_v_f16m8(_v, vl);
    _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(_v, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
    // _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(_v, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
    return _reciprocal;
}

#endif // RVV_MATHFUN_FP16S_H
