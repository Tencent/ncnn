// Copyright 2025 AtomAlpaca <atal@anche.no>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LASX_MATHFUN_H
#define LASX_MATHFUN_H

#include "loongarch_usability.h"

#include <lasxintrin.h>

_LOONGARCH_FLOAT_CONST_PS256(c_0, 0.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_1, 1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_2, 2.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_3, 3.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_4, 4.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_n1, -1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_n3, -3.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_0p5, 0.5f);
_LOONGARCH_FLOAT_CONST_PS256(c_eps, 1E-8f);

#define c_inv_mant_mask ~0x7f800000u
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_SQRTHF, 0.707106781186547524);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p0, 7.0376836292E-2);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p1, -1.1514610310E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p2, 1.1676998740E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p3, -1.2420140846E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p4, +1.4249322787E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p5, -1.6668057665E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p6, +2.0000714765E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p7, -2.4999993993E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_p8, +3.3333331174E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_q1, -2.12194440e-4);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_log_q2, 0.693359375);

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline __m256 log256_ps(__m256 x)
{
    __m256 one = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i);

    x = __lasx_xvfmax_s(x, (__m256)__lasx_xvreplgr2vr_w(0)); /* force flush to zero on denormal values */
    __m256i invalid_mask = __lasx_xvfcmp_cle_s(x, (__m256)__lasx_xvreplgr2vr_w(0));

    __m256i ux = (__m256i)(x);

    __m256i emm0 = __lasx_xvsrl_w(ux, __lasx_xvreplgr2vr_w(23));

    /* keep only the fractional part */
    ux = __lasx_xvand_v(ux, __lasx_xvreplgr2vr_w(c_inv_mant_mask));
    ux = __lasx_xvor_v(ux, __lasx_xvreplgr2vr_w(_ps256_c_0p5.i));
    x = (__m256)(ux);

    emm0 = __lasx_xvsub_w(emm0, __lasx_xvreplgr2vr_w(0x7f));
    __m256 e = __lasx_xvffint_s_w(emm0);

    e = __lasx_xvfadd_s(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    __m256i mask = __lasx_xvfcmp_clt_s((__m256)x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_SQRTHF.i));
    __m256 tmp = (__m256)(__lasx_xvand_v((__m256i)(x), (__m256i)mask));
    x = __lasx_xvfsub_s(x, one);
    e = __lasx_xvfsub_s(e, (__m256)(__lasx_xvand_v((__m256i)(one), (__m256i)mask)));
    x = __lasx_xvfadd_s(x, tmp);

    __m256 z = __lasx_xvfmul_s(x, x);

    __m256 y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p0.i);

    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p1.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p2.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p3.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p4.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p5.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p6.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p7.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_p8.i));
    y = __lasx_xvfmul_s(y, x);

    y = __lasx_xvfmul_s(y, z);

    tmp = __lasx_xvfmul_s(e, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_q1.i));
    y = __lasx_xvfadd_s(y, tmp);

    tmp = __lasx_xvfmul_s(z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));
    y = __lasx_xvfsub_s(y, tmp);

    tmp = __lasx_xvfmul_s(e, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_log_q2.i));
    x = __lasx_xvfadd_s(x, y);
    x = __lasx_xvfadd_s(x, tmp);
    x = (__m256)(__lasx_xvor_v((__m256i)(x), (__m256i)invalid_mask)); // negative arg will be NAN
    return x;
}

_LOONGARCH_FLOAT_CONST_PS256(c_exp_hi, 88.3762626647949f);
_LOONGARCH_FLOAT_CONST_PS256(c_exp_lo, -88.3762626647949f);

_LOONGARCH_FLOAT_CONST_PS256(c_cephes_LOG2EF, 1.44269504088896341);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_C1, 0.693359375);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_C2, -2.12194440e-4);

_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p0, 1.9875691500E-4);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p1, 1.3981999507E-3);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p2, 8.3334519073E-3);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p3, 4.1665795894E-2);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p4, 1.6666665459E-1);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_exp_p5, 5.0000001201E-1);

/* exp() computed for 4 float at once */
static inline __m256 exp256_ps(__m256 x)
{
    __m256 tmp, fx;

    __m256 one = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i);
    x = __lasx_xvfmin_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_exp_hi.i));
    x = __lasx_xvfmax_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_exp_lo.i));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __lasx_xvfmul_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_LOG2EF.i));
    fx = __lasx_xvfadd_s(fx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));

    /* perform a floorf */
    tmp = __lasx_xvffint_s_w(__lasx_xvftint_w_s(fx));

    /* if greater, substract 1 */
    __m256i mask = __lasx_xvfcmp_clt_s(fx, tmp);
    mask = __lasx_xvand_v(mask, (__m256i)one);

    fx = __lasx_xvfsub_s(tmp, (__m256)mask);

    tmp = __lasx_xvfmul_s(fx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_C1.i));
    __m256 z = __lasx_xvfmul_s(fx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_C2.i));
    x = __lasx_xvfsub_s(x, tmp);
    x = __lasx_xvfsub_s(x, z);

    __m256 y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p0.i);

    z = __lasx_xvfmul_s(x, x);

    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p1.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p2.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p3.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p4.i));
    y = __lasx_xvfmadd_s(x, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_exp_p5.i));

    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfadd_s(y, x);
    y = __lasx_xvfadd_s(y, one);

    /* build 2^n */
    __m256i mm;
    mm = __lasx_xvftintrz_w_s(fx);
    mm = __lasx_xvadd_w(mm, __lasx_xvreplgr2vr_w(0x7f));
    mm = __lasx_xvsll_w(mm, __lasx_xvreplgr2vr_w(23));

    y = __lasx_xvfmul_s(y, (__m256)mm);
    return y;
}

_LOONGARCH_FLOAT_CONST_PS256(c_minus_cephes_DP1, -0.78515625f);
_LOONGARCH_FLOAT_CONST_PS256(c_minus_cephes_DP2, -2.4187564849853515625e-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_minus_cephes_DP3, -3.77489497744594108e-8f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_sin_p0, -1.9515295891E-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_sin_p1, 8.3321608736E-3f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_sin_p2, -1.6666654611E-1f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_cos_p0, 2.443315711809948E-005f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_cos_p1, -1.388731625493765E-003f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_cos_p2, 4.166664568298827E-002f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_FOPI, 1.27323954473516f); // 4/PI

static inline __m256 sin256_ps(__m256 x)
{
    __m256 y;
    __m256i swap_sign_bit, poly_mask, sign_bit;
    __m256 n0p5 = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_n1.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));

    sign_bit = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    y = __lasx_xvfmul_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_FOPI.i));

    poly_mask = __lasx_xvftintrz_w_s(y);
    poly_mask = __lasx_xvadd_w(poly_mask, __lasx_xvreplgr2vr_w(1));
    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(~1));
    y = __lasx_xvffint_s_w(poly_mask);

    swap_sign_bit = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(4));
    swap_sign_bit = __lasx_xvslli_w(swap_sign_bit, 29);

    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(2));
    poly_mask = __lasx_xvseq_w(poly_mask, __lasx_xvreplgr2vr_w(0));

    sign_bit = __lasx_xvxor_v(sign_bit, swap_sign_bit);

    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP1.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP2.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP3.i), x);

    y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p0.i);
    __m256 z = __lasx_xvfmul_s(x, x);
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p1.i));
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p2.i));
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmadd_s(z, n0p5, y);
    y = __lasx_xvfadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i));

    __m256 y2 = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p0.i);
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p1.i));
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p2.i));
    y2 = __lasx_xvfmul_s(y2, z);
    y2 = __lasx_xvfmadd_s(y2, x, x);

    y2 = (__m256)__lasx_xvand_v((__m256i)y2, poly_mask);
    y = (__m256)__lasx_xvand_v(__lasx_xvxor_v(poly_mask, __lasx_xvreplgr2vr_w(0xffffffff)), (__m256i)y);
    y = __lasx_xvfadd_s(y, y2);
    y = (__m256)__lasx_xvxor_v((__m256i)y, sign_bit);

    return y;
}

static inline __m256 cos256_ps(__m256 x)
{
    __m256 y;
    __m256i swap_sign_bit, poly_mask, sign_bit;
    __m256 n0p5 = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_n1.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));

    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    y = __lasx_xvfmul_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_FOPI.i));

    poly_mask = __lasx_xvftintrz_w_s(y);
    poly_mask = __lasx_xvadd_w(poly_mask, __lasx_xvreplgr2vr_w(1));
    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(~1));
    y = __lasx_xvffint_s_w(poly_mask);
    poly_mask = __lasx_xvsub_w(poly_mask, __lasx_xvreplgr2vr_w(2));

    swap_sign_bit = __lasx_xvandn_v(poly_mask, __lasx_xvreplgr2vr_w(4));
    swap_sign_bit = __lasx_xvslli_w(swap_sign_bit, 29);

    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(2));
    poly_mask = __lasx_xvseq_w(poly_mask, __lasx_xvreplgr2vr_w(0));

    sign_bit = swap_sign_bit;

    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP1.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP2.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP3.i), x);

    y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p0.i);
    __m256 z = __lasx_xvfmul_s(x, x);
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p1.i));
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p2.i));
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmadd_s(z, n0p5, y);
    y = __lasx_xvfadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i));

    __m256 y2 = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p0.i);
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p1.i));
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p2.i));
    y2 = __lasx_xvfmul_s(y2, z);
    y2 = __lasx_xvfmadd_s(y2, x, x);

    y2 = (__m256)__lasx_xvand_v((__m256i)y2, poly_mask);
    y = (__m256)__lasx_xvandn_v(poly_mask, (__m256i)y);
    y = __lasx_xvfadd_s(y, y2);
    y = (__m256)__lasx_xvxor_v((__m256i)y, sign_bit);

    return y;
}

static inline void sincos256_ps(__m256 x, __m256* s, __m256* c)
{
    __m256 y;
    __m256i swap_sign_bit_cos, swap_sign_bit_sin, poly_mask, sign_bit_sin, sign_bit_cos;
    __m256 n0p5 = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_n1.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));

    sign_bit_sin = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    y = __lasx_xvfmul_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_FOPI.i));

    poly_mask = __lasx_xvftintrz_w_s(y);
    poly_mask = __lasx_xvadd_w(poly_mask, __lasx_xvreplgr2vr_w(1));
    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(~1));
    y = __lasx_xvffint_s_w(poly_mask);

    swap_sign_bit_cos = __lasx_xvsub_w(poly_mask, __lasx_xvreplgr2vr_w(2));
    swap_sign_bit_cos = __lasx_xvandn_v(swap_sign_bit_cos, __lasx_xvreplgr2vr_w(4));
    swap_sign_bit_cos = __lasx_xvslli_w(swap_sign_bit_cos, 29);

    swap_sign_bit_sin = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(4));
    swap_sign_bit_sin = __lasx_xvslli_w(swap_sign_bit_sin, 29);

    poly_mask = __lasx_xvand_v(poly_mask, __lasx_xvreplgr2vr_w(2));
    poly_mask = __lasx_xvseq_w(poly_mask, __lasx_xvreplgr2vr_w(0));

    sign_bit_sin = __lasx_xvxor_v(sign_bit_sin, swap_sign_bit_sin);
    sign_bit_cos = swap_sign_bit_cos;

    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP1.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP2.i), x);
    x = __lasx_xvfmadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_minus_cephes_DP3.i), x);

    __m256 z = __lasx_xvfmul_s(x, x);
    y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p0.i);
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p1.i));
    y = __lasx_xvfmadd_s(y, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_cos_p2.i));
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmul_s(y, z);
    y = __lasx_xvfmadd_s(z, n0p5, y);
    y = __lasx_xvfadd_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i));

    __m256 y2 = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p0.i);
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p1.i));
    y2 = __lasx_xvfmadd_s(y2, z, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_sin_p2.i));
    y2 = __lasx_xvfmul_s(y2, z);
    y2 = __lasx_xvfmadd_s(y2, x, x);

    __m256 ysin1 = (__m256)__lasx_xvandn_v(poly_mask, (__m256i)y);
    __m256 ysin2 = (__m256)__lasx_xvand_v(poly_mask, (__m256i)y2);
    y2 = __lasx_xvfsub_s(y2, ysin2);
    y = __lasx_xvfsub_s(y, ysin1);

    ysin1 = __lasx_xvfadd_s(ysin1, ysin2);
    y = __lasx_xvfadd_s(y, y2);

    *s = (__m256)__lasx_xvxor_v((__m256i)ysin1, sign_bit_sin);
    *c = (__m256)__lasx_xvxor_v((__m256i)y, sign_bit_cos);
}

static inline __m256 tan256_ps(__m256 x)
{
    __m256 ysin, ycos;
    __m256 eps = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_eps.i);
    __m256 zero = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0.i);
    sincos256_ps(x, &ysin, &ycos);
    __m256i mask = __lasx_xvfcmp_ceq_s(ycos, eps);
    mask = __lasx_xvand_v(mask, (__m256i)eps);
    ycos = __lasx_xvfadd_s(ycos, (__m256)mask);
    __m256 ytan = __lasx_xvfdiv_s(ysin, ycos);
    return ytan;
}

_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a4, 0.023994016f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a5, 0.042417344f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a2, 0.07494697f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a3, 0.045520633f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a0, 1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_a1, 0.166667819f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_half_pi, 1.5707964f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_pi, 3.1415927f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_asin_npi, -3.1415927f);

static inline __m256 asin256_ps(__m256 x)
{
    __m256 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    __m256 is_big_input_one, output_approx, final_approx;
    __m256 tmp1, tmp2, tmp3, tmp4;
    __m256i mask, is_small_input, is_big_input;

    mask = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    is_small_input = __lasx_xvfcmp_cle_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));
    is_big_input = __lasx_xvxor_v(is_small_input, __lasx_xvreplgr2vr_w(0xffffffff));
    is_big_input_one = (__m256)__lasx_xvand_v(__lasx_xvreplgr2vr_w(_ps256_c_1.i), is_big_input);

    big_input_approx = __lasx_xvfsub_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i), x);
    big_input_approx = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i), big_input_approx);
    big_input_approx = __lasx_xvfsqrt_s(big_input_approx);

    input_approx = (__m256)__lasx_xvand_v(is_small_input, (__m256i)x);
    input_approx = (__m256)__lasx_xvor_v((__m256i)input_approx, __lasx_xvand_v(is_big_input, (__m256i)big_input_approx));

    square_of_input_approx = __lasx_xvfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lasx_xvfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a4.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a2.i));
    tmp2 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a5.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a3.i));
    tmp3 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp1, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a0.i));
    tmp4 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp2, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a1.i));
    output_approx = __lasx_xvfmadd_s(square_of_input_approx, tmp4, tmp3);

    tmp1 = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_half_pi.i), is_big_input_one);
    tmp2 = __lasx_xvfmul_s(output_approx, input_approx);
    tmp3 = __lasx_xvfmadd_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_n3.i), is_big_input_one, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i));

    final_approx = __lasx_xvfmadd_s(tmp2, tmp3, tmp1);
    final_approx = (__m256)__lasx_xvor_v((__m256i)final_approx, mask);

    return final_approx;
}

static inline __m256 acos256_ps(__m256 x)
{
    __m256 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    __m256 output_approx, final_approx, small_final_approx, big_final_approx;
    __m256 tmp1, tmp2, tmp3, tmp4;
    __m256i mask, mask2, is_small_input, is_big_input, lt_zero;

    lt_zero = __lasx_xvfcmp_clt_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0.i));
    mask = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    is_small_input = __lasx_xvfcmp_cle_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i));
    is_big_input = __lasx_xvxor_v(is_small_input, __lasx_xvreplgr2vr_w(0xffffffff));

    big_input_approx = __lasx_xvfsub_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i), x);
    big_input_approx = __lasx_xvfmul_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_0p5.i), big_input_approx);
    big_input_approx = __lasx_xvfsqrt_s(big_input_approx);

    input_approx = (__m256)__lasx_xvand_v(is_small_input, (__m256i)x);
    input_approx = (__m256)__lasx_xvor_v((__m256i)input_approx, __lasx_xvand_v(is_big_input, (__m256i)big_input_approx));

    square_of_input_approx = __lasx_xvfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lasx_xvfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a4.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a2.i));
    tmp2 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a5.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a3.i));
    tmp3 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp1, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a0.i));
    tmp4 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp2, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_a1.i));
    output_approx = __lasx_xvfmadd_s(square_of_input_approx, tmp4, tmp3);

    tmp1 = __lasx_xvfmul_s(input_approx, output_approx);

    small_final_approx = (__m256)__lasx_xvor_v((__m256i)tmp1, mask);
    small_final_approx = __lasx_xvfsub_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_half_pi.i), small_final_approx);

    big_final_approx = (__m256)__lasx_xvand_v(lt_zero, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_pi.i));
    tmp1 = __lasx_xvfadd_s(tmp1, tmp1);
    tmp1 = (__m256)__lasx_xvor_v((__m256i)tmp1, mask);
    big_final_approx = __lasx_xvfadd_s(big_final_approx, tmp1);

    final_approx = (__m256)__lasx_xvand_v(is_small_input, (__m256i)small_final_approx);
    final_approx = (__m256)__lasx_xvor_v((__m256i)final_approx, __lasx_xvand_v(is_big_input, (__m256i)big_final_approx));

    return final_approx;
}

_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x0, 1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x1, -0.33333072f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x2, 0.1999262f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x3, -0.14203644f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x4, 0.10640934f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x5, -0.07504295f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x6, 0.04269152f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x7, -0.01606863f);
_LOONGARCH_FLOAT_CONST_PS256(c_cephes_atan_x8, 0.0028498897f);

static inline __m256 atan256_ps(__m256 x)
{
    __m256i mask, is_small_input, is_big_input;
    __m256 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, input_approx, output_approx;
    __m256 square_of_input_approx, fourth_power_of_input_approx;

    mask = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    x = (__m256)__lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x7fffffff));

    is_small_input = __lasx_xvfcmp_clt_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_1.i), x);
    is_big_input = __lasx_xvxor_v(is_small_input, __lasx_xvreplgr2vr_w(0xffffffff));

    tmp1 = (__m256)__lasx_xvand_v(is_small_input, __lasx_xvreplgr2vr_w(_ps256_c_n1.i));
    tmp1 = (__m256)__lasx_xvor_v(__lasx_xvand_v(is_big_input, (__m256i)x), (__m256i)tmp1);

    tmp2 = (__m256)__lasx_xvand_v(is_small_input, (__m256i)x);
    tmp2 = (__m256)__lasx_xvor_v(__lasx_xvand_v((__m256i)is_big_input, __lasx_xvreplgr2vr_w(_ps256_c_1.i)), (__m256i)tmp2);

    input_approx = __lasx_xvfdiv_s(tmp1, tmp2);
    square_of_input_approx = __lasx_xvfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lasx_xvfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x7.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x5.i));
    tmp2 = __lasx_xvfmadd_s(fourth_power_of_input_approx, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x8.i), (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x6.i));
    tmp3 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp1, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x3.i));
    tmp4 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp2, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x4.i));
    tmp5 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp3, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x1.i));
    tmp6 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp4, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x2.i));
    tmp7 = __lasx_xvfmadd_s(fourth_power_of_input_approx, tmp6, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_cephes_atan_x0.i));
    output_approx = __lasx_xvfmadd_s(square_of_input_approx, tmp5, tmp7);

    tmp1 = __lasx_xvfmul_s(input_approx, output_approx);
    tmp2 = (__m256)__lasx_xvand_v(is_small_input, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_half_pi.i));
    tmp1 = __lasx_xvfadd_s(tmp1, tmp2);
    tmp1 = (__m256)__lasx_xvxor_v(mask, (__m256i)tmp1);
    return tmp1;
}

static inline __m256 atan2256_ps(__m256 y, __m256 x)
{
    __m256i not_eq_zero_x, not_eq_zero_y, normal_mode, negative_mask_x, negative_mask_y;
    __m256i lt_zero_mask_x, lt_zero_mask_y, ge_zero_mask_y, eq_zero_y;
    __m256 pi_additions, tmp1, tmp2, normal_result, special_result, final_result;

    not_eq_zero_x = __lasx_xvfcmp_cne_s(x, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0.i));
    not_eq_zero_y = __lasx_xvfcmp_cne_s(y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_0.i));
    eq_zero_y = __lasx_xvxor_v(not_eq_zero_y, __lasx_xvreplgr2vr_w(0xffffffff));
    normal_mode = __lasx_xvand_v(not_eq_zero_x, not_eq_zero_y);
    negative_mask_x = __lasx_xvand_v((__m256i)x, __lasx_xvreplgr2vr_w(0x80000000));
    negative_mask_y = __lasx_xvand_v((__m256i)y, __lasx_xvreplgr2vr_w(0x80000000));

    lt_zero_mask_x = __lasx_xvfcmp_clt_s(x, (__m256)__lasx_xvreplgr2vr_w(0));
    lt_zero_mask_y = __lasx_xvfcmp_clt_s(y, (__m256)__lasx_xvreplgr2vr_w(0));
    ge_zero_mask_y = __lasx_xvxor_v(lt_zero_mask_y, __lasx_xvreplgr2vr_w(0xffffffff));

    pi_additions = (__m256)__lasx_xvand_v(lt_zero_mask_y, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_npi.i));
    pi_additions = (__m256)__lasx_xvor_v(__lasx_xvand_v(ge_zero_mask_y, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_pi.i)), (__m256i)pi_additions);
    pi_additions = (__m256)__lasx_xvand_v(lt_zero_mask_x, (__m256i)pi_additions);

    normal_result = __lasx_xvfdiv_s(y, x);
    normal_result = __lasx_xvfadd_s(atan256_ps(normal_result), pi_additions);

    tmp1 = (__m256)__lasx_xvand_v(negative_mask_y, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_half_pi.i));
    tmp2 = (__m256)__lasx_xvand_v(negative_mask_x, __lasx_xvreplgr2vr_w(_ps256_c_cephes_asin_pi.i));
    special_result = (__m256)__lasx_xvand_v(not_eq_zero_y, (__m256i)tmp1);
    special_result = (__m256)__lasx_xvor_v(__lasx_xvand_v(eq_zero_y, (__m256i)tmp2), (__m256i)special_result);

    final_result = (__m256)__lasx_xvand_v(normal_mode, (__m256i)normal_result);
    normal_mode = __lasx_xvxor_v(normal_mode, __lasx_xvreplgr2vr_w(0xffffffff));
    final_result = (__m256)__lasx_xvor_v(__lasx_xvand_v(normal_mode, (__m256i)special_result), (__m256i)final_result);

    return final_result;
}

_LOONGARCH_FLOAT_CONST_PS256(c_tanh_tiny, 1e-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_hi, 9.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_1, 4.89352455891786e-3f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_3, 6.37261928875436e-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_5, 1.48572235717979e-5f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_7, 5.12229709037114e-8f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_9, -8.60467152213735e-11f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_11, 2.00018790482477e-13f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_alpha_13, -2.76076847742355e-16f);
// The monomial coefficients of the denominator polynomial (even).
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_beta_0, 4.89352518554385e-3f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_beta_2, 2.26843463243900e-3f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_beta_4, 1.18534705686654e-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_beta_6, 1.19825839466702e-6f);

/* tanh() computed for 4 float at once */
static inline __m256 tanh256_ps(__m256 x)
{
    __m256 x2 = (__m256)__lasx_xvbitclri_w((__m256i)x, 31);
    __m256i tiny_mask = __lasx_xvfcmp_clt_s((__m256)x2, (__m256)(__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_tiny.i));
    __m256i sig_mask = __lasx_xvreplgr2vr_w(1 << 31);
    __m256i sig_save = __lasx_xvand_v((__m256i)x, sig_mask);

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = (__m256)__lasx_xvbitsel_v((__m256i)x2, (__m256i)__lasx_xvreplgr2vr_w(_ps256_c_tanh_hi.i), (__m256i)__lasx_xvfcmp_clt_s((__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_hi.i), (__m256)x2));

    // since the polynomials are odd/even, we need x**2.
    __m256 z = __lasx_xvfmul_s(x2, x2);

    // evaluate the numerator polynomial y.
    __m256 y = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_13.i);
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_11.i));
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_9.i));
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_7.i));
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_5.i));
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_3.i));
    y = __lasx_xvfmadd_s(z, y, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_alpha_1.i));
    y = __lasx_xvfmul_s(y, x2);

    // evaluate the denominator polynomial w.
    __m256 w = (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_beta_6.i);
    w = __lasx_xvfmadd_s(z, w, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_beta_4.i));
    w = __lasx_xvfmadd_s(z, w, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_beta_2.i));
    w = __lasx_xvfmadd_s(z, w, (__m256)__lasx_xvreplgr2vr_w(_ps256_c_tanh_beta_0.i));

    // divide the numerator by the denominator.
    y = __lasx_xvfdiv_s(y, w);

    // reinstate the sign.
    y = (__m256)__lasx_xvor_v((__m256i)y, sig_save);

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = (__m256)__lasx_xvbitsel_v((__m256i)y, (__m256i)x, (__m256i)tiny_mask);

    return y;
}

static inline __m256 pow256_ps(__m256 a, __m256 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp256_ps(__lasx_xvfmul_s(b, log256_ps(a)));
}

static inline __m256 sigmoid256_ps(__m256 _v)
{
    __m256 _one = __lasx_xvreplfr2vr_s(1.f);
    _v = (__m256)__lasx_xvbitrevi_w((__m256i)_v, 31);
    _v = exp256_ps(_v);
    _v = __lasx_xvfadd_s(_v, _one);
    return __lasx_xvfdiv_s(_one, _v);
}

#endif // LASX_MATHFUN_H
