/* LOONGARCH implementation of mathfun
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 *   Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
 */

/*
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

#ifndef LSX_MATHFUN_H
#define LSX_MATHFUN_H

#include "loongarch_usability.h"

#include <lsxintrin.h>

_LOONGARCH_FLOAT_CONST(c_0, 0.0f);
_LOONGARCH_FLOAT_CONST(c_1, 1.0f);
_LOONGARCH_FLOAT_CONST(c_2, 2.0f);
_LOONGARCH_FLOAT_CONST(c_3, 3.0f);
_LOONGARCH_FLOAT_CONST(c_4, 4.0f);
_LOONGARCH_FLOAT_CONST(c_n1, -1.0f);
_LOONGARCH_FLOAT_CONST(c_n3, -3.0f);
_LOONGARCH_FLOAT_CONST(c_0p5, 0.5f);
_LOONGARCH_FLOAT_CONST(c_eps, 1E-8f);

#define c_inv_mant_mask ~0x7f800000u
_LOONGARCH_FLOAT_CONST(c_cephes_SQRTHF, 0.707106781186547524);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p0, 7.0376836292E-2);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p1, -1.1514610310E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p2, 1.1676998740E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p3, -1.2420140846E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p4, +1.4249322787E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p5, -1.6668057665E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p6, +2.0000714765E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p7, -2.4999993993E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_p8, +3.3333331174E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_log_q1, -2.12194440e-4);
_LOONGARCH_FLOAT_CONST(c_cephes_log_q2, 0.693359375);

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline __m128 log_ps(__m128 x)
{
    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);

    x = __lsx_vfmax_s(x, (__m128)__lsx_vreplgr2vr_w(0)); /* force flush to zero on denormal values */
    __m128i invalid_mask = __lsx_vfcmp_cle_s(x, (__m128)__lsx_vreplgr2vr_w(0));

    __m128i ux = (__m128i)(x);

    __m128i emm0 = __lsx_vsrl_w(ux, __lsx_vreplgr2vr_w(23));

    /* keep only the fractional part */
    ux = __lsx_vand_v(ux, __lsx_vreplgr2vr_w(c_inv_mant_mask));
    ux = __lsx_vor_v(ux, __lsx_vreplgr2vr_w(c_0p5.i));
    x = (__m128)(ux);

    emm0 = __lsx_vsub_w(emm0, __lsx_vreplgr2vr_w(0x7f));
    __m128 e = __lsx_vffint_s_w(emm0);

    e = __lsx_vfadd_s(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    __m128i mask = __lsx_vfcmp_clt_s((__m128)x, (__m128)__lsx_vreplgr2vr_w(c_cephes_SQRTHF.i));
    __m128 tmp = (__m128)(__lsx_vand_v((__m128i)(x), (__m128i)mask));
    x = __lsx_vfsub_s(x, one);
    e = __lsx_vfsub_s(e, (__m128)(__lsx_vand_v((__m128i)(one), (__m128i)mask)));
    x = __lsx_vfadd_s(x, tmp);

    __m128 z = __lsx_vfmul_s(x, x);

    __m128 y = (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p0.i);

    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p1.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p2.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p3.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p4.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p5.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p6.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p7.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_p8.i));
    y = __lsx_vfmul_s(y, x);

    y = __lsx_vfmul_s(y, z);

    tmp = __lsx_vfmul_s(e, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_q1.i));
    y = __lsx_vfadd_s(y, tmp);

    tmp = __lsx_vfmul_s(z, (__m128)__lsx_vreplgr2vr_w(c_0p5.i));
    y = __lsx_vfsub_s(y, tmp);

    tmp = __lsx_vfmul_s(e, (__m128)__lsx_vreplgr2vr_w(c_cephes_log_q2.i));
    x = __lsx_vfadd_s(x, y);
    x = __lsx_vfadd_s(x, tmp);
    x = (__m128)(__lsx_vor_v((__m128i)(x), (__m128i)invalid_mask)); // negative arg will be NAN
    return x;
}

_LOONGARCH_FLOAT_CONST(c_exp_hi, 88.3762626647949f);
_LOONGARCH_FLOAT_CONST(c_exp_lo, -88.3762626647949f);

_LOONGARCH_FLOAT_CONST(c_cephes_LOG2EF, 1.44269504088896341);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_C1, 0.693359375);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_C2, -2.12194440e-4);

_LOONGARCH_FLOAT_CONST(c_cephes_exp_p0, 1.9875691500E-4);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_p1, 1.3981999507E-3);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_p2, 8.3334519073E-3);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_p3, 4.1665795894E-2);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_p4, 1.6666665459E-1);
_LOONGARCH_FLOAT_CONST(c_cephes_exp_p5, 5.0000001201E-1);

/* exp() computed for 4 float at once */
static inline __m128 exp_ps(__m128 x)
{
    __m128 tmp, fx;

    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    x = __lsx_vfmin_s(x, (__m128)__lsx_vreplgr2vr_w(c_exp_hi.i));
    x = __lsx_vfmax_s(x, (__m128)__lsx_vreplgr2vr_w(c_exp_lo.i));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __lsx_vfmul_s(x, (__m128)__lsx_vreplgr2vr_w(c_cephes_LOG2EF.i));
    fx = __lsx_vfadd_s(fx, (__m128)__lsx_vreplgr2vr_w(c_0p5.i));

    /* perform a floorf */
    tmp = __lsx_vffint_s_w(__lsx_vftint_w_s(fx));

    /* if greater, substract 1 */
    __m128i mask = __lsx_vfcmp_clt_s(fx, tmp);
    mask = __lsx_vand_v(mask, (__m128i)one);

    fx = __lsx_vfsub_s(tmp, (__m128)mask);

    tmp = __lsx_vfmul_s(fx, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_C1.i));
    __m128 z = __lsx_vfmul_s(fx, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_C2.i));
    x = __lsx_vfsub_s(x, tmp);
    x = __lsx_vfsub_s(x, z);

    __m128 y = (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p0.i);

    z = __lsx_vfmul_s(x, x);

    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p1.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p2.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p3.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p4.i));
    y = __lsx_vfmadd_s(x, y, (__m128)__lsx_vreplgr2vr_w(c_cephes_exp_p5.i));

    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfadd_s(y, x);
    y = __lsx_vfadd_s(y, one);

    /* build 2^n */
    __m128i mm;
    mm = __lsx_vftintrz_w_s(fx);
    mm = __lsx_vadd_w(mm, __lsx_vreplgr2vr_w(0x7f));
    mm = __lsx_vsll_w(mm, __lsx_vreplgr2vr_w(23));

    y = __lsx_vfmul_s(y, (__m128)mm);
    return y;
}

_LOONGARCH_FLOAT_CONST(c_tanh_tiny, 1e-4f);
_LOONGARCH_FLOAT_CONST(c_tanh_hi, 9.0f);
// The monomial coefficients of the numerator polynomial (odd).
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_1, 4.89352455891786e-3f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_3, 6.37261928875436e-4f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_5, 1.48572235717979e-5f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_7, 5.12229709037114e-8f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_9, -8.60467152213735e-11f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_11, 2.00018790482477e-13f);
_LOONGARCH_FLOAT_CONST(c_tanh_alpha_13, -2.76076847742355e-16f);
// The monomial coefficients of the denominator polynomial (even).
_LOONGARCH_FLOAT_CONST(c_tanh_beta_0, 4.89352518554385e-3f);
_LOONGARCH_FLOAT_CONST(c_tanh_beta_2, 2.26843463243900e-3f);
_LOONGARCH_FLOAT_CONST(c_tanh_beta_4, 1.18534705686654e-4f);
_LOONGARCH_FLOAT_CONST(c_tanh_beta_6, 1.19825839466702e-6f);

/* tanh() computed for 4 float at once */
static inline __m128 tanh_ps(__m128 x)
{
    __m128 x2 = (__m128)__lsx_vbitclri_w((__m128i)x, 31);
    __m128i tiny_mask = __lsx_vfcmp_clt_s((__m128)x2, (__m128)(__m128)__lsx_vreplgr2vr_w(c_tanh_tiny.i));
    __m128i sig_mask = __lsx_vreplgr2vr_w(1 << 31);
    __m128i sig_save = __lsx_vand_v((__m128i)x, sig_mask);

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = (__m128)__lsx_vbitsel_v((__m128i)x2, (__m128i)__lsx_vreplgr2vr_w(c_tanh_hi.i), (__m128i)__lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(c_tanh_hi.i), (__m128)x2));

    // since the polynomials are odd/even, we need x**2.
    __m128 z = __lsx_vfmul_s(x2, x2);

    // evaluate the numerator polynomial y.
    __m128 y = (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_13.i);
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_11.i));
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_9.i));
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_7.i));
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_5.i));
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_3.i));
    y = __lsx_vfmadd_s(z, y, (__m128)__lsx_vreplgr2vr_w(c_tanh_alpha_1.i));
    y = __lsx_vfmul_s(y, x2);

    // evaluate the denominator polynomial w.
    __m128 w = (__m128)__lsx_vreplgr2vr_w(c_tanh_beta_6.i);
    w = __lsx_vfmadd_s(z, w, (__m128)__lsx_vreplgr2vr_w(c_tanh_beta_4.i));
    w = __lsx_vfmadd_s(z, w, (__m128)__lsx_vreplgr2vr_w(c_tanh_beta_2.i));
    w = __lsx_vfmadd_s(z, w, (__m128)__lsx_vreplgr2vr_w(c_tanh_beta_0.i));

    // divide the numerator by the denominator.
    y = __lsx_vfdiv_s(y, w);

    // reinstate the sign.
    y = (__m128)__lsx_vor_v((__m128i)y, sig_save);

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = (__m128)__lsx_vbitsel_v((__m128i)y, (__m128i)x, (__m128i)tiny_mask);

    return y;
}

_LOONGARCH_FLOAT_CONST(c_minus_cephes_DP1, -0.78515625f);
_LOONGARCH_FLOAT_CONST(c_minus_cephes_DP2, -2.4187564849853515625e-4f);
_LOONGARCH_FLOAT_CONST(c_minus_cephes_DP3, -3.77489497744594108e-8f);
_LOONGARCH_FLOAT_CONST(c_cephes_sin_p0, -1.9515295891E-4f);
_LOONGARCH_FLOAT_CONST(c_cephes_sin_p1, 8.3321608736E-3f);
_LOONGARCH_FLOAT_CONST(c_cephes_sin_p2, -1.6666654611E-1f);
_LOONGARCH_FLOAT_CONST(c_cephes_cos_p0, 2.443315711809948E-005f);
_LOONGARCH_FLOAT_CONST(c_cephes_cos_p1, -1.388731625493765E-003f);
_LOONGARCH_FLOAT_CONST(c_cephes_cos_p2, 4.166664568298827E-002f);
_LOONGARCH_FLOAT_CONST(c_cephes_FOPI, 1.27323954473516f); // 4/PI

static inline __m128 sin_ps(__m128 x)
{
    __m128 y;
    __m128i swap_sign_bit, poly_mask, sign_bit;
    __m128 n0p5 = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_n1.i), (__m128)__lsx_vreplgr2vr_w(c_0p5.i));

    sign_bit = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    y = __lsx_vfmul_s(x, (__m128)__lsx_vreplgr2vr_w(c_cephes_FOPI.i));

    poly_mask = __lsx_vftintrz_w_s(y);
    poly_mask = __lsx_vadd_w(poly_mask, __lsx_vreplgr2vr_w(1));
    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(~1));
    y = __lsx_vffint_s_w(poly_mask);

    swap_sign_bit = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(4));
    swap_sign_bit = __lsx_vslli_w(swap_sign_bit, 29);

    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(2));
    poly_mask = __lsx_vseq_w(poly_mask, __lsx_vreplgr2vr_w(0));

    sign_bit = __lsx_vxor_v(sign_bit, swap_sign_bit);

    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP1.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP2.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP3.i), x);

    y = (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p0.i);
    __m128 z = __lsx_vfmul_s(x, x);
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p1.i));
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p2.i));
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmadd_s(z, n0p5, y);
    y = __lsx_vfadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_1.i));

    __m128 y2 = (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p0.i);
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p1.i));
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p2.i));
    y2 = __lsx_vfmul_s(y2, z);
    y2 = __lsx_vfmadd_s(y2, x, x);

    y2 = (__m128)__lsx_vand_v((__m128i)y2, poly_mask);
    y = (__m128)__lsx_vand_v(__lsx_vxor_v(poly_mask, __lsx_vreplgr2vr_w(0xffffffff)), (__m128i)y);
    y = __lsx_vfadd_s(y, y2);
    y = (__m128)__lsx_vxor_v((__m128i)y, sign_bit);

    return y;
}

static inline __m128 cos_ps(__m128 x)
{
    __m128 y;
    __m128i swap_sign_bit, poly_mask, sign_bit;
    __m128 n0p5 = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_n1.i), (__m128)__lsx_vreplgr2vr_w(c_0p5.i));

    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    y = __lsx_vfmul_s(x, (__m128)__lsx_vreplgr2vr_w(c_cephes_FOPI.i));

    poly_mask = __lsx_vftintrz_w_s(y);
    poly_mask = __lsx_vadd_w(poly_mask, __lsx_vreplgr2vr_w(1));
    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(~1));
    y = __lsx_vffint_s_w(poly_mask);
    poly_mask = __lsx_vsub_w(poly_mask, __lsx_vreplgr2vr_w(2));

    swap_sign_bit = __lsx_vandn_v(poly_mask, __lsx_vreplgr2vr_w(4));
    swap_sign_bit = __lsx_vslli_w(swap_sign_bit, 29);

    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(2));
    poly_mask = __lsx_vseq_w(poly_mask, __lsx_vreplgr2vr_w(0));

    sign_bit = swap_sign_bit;

    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP1.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP2.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP3.i), x);

    y = (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p0.i);
    __m128 z = __lsx_vfmul_s(x, x);
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p1.i));
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p2.i));
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmadd_s(z, n0p5, y);
    y = __lsx_vfadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_1.i));

    __m128 y2 = (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p0.i);
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p1.i));
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p2.i));
    y2 = __lsx_vfmul_s(y2, z);
    y2 = __lsx_vfmadd_s(y2, x, x);

    y2 = (__m128)__lsx_vand_v((__m128i)y2, poly_mask);
    y = (__m128)__lsx_vandn_v(poly_mask, (__m128i)y);
    y = __lsx_vfadd_s(y, y2);
    y = (__m128)__lsx_vxor_v((__m128i)y, sign_bit);

    return y;
}

static inline void sincos_ps(__m128 x, __m128* s, __m128* c)
{
    __m128 y;
    __m128i swap_sign_bit_cos, swap_sign_bit_sin, poly_mask, sign_bit_sin, sign_bit_cos;
    __m128 n0p5 = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_n1.i), (__m128)__lsx_vreplgr2vr_w(c_0p5.i));

    sign_bit_sin = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    y = __lsx_vfmul_s(x, (__m128)__lsx_vreplgr2vr_w(c_cephes_FOPI.i));

    poly_mask = __lsx_vftintrz_w_s(y);
    poly_mask = __lsx_vadd_w(poly_mask, __lsx_vreplgr2vr_w(1));
    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(~1));
    y = __lsx_vffint_s_w(poly_mask);

    swap_sign_bit_cos = __lsx_vsub_w(poly_mask, __lsx_vreplgr2vr_w(2));
    swap_sign_bit_cos = __lsx_vandn_v(swap_sign_bit_cos, __lsx_vreplgr2vr_w(4));
    swap_sign_bit_cos = __lsx_vslli_w(swap_sign_bit_cos, 29);

    swap_sign_bit_sin = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(4));
    swap_sign_bit_sin = __lsx_vslli_w(swap_sign_bit_sin, 29);

    poly_mask = __lsx_vand_v(poly_mask, __lsx_vreplgr2vr_w(2));
    poly_mask = __lsx_vseq_w(poly_mask, __lsx_vreplgr2vr_w(0));

    sign_bit_sin = __lsx_vxor_v(sign_bit_sin, swap_sign_bit_sin);
    sign_bit_cos = swap_sign_bit_cos;

    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP1.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP2.i), x);
    x = __lsx_vfmadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_minus_cephes_DP3.i), x);

    __m128 z = __lsx_vfmul_s(x, x);
    y = (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p0.i);
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p1.i));
    y = __lsx_vfmadd_s(y, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_cos_p2.i));
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmul_s(y, z);
    y = __lsx_vfmadd_s(z, n0p5, y);
    y = __lsx_vfadd_s(y, (__m128)__lsx_vreplgr2vr_w(c_1.i));

    __m128 y2 = (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p0.i);
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p1.i));
    y2 = __lsx_vfmadd_s(y2, z, (__m128)__lsx_vreplgr2vr_w(c_cephes_sin_p2.i));
    y2 = __lsx_vfmul_s(y2, z);
    y2 = __lsx_vfmadd_s(y2, x, x);

    __m128 ysin1 = (__m128)__lsx_vandn_v(poly_mask, (__m128i)y);
    __m128 ysin2 = (__m128)__lsx_vand_v(poly_mask, (__m128i)y2);
    y2 = __lsx_vfsub_s(y2, ysin2);
    y = __lsx_vfsub_s(y, ysin1);

    ysin1 = __lsx_vfadd_s(ysin1, ysin2);
    y = __lsx_vfadd_s(y, y2);

    *s = (__m128)__lsx_vxor_v((__m128i)ysin1, sign_bit_sin);
    *c = (__m128)__lsx_vxor_v((__m128i)y, sign_bit_cos);
}

static inline __m128 tan_ps(__m128 x)
{
    __m128 ysin, ycos;
    __m128 eps = (__m128)__lsx_vreplgr2vr_w(c_eps.i);
    __m128 zero = (__m128)__lsx_vreplgr2vr_w(c_0.i);
    sincos_ps(x, &ysin, &ycos);
    __m128i mask = __lsx_vfcmp_ceq_s(ycos, eps);
    mask = __lsx_vand_v(mask, (__m128i)eps);
    ycos = __lsx_vfadd_s(ycos, (__m128)mask);
    __m128 ytan = __lsx_vfdiv_s(ysin, ycos);
    return ytan;
}

static inline __m128 pow_ps(__m128 a, __m128 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp_ps(__lsx_vfmul_s(b, log_ps(a)));
}

static inline __m128 sigmoid_ps(__m128 _v)
{
    __m128 _one = __lsx_vreplfr2vr_s(1.f);
    _v = (__m128)__lsx_vbitrevi_w((__m128i)_v, 31);
    _v = exp_ps(_v);
    _v = __lsx_vfadd_s(_v, _one);
    return __lsx_vfdiv_s(_one, _v);
}

_LOONGARCH_FLOAT_CONST(c_cephes_asin_a4, 0.023994016f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_a5, 0.042417344f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_a2, 0.07494697f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_a3, 0.045520633f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_a0, 1.0f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_a1, 0.166667819f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_half_pi, 1.5707964f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_pi, 3.1415927f);
_LOONGARCH_FLOAT_CONST(c_cephes_asin_npi, -3.1415927f);

static inline __m128 asin_ps(__m128 x)
{
    __m128 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    __m128 is_big_input_one, output_approx, final_approx;
    __m128 tmp1, tmp2, tmp3, tmp4;
    __m128i mask, is_small_input, is_big_input;

    mask = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    is_small_input = __lsx_vfcmp_cle_s(x, (__m128)__lsx_vreplgr2vr_w(c_0p5.i));
    is_big_input = __lsx_vxor_v(is_small_input, __lsx_vreplgr2vr_w(0xffffffff));
    is_big_input_one = (__m128)__lsx_vand_v(__lsx_vreplgr2vr_w(c_1.i), is_big_input);

    big_input_approx = __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_1.i), x);
    big_input_approx = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_0p5.i), big_input_approx);
    big_input_approx = __lsx_vfsqrt_s(big_input_approx);

    input_approx = (__m128)__lsx_vand_v(is_small_input, (__m128i)x);
    input_approx = (__m128)__lsx_vor_v((__m128i)input_approx, __lsx_vand_v(is_big_input, (__m128i)big_input_approx));

    square_of_input_approx = __lsx_vfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lsx_vfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a4.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a2.i));
    tmp2 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a5.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a3.i));
    tmp3 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp1, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a0.i));
    tmp4 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp2, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a1.i));
    output_approx = __lsx_vfmadd_s(square_of_input_approx, tmp4, tmp3);

    tmp1 = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_cephes_asin_half_pi.i), is_big_input_one);
    tmp2 = __lsx_vfmul_s(output_approx, input_approx);
    tmp3 = __lsx_vfmadd_s((__m128)__lsx_vreplgr2vr_w(c_n3.i), is_big_input_one, (__m128)__lsx_vreplgr2vr_w(c_1.i));

    final_approx = __lsx_vfmadd_s(tmp2, tmp3, tmp1);
    final_approx = (__m128)__lsx_vor_v((__m128i)final_approx, mask);

    return final_approx;
}

static inline __m128 acos_ps(__m128 x)
{
    __m128 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    __m128 output_approx, final_approx, small_final_approx, big_final_approx;
    __m128 tmp1, tmp2, tmp3, tmp4;
    __m128i mask, mask2, is_small_input, is_big_input, lt_zero;

    lt_zero = __lsx_vfcmp_clt_s(x, (__m128)__lsx_vreplgr2vr_w(c_0.i));
    mask = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    is_small_input = __lsx_vfcmp_cle_s(x, (__m128)__lsx_vreplgr2vr_w(c_0p5.i));
    is_big_input = __lsx_vxor_v(is_small_input, __lsx_vreplgr2vr_w(0xffffffff));

    big_input_approx = __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_1.i), x);
    big_input_approx = __lsx_vfmul_s((__m128)__lsx_vreplgr2vr_w(c_0p5.i), big_input_approx);
    big_input_approx = __lsx_vfsqrt_s(big_input_approx);

    input_approx = (__m128)__lsx_vand_v(is_small_input, (__m128i)x);
    input_approx = (__m128)__lsx_vor_v((__m128i)input_approx, __lsx_vand_v(is_big_input, (__m128i)big_input_approx));

    square_of_input_approx = __lsx_vfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lsx_vfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a4.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a2.i));
    tmp2 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a5.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a3.i));
    tmp3 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp1, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a0.i));
    tmp4 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp2, (__m128)__lsx_vreplgr2vr_w(c_cephes_asin_a1.i));
    output_approx = __lsx_vfmadd_s(square_of_input_approx, tmp4, tmp3);

    tmp1 = __lsx_vfmul_s(input_approx, output_approx);

    small_final_approx = (__m128)__lsx_vor_v((__m128i)tmp1, mask);
    small_final_approx = __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_cephes_asin_half_pi.i), small_final_approx);

    big_final_approx = (__m128)__lsx_vand_v(lt_zero, __lsx_vreplgr2vr_w(c_cephes_asin_pi.i));
    tmp1 = __lsx_vfadd_s(tmp1, tmp1);
    tmp1 = (__m128)__lsx_vor_v((__m128i)tmp1, mask);
    big_final_approx = __lsx_vfadd_s(big_final_approx, tmp1);

    final_approx = (__m128)__lsx_vand_v(is_small_input, (__m128i)small_final_approx);
    final_approx = (__m128)__lsx_vor_v((__m128i)final_approx, __lsx_vand_v(is_big_input, (__m128i)big_final_approx));

    return final_approx;
}

_LOONGARCH_FLOAT_CONST(c_cephes_atan_x0, 1.0f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x1, -0.33333072f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x2, 0.1999262f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x3, -0.14203644f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x4, 0.10640934f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x5, -0.07504295f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x6, 0.04269152f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x7, -0.01606863f);
_LOONGARCH_FLOAT_CONST(c_cephes_atan_x8, 0.0028498897f);

static inline __m128 atan_ps(__m128 x)
{
    __m128i mask, is_small_input, is_big_input;
    __m128 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, input_approx, output_approx;
    __m128 square_of_input_approx, fourth_power_of_input_approx;

    mask = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    x = (__m128)__lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x7fffffff));

    is_small_input = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(c_1.i), x);
    is_big_input = __lsx_vxor_v(is_small_input, __lsx_vreplgr2vr_w(0xffffffff));

    tmp1 = (__m128)__lsx_vand_v(is_small_input, __lsx_vreplgr2vr_w(c_n1.i));
    tmp1 = (__m128)__lsx_vor_v(__lsx_vand_v(is_big_input, (__m128i)x), (__m128i)tmp1);

    tmp2 = (__m128)__lsx_vand_v(is_small_input, (__m128i)x);
    tmp2 = (__m128)__lsx_vor_v(__lsx_vand_v((__m128i)is_big_input, __lsx_vreplgr2vr_w(c_1.i)), (__m128i)tmp2);

    input_approx = __lsx_vfdiv_s(tmp1, tmp2);
    square_of_input_approx = __lsx_vfmul_s(input_approx, input_approx);
    fourth_power_of_input_approx = __lsx_vfmul_s(square_of_input_approx, square_of_input_approx);

    tmp1 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x7.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x5.i));
    tmp2 = __lsx_vfmadd_s(fourth_power_of_input_approx, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x8.i), (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x6.i));
    tmp3 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp1, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x3.i));
    tmp4 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp2, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x4.i));
    tmp5 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp3, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x1.i));
    tmp6 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp4, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x2.i));
    tmp7 = __lsx_vfmadd_s(fourth_power_of_input_approx, tmp6, (__m128)__lsx_vreplgr2vr_w(c_cephes_atan_x0.i));
    output_approx = __lsx_vfmadd_s(square_of_input_approx, tmp5, tmp7);

    tmp1 = __lsx_vfmul_s(input_approx, output_approx);
    tmp2 = (__m128)__lsx_vand_v(is_small_input, __lsx_vreplgr2vr_w(c_cephes_asin_half_pi.i));
    tmp1 = __lsx_vfadd_s(tmp1, tmp2);
    tmp1 = (__m128)__lsx_vxor_v(mask, (__m128i)tmp1);
    return tmp1;
}

static inline __m128 atan2_ps(__m128 y, __m128 x)
{
    __m128i not_eq_zero_x, not_eq_zero_y, normal_mode, negative_mask_x, negative_mask_y;
    __m128i lt_zero_mask_x, lt_zero_mask_y, ge_zero_mask_y, eq_zero_y;
    __m128 pi_additions, tmp1, tmp2, normal_result, special_result, final_result;

    not_eq_zero_x = __lsx_vfcmp_cne_s(x, (__m128)__lsx_vreplgr2vr_w(c_0.i));
    not_eq_zero_y = __lsx_vfcmp_cne_s(y, (__m128)__lsx_vreplgr2vr_w(c_0.i));
    eq_zero_y = __lsx_vxor_v(not_eq_zero_y, __lsx_vreplgr2vr_w(0xffffffff));
    normal_mode = __lsx_vand_v(not_eq_zero_x, not_eq_zero_y);
    negative_mask_x = __lsx_vand_v((__m128i)x, __lsx_vreplgr2vr_w(0x80000000));
    negative_mask_y = __lsx_vand_v((__m128i)y, __lsx_vreplgr2vr_w(0x80000000));

    lt_zero_mask_x = __lsx_vfcmp_clt_s(x, (__m128)__lsx_vreplgr2vr_w(0));
    lt_zero_mask_y = __lsx_vfcmp_clt_s(y, (__m128)__lsx_vreplgr2vr_w(0));
    ge_zero_mask_y = __lsx_vxor_v(lt_zero_mask_y, __lsx_vreplgr2vr_w(0xffffffff));

    pi_additions = (__m128)__lsx_vand_v(lt_zero_mask_y, __lsx_vreplgr2vr_w(c_cephes_asin_npi.i));
    pi_additions = (__m128)__lsx_vor_v(__lsx_vand_v(ge_zero_mask_y, __lsx_vreplgr2vr_w(c_cephes_asin_pi.i)), (__m128i)pi_additions);
    pi_additions = (__m128)__lsx_vand_v(lt_zero_mask_x, (__m128i)pi_additions);

    normal_result = __lsx_vfdiv_s(y, x);
    normal_result = __lsx_vfadd_s(atan_ps(normal_result), pi_additions);

    tmp1 = (__m128)__lsx_vand_v(negative_mask_y, __lsx_vreplgr2vr_w(c_cephes_asin_half_pi.i));
    tmp2 = (__m128)__lsx_vand_v(negative_mask_x, __lsx_vreplgr2vr_w(c_cephes_asin_pi.i));
    special_result = (__m128)__lsx_vand_v(not_eq_zero_y, (__m128i)tmp1);
    special_result = (__m128)__lsx_vor_v(__lsx_vand_v(eq_zero_y, (__m128i)tmp2), (__m128i)special_result);

    final_result = (__m128)__lsx_vand_v(normal_mode, (__m128i)normal_result);
    normal_mode = __lsx_vxor_v(normal_mode, __lsx_vreplgr2vr_w(0xffffffff));
    final_result = (__m128)__lsx_vor_v(__lsx_vand_v(normal_mode, (__m128i)special_result), (__m128i)final_result);

    return final_result;
}

_LOONGARCH_FLOAT_CONST(c_erf_threshold, 0.927734375f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c0, -1.72853470e-5f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c1, 3.83197126e-4f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c2, -3.88396438e-3f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c3, 2.42546219e-2f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c4, -1.06777877e-1f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c5, -6.34846687e-1f);
_LOONGARCH_FLOAT_CONST(c_erf_hi_c6, -1.28717512e-1f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c0, -5.96761703e-4f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c1, 4.99119423e-3f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c2, -2.67681349e-2f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c3, 1.12819925e-1f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c4, -3.76125336e-1f);
_LOONGARCH_FLOAT_CONST(c_erf_lo_c5, 1.28379166e-1f);

static inline __m128 erf_ps(__m128 a)
{
    __m128 t = (__m128)__lsx_vbitclri_w((__m128i)a, 31);
    __m128 s = __lsx_vfmul_s(a, a);

    __m128i hi_mask = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(c_erf_threshold.i), t);

    __m128 r_hi = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c0.i), (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c1.i));
    __m128 u = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c2.i), (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c3.i));
    r_hi = __lsx_vfmadd_s(r_hi, s, u);
    r_hi = __lsx_vfmadd_s(r_hi, t, (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c4.i));
    r_hi = __lsx_vfmadd_s(r_hi, t, (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c5.i));
    r_hi = __lsx_vfmadd_s(r_hi, t, (__m128)__lsx_vreplgr2vr_w(c_erf_hi_c6.i));
    r_hi = __lsx_vfmadd_s(r_hi, t, __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_0.i), t));
    r_hi = __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_1.i), exp_ps(r_hi));

    __m128 r_lo = (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c0.i);
    r_lo = __lsx_vfmadd_s(r_lo, s, (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c1.i));
    r_lo = __lsx_vfmadd_s(r_lo, s, (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c2.i));
    r_lo = __lsx_vfmadd_s(r_lo, s, (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c3.i));
    r_lo = __lsx_vfmadd_s(r_lo, s, (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c4.i));
    r_lo = __lsx_vfmadd_s(r_lo, s, (__m128)__lsx_vreplgr2vr_w(c_erf_lo_c5.i));
    r_lo = __lsx_vfmadd_s(r_lo, a, a);

    __m128 r = (__m128)__lsx_vbitsel_v((__m128i)r_lo, (__m128i)r_hi, hi_mask);

    __m128i sign_mask = __lsx_vreplgr2vr_w(0x80000000);
    __m128i sign_a = __lsx_vand_v((__m128i)a, sign_mask);
    r = (__m128)__lsx_vor_v((__m128i)r, sign_a);

    return r;
}

static inline __m128 fmod_ps(__m128 a, __m128 b)
{
    // fmod(a,b) = a - trunc(a/b)*b  (trunc toward 0)
    __m128 q = __lsx_vfdiv_s(a, b);
    __m128i qi = __lsx_vftintrz_w_s(q); // float -> int32 trunc toward zero
    __m128 qf = __lsx_vffint_s_w(qi);   // int32 -> float
    return __lsx_vfsub_s(a, __lsx_vfmul_s(qf, b));
}

static inline __m128 round_ps(__m128 x)
{
    __m128 half = (__m128)__lsx_vreplgr2vr_w(c_0p5.i);
    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    __m128i sign_mask = __lsx_vfcmp_clt_s(x, (__m128)__lsx_vreplgr2vr_w(0));
    __m128 abs_x = (__m128)__lsx_vbitclri_w((__m128i)x, 31);
    __m128i xi = __lsx_vftintrz_w_s(abs_x);
    __m128 xf = __lsx_vffint_s_w(xi);
    __m128 diff = __lsx_vfsub_s(abs_x, xf);
    __m128i diff_gt_half = __lsx_vfcmp_clt_s(half, diff);
    __m128i diff_eq_half = __lsx_vfcmp_ceq_s(diff, half);
    __m128i xi_and_1 = __lsx_vand_v(xi, __lsx_vreplgr2vr_w(1));
    __m128i is_odd = __lsx_vseq_w(xi_and_1, __lsx_vreplgr2vr_w(1));
    __m128i round_up = __lsx_vor_v(diff_gt_half, __lsx_vand_v(diff_eq_half, is_odd));
    __m128 rounded = __lsx_vfadd_s(xf, (__m128)__lsx_vand_v(round_up, (__m128i)one));
    return (__m128)__lsx_vbitsel_v((__m128i)rounded, (__m128i)__lsx_vbitrevi_w((__m128i)rounded, 31), sign_mask);
}

static inline __m128 logaddexp_ps(__m128 a, __m128 b)
{
    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    __m128 max_xy = __lsx_vfmax_s(a, b);
    __m128 min_xy = __lsx_vfmin_s(a, b);
    __m128 diff = __lsx_vfsub_s(min_xy, max_xy);
    __m128 exp_diff = exp_ps(diff);
    __m128 one_plus_exp = __lsx_vfadd_s(one, exp_diff);
    __m128 log_result = log_ps(one_plus_exp);
    return __lsx_vfadd_s(max_xy, log_result);
}

static inline __m128 floor_ps(__m128 x)
{
    __m128i xi = __lsx_vftintrz_w_s(x);
    __m128 xf = __lsx_vffint_s_w(xi);
    __m128i need_adjust = __lsx_vfcmp_clt_s(x, xf);
    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    return __lsx_vfsub_s(xf, (__m128)__lsx_vand_v(need_adjust, (__m128i)one));
}

static inline __m128 floor_divide_ps(__m128 a, __m128 b)
{
    __m128 q = __lsx_vfdiv_s(a, b);
    return floor_ps(q);
}

static inline __m128 remainder_ps(__m128 a, __m128 b)
{
    __m128 q = __lsx_vfdiv_s(a, b);
    __m128 rq = round_ps(q);
    return __lsx_vfsub_s(a, __lsx_vfmul_s(rq, b));
}

static inline __m128 elu_ps(__m128 _v, __m128 _alpha)
{
    __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
    __m128 _one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    __m128 _pos = __lsx_vfmax_s(_v, _zero);
    __m128 _neg = __lsx_vfmin_s(_v, _zero);
    _neg = __lsx_vfsub_s(exp_ps(_neg), _one);
    return __lsx_vfadd_s(_pos, __lsx_vfmul_s(_alpha, _neg));
}

_LOONGARCH_FLOAT_CONST(c_erfc_erx, 8.4506291151e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pp0, 1.2837916613e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pp1, -3.2504209876e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pp2, -2.8481749818e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_pp3, -5.7702702470e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_pp4, -2.3763017452e-05f);
_LOONGARCH_FLOAT_CONST(c_erfc_qq1, 3.9791721106e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_qq2, 6.5022252500e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_qq3, 5.0813062117e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_qq4, 1.3249473704e-4f);
_LOONGARCH_FLOAT_CONST(c_erfc_qq5, -3.9602282413e-6f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa0, -2.3621185683e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa1, 4.1485610604e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa2, -3.7220788002e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa3, 3.1834661961e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa4, -1.1089469492e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa5, 3.5478305072e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_pa6, -2.1663755178e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa1, 1.0642088205e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa2, 5.4039794207e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa3, 7.1828655899e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa4, 1.2617121637e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa5, 1.3637083583e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_qa6, 1.1984500103e-2f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra0, -9.8649440333e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra1, -6.9385856390e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra2, -1.0558626175e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra3, -6.2375331879e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra4, -1.6239666748e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra5, -1.8460508728e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra6, -8.1287437439e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_ra7, -9.8143291473e+00f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa1, 1.9651271820e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa2, 1.3765776062e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa3, 4.3456588745e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa4, 6.4538726807e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa5, 4.2900814819e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa6, 1.0863500214e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa7, 6.5702495575e+00f);
_LOONGARCH_FLOAT_CONST(c_erfc_sa8, -6.0424413532e-02f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb0, -9.8649431020e-3f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb1, -7.9928326607e-1f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb2, -1.7757955551e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb3, -1.6063638306e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb4, -6.3756646729e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb5, -1.0250950928e+3f);
_LOONGARCH_FLOAT_CONST(c_erfc_rb6, -4.8351919556e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb1, 3.0338060379e+1f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb2, 3.2579251099e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb3, 1.5367296143e+3f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb4, 3.1998581543e+3f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb5, 2.5530502930e+3f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb6, 4.7452853394e+2f);
_LOONGARCH_FLOAT_CONST(c_erfc_sb7, -2.2440952301e+1f);

static inline __m128 erfc_ps(__m128 x)
{
    __m128 one = (__m128)__lsx_vreplgr2vr_w(c_1.i);
    __m128 two = (__m128)__lsx_vreplgr2vr_w(c_2.i);
    __m128 zero = (__m128)__lsx_vreplgr2vr_w(c_0.i);

    __m128 absx = (__m128)__lsx_vbitclri_w((__m128i)x, 31);
    __m128 x2 = __lsx_vfmul_s(x, x);
    __m128 t = __lsx_vfdiv_s(x2, one);
    __m128 tt = __lsx_vfsub_s(absx, one);

    __m128i mask_ge_1_25 = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(0x3fa00000), absx);
    t = (__m128)__lsx_vbitsel_v((__m128i)tt, (__m128i)t, mask_ge_1_25);

    __m128i mask_ge_0_84375 = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(0x3f580000), absx);
    t = (__m128)__lsx_vbitsel_v((__m128i)x2, (__m128i)t, mask_ge_0_84375);

    __m128 u = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb6.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_rb5.i));
    u = __lsx_vfmadd_s(t, u, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb4.i));
    u = __lsx_vfmadd_s(t, u, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb3.i));
    u = __lsx_vfmadd_s(t, u, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb2.i));
    u = __lsx_vfmadd_s(t, u, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb1.i));
    u = __lsx_vfmadd_s(t, u, (__m128)__lsx_vreplgr2vr_w(c_erfc_rb0.i));

    __m128 v = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb7.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_sb6.i));
    v = __lsx_vfmadd_s(t, v, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb5.i));
    v = __lsx_vfmadd_s(t, v, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb4.i));
    v = __lsx_vfmadd_s(t, v, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb3.i));
    v = __lsx_vfmadd_s(t, v, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb2.i));
    v = __lsx_vfmadd_s(t, v, (__m128)__lsx_vreplgr2vr_w(c_erfc_sb1.i));

    __m128 tu = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra7.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_ra6.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra5.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra4.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra3.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra2.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra1.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_ra0.i));

    __m128 tv = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa8.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_sa7.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa6.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa5.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa4.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa3.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa2.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_sa1.i));

    __m128i mask_ge_2_857143 = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(0x4036db6d), absx);
    u = (__m128)__lsx_vbitsel_v((__m128i)tu, (__m128i)u, mask_ge_2_857143);
    v = (__m128)__lsx_vbitsel_v((__m128i)tv, (__m128i)v, mask_ge_2_857143);

    tu = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa6.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_pa5.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa4.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa3.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa2.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa1.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pa0.i));

    tv = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_qa6.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_qa5.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qa4.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qa3.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qa2.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qa1.i));

    u = (__m128)__lsx_vbitsel_v((__m128i)tu, (__m128i)u, mask_ge_1_25);
    v = (__m128)__lsx_vbitsel_v((__m128i)tv, (__m128i)v, mask_ge_1_25);

    tu = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_pp4.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_pp3.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pp2.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pp1.i));
    tu = __lsx_vfmadd_s(t, tu, (__m128)__lsx_vreplgr2vr_w(c_erfc_pp0.i));

    tv = __lsx_vfmadd_s(t, (__m128)__lsx_vreplgr2vr_w(c_erfc_qq5.i), (__m128)__lsx_vreplgr2vr_w(c_erfc_qq4.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qq3.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qq2.i));
    tv = __lsx_vfmadd_s(t, tv, (__m128)__lsx_vreplgr2vr_w(c_erfc_qq1.i));

    u = (__m128)__lsx_vbitsel_v((__m128i)tu, (__m128i)u, mask_ge_0_84375);
    v = (__m128)__lsx_vbitsel_v((__m128i)tv, (__m128i)v, mask_ge_0_84375);

    v = __lsx_vfmadd_s(t, v, one);

    __m128 q = __lsx_vfdiv_s(u, v);
    __m128 ret = zero;

    __m128i z_i = __lsx_vand_v((__m128i)absx, __lsx_vreplgr2vr_w(0xfffff000));
    __m128 z = (__m128)z_i;

    __m128 neg_z = (__m128)__lsx_vxor_v((__m128i)z, __lsx_vreplgr2vr_w(0x80000000));
    __m128 r = exp_ps(__lsx_vfmadd_s(neg_z, z, (__m128)__lsx_vreplgr2vr_w(0xbf100000)));
    __m128 tmp = __lsx_vfmadd_s(__lsx_vfsub_s(z, absx), __lsx_vfadd_s(z, absx), q);
    r = __lsx_vfmul_s(r, exp_ps(tmp));
    r = __lsx_vfdiv_s(r, absx);
    t = __lsx_vfsub_s(two, r);
    __m128i mask_x_ge_0 = __lsx_vfcmp_cle_s(zero, x);
    r = (__m128)__lsx_vbitsel_v((__m128i)t, (__m128i)r, mask_x_ge_0);
    __m128i mask_absx_ge_28 = __lsx_vfcmp_clt_s((__m128)__lsx_vreplgr2vr_w(0x41e00000), absx);
    ret = (__m128)__lsx_vbitsel_v((__m128i)r, (__m128i)ret, mask_absx_ge_28);

    r = __lsx_vfsub_s(__lsx_vfsub_s(one, (__m128)__lsx_vreplgr2vr_w(c_erfc_erx.i)), q);
    t = __lsx_vfadd_s(q, __lsx_vfadd_s(one, (__m128)__lsx_vreplgr2vr_w(c_erfc_erx.i)));
    r = (__m128)__lsx_vbitsel_v((__m128i)t, (__m128i)r, mask_x_ge_0);
    ret = (__m128)__lsx_vbitsel_v((__m128i)r, (__m128i)ret, mask_ge_1_25);

    r = __lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(c_0p5.i), __lsx_vfmadd_s(x, q, __lsx_vfsub_s(x, (__m128)__lsx_vreplgr2vr_w(c_0p5.i))));
    ret = (__m128)__lsx_vbitsel_v((__m128i)r, (__m128i)ret, mask_ge_0_84375);

    __m128i mask_x_lt_m6 = __lsx_vfcmp_clt_s(x, (__m128)__lsx_vreplgr2vr_w(0xc0c00000));
    ret = (__m128)__lsx_vbitsel_v((__m128i)ret, (__m128i)two, mask_x_lt_m6);

    __m128i mask_nan = __lsx_vfcmp_cne_s(x, x);
    ret = (__m128)__lsx_vbitsel_v((__m128i)ret, (__m128i)x, mask_nan);

    return ret;
}

#endif // LSX_MATHFUN_H
