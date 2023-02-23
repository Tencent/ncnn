/* LOONGARCH implementation of exp
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

_LOONGARCH_FLOAT_CONST(c_1, 1.0f);
_LOONGARCH_FLOAT_CONST(c_2, 2.0f);
_LOONGARCH_FLOAT_CONST(c_n1, -1.0f);
_LOONGARCH_FLOAT_CONST(c_0p5, 0.5f);

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

static inline __m128 atan2_ps(__m128 a, __m128 b)
{
    //TODO lsx optimize
    float tmpx[4];
    float tmpy[4];
    __lsx_vst(a, tmpx, 0);
    __lsx_vst(b, tmpy, 0);
    tmpx[0] = atan2(tmpx[0], tmpy[0]);
    tmpx[1] = atan2(tmpx[1], tmpy[1]);
    tmpx[2] = atan2(tmpx[2], tmpy[2]);
    tmpx[3] = atan2(tmpx[3], tmpy[3]);
    return (__m128)__lsx_vld(tmpx, 0);
}

#endif // LSX_MATHFUN_H
