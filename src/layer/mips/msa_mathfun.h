/* MIPS implementation of exp
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 */

/* Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#ifndef MSA_MATHFUN_H
#define MSA_MATHFUN_H

#include "mips_usability.h"

#include <msa.h>

_MIPS_FLOAT_CONST(c_1, 1.0f);
_MIPS_FLOAT_CONST(c_2, 2.0f);
_MIPS_FLOAT_CONST(c_n1, -1.0f);
_MIPS_FLOAT_CONST(c_0p5, 0.5f);

#define c_inv_mant_mask ~0x7f800000u
_MIPS_FLOAT_CONST(c_cephes_SQRTHF, 0.707106781186547524);
_MIPS_FLOAT_CONST(c_cephes_log_p0, 7.0376836292E-2);
_MIPS_FLOAT_CONST(c_cephes_log_p1, -1.1514610310E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p2, 1.1676998740E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p3, -1.2420140846E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p4, +1.4249322787E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p5, -1.6668057665E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p6, +2.0000714765E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p7, -2.4999993993E-1);
_MIPS_FLOAT_CONST(c_cephes_log_p8, +3.3333331174E-1);
_MIPS_FLOAT_CONST(c_cephes_log_q1, -2.12194440e-4);
_MIPS_FLOAT_CONST(c_cephes_log_q2, 0.693359375);

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static inline v4f32 log_ps(v4f32 x)
{
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);

    x = __msa_fmax_w(x, (v4f32)__msa_fill_w(0)); /* force flush to zero on denormal values */
    v4i32 invalid_mask = __msa_fcle_w(x, (v4f32)__msa_fill_w(0));

    v4i32 ux = (v4i32)(x);

    v4i32 emm0 = __msa_srl_w(ux, (v4i32)__msa_fill_w(23));

    /* keep only the fractional part */
    ux = (v4i32)__msa_and_v((v16u8)ux, (v16u8)__msa_fill_w(c_inv_mant_mask));
    ux = (v4i32)__msa_or_v((v16u8)ux, (v16u8)__msa_fill_w(c_0p5.i));
    x = (v4f32)(ux);

    emm0 = __msa_subv_w(emm0, (v4i32)__msa_fill_w(0x7f));
    v4f32 e = __msa_ffint_s_w(emm0);

    e = __msa_fadd_w(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    v4i32 mask = __msa_fclt_w(x, (v4f32)__msa_fill_w(c_cephes_SQRTHF.i));
    v4f32 tmp = (v4f32)(__msa_and_v((v16u8)(x), (v16u8)mask));
    x = __msa_fsub_w(x, one);
    e = __msa_fsub_w(e, (v4f32)(__msa_and_v((v16u8)(one), (v16u8)mask)));
    x = __msa_fadd_w(x, tmp);

    v4f32 z = __msa_fmul_w(x, x);

    v4f32 y = (v4f32)__msa_fill_w(c_cephes_log_p0.i);

    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p1.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p2.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p3.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p4.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p5.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p6.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p7.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p8.i), y, x);
    y = __msa_fmul_w(y, x);

    y = __msa_fmul_w(y, z);

    tmp = __msa_fmul_w(e, (v4f32)__msa_fill_w(c_cephes_log_q1.i));
    y = __msa_fadd_w(y, tmp);

    tmp = __msa_fmul_w(z, (v4f32)__msa_fill_w(c_0p5.i));
    y = __msa_fsub_w(y, tmp);

    tmp = __msa_fmul_w(e, (v4f32)__msa_fill_w(c_cephes_log_q2.i));
    x = __msa_fadd_w(x, y);
    x = __msa_fadd_w(x, tmp);
    x = (v4f32)(__msa_or_v((v16u8)(x), (v16u8)invalid_mask)); // negative arg will be NAN
    return x;
}

_MIPS_FLOAT_CONST(c_exp_hi, 88.3762626647949f);
_MIPS_FLOAT_CONST(c_exp_lo, -88.3762626647949f);

_MIPS_FLOAT_CONST(c_cephes_LOG2EF, 1.44269504088896341);
_MIPS_FLOAT_CONST(c_cephes_exp_C1, 0.693359375);
_MIPS_FLOAT_CONST(c_cephes_exp_C2, -2.12194440e-4);

_MIPS_FLOAT_CONST(c_cephes_exp_p0, 1.9875691500E-4);
_MIPS_FLOAT_CONST(c_cephes_exp_p1, 1.3981999507E-3);
_MIPS_FLOAT_CONST(c_cephes_exp_p2, 8.3334519073E-3);
_MIPS_FLOAT_CONST(c_cephes_exp_p3, 4.1665795894E-2);
_MIPS_FLOAT_CONST(c_cephes_exp_p4, 1.6666665459E-1);
_MIPS_FLOAT_CONST(c_cephes_exp_p5, 5.0000001201E-1);

/* exp() computed for 4 float at once */
static inline v4f32 exp_ps(v4f32 x)
{
    v4f32 tmp, fx;

    v4f32 one = (v4f32)__msa_fill_w(c_1.i);
    x = __msa_fmin_w(x, (v4f32)__msa_fill_w(c_exp_hi.i));
    x = __msa_fmax_w(x, (v4f32)__msa_fill_w(c_exp_lo.i));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __msa_fmul_w(x, (v4f32)__msa_fill_w(c_cephes_LOG2EF.i));
    fx = __msa_fadd_w(fx, (v4f32)__msa_fill_w(c_0p5.i));

    /* perform a floorf */
    tmp = __msa_ffint_s_w(__msa_ftint_s_w(fx));

    /* if greater, substract 1 */
    v4i32_w mask = __msa_fclt_w(fx, tmp);
    mask = (v4i32_w)__msa_and_v((v16u8)mask, (v16u8)one);

    fx = __msa_fsub_w(tmp, (v4f32)mask);

    tmp = __msa_fmul_w(fx, (v4f32)__msa_fill_w(c_cephes_exp_C1.i));
    v4f32 z = __msa_fmul_w(fx, (v4f32)__msa_fill_w(c_cephes_exp_C2.i));
    x = __msa_fsub_w(x, tmp);
    x = __msa_fsub_w(x, z);

    v4f32 y = (v4f32)__msa_fill_w(c_cephes_exp_p0.i);

    z = __msa_fmul_w(x, x);

    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p1.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p2.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p3.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p4.i), y, x);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p5.i), y, x);

    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, x);
    y = __msa_fadd_w(y, one);

    /* build 2^n */
    v4i32 mm;
    mm = __msa_ftrunc_s_w(fx);
    mm = __msa_addv_w(mm, __msa_fill_w(0x7f));
    mm = __msa_sll_w(mm, __msa_fill_w(23));

    y = __msa_fmul_w(y, (v4f32)mm);
    return y;
}

_MIPS_FLOAT_CONST(c_tanh_tiny, 1e-4f);
_MIPS_FLOAT_CONST(c_tanh_hi, 9.0f);
// The monomial coefficients of the numerator polynomial (odd).
_MIPS_FLOAT_CONST(c_tanh_alpha_1, 4.89352455891786e-3f);
_MIPS_FLOAT_CONST(c_tanh_alpha_3, 6.37261928875436e-4f);
_MIPS_FLOAT_CONST(c_tanh_alpha_5, 1.48572235717979e-5f);
_MIPS_FLOAT_CONST(c_tanh_alpha_7, 5.12229709037114e-8f);
_MIPS_FLOAT_CONST(c_tanh_alpha_9, -8.60467152213735e-11f);
_MIPS_FLOAT_CONST(c_tanh_alpha_11, 2.00018790482477e-13f);
_MIPS_FLOAT_CONST(c_tanh_alpha_13, -2.76076847742355e-16f);
// The monomial coefficients of the denominator polynomial (even).
_MIPS_FLOAT_CONST(c_tanh_beta_0, 4.89352518554385e-3f);
_MIPS_FLOAT_CONST(c_tanh_beta_2, 2.26843463243900e-3f);
_MIPS_FLOAT_CONST(c_tanh_beta_4, 1.18534705686654e-4f);
_MIPS_FLOAT_CONST(c_tanh_beta_6, 1.19825839466702e-6f);

/* tanh() computed for 4 float at once */
static inline v4f32 tanh_ps(v4f32 x)
{
    v4f32 x2 = (v4f32)__msa_bclri_w((v4u32)x, 31);
    v4i32 tiny_mask = __msa_fclt_w(x2, (v4f32)__msa_fill_w(c_tanh_tiny.i));

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = (v4f32)__msa_bsel_v((v16u8)__msa_fclt_w((v4f32)__msa_fill_w(c_tanh_hi.i), x2), (v16u8)x2, (v16u8)__msa_fill_w(c_tanh_hi.i));

    // since the polynomials are odd/even, we need x**2.
    v4f32 z = __msa_fmul_w(x2, x2);

    // evaluate the numerator polynomial y.
    v4f32 y = (v4f32)__msa_fill_w(c_tanh_alpha_13.i);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_11.i), y, z);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_9.i), y, z);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_7.i), y, z);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_5.i), y, z);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_3.i), y, z);
    y = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_1.i), y, z);
    y = __msa_fmul_w(y, x2);

    // evaluate the denominator polynomial w.
    v4f32 w = (v4f32)__msa_fill_w(c_tanh_beta_6.i);
    w = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_4.i), w, z);
    w = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_2.i), w, z);
    w = __msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_0.i), w, z);

    // divide the numerator by the denominator.
    y = __msa_fdiv_w(y, w);

    // reinstate the sign.
    y = (v4f32)__msa_binsli_w((v4u32)y, (v4u32)x, 0);

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = (v4f32)__msa_bsel_v((v16u8)tiny_mask, (v16u8)y, (v16u8)x);

    return y;
}

static inline v4f32 pow_ps(v4f32 a, v4f32 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp_ps(__msa_fmul_w(b, log_ps(a)));
}

static inline v4f32 sigmoid_ps(v4f32 _v)
{
    v4f32 _one = __msa_fill_w_f32(1.f);
    _v = (v4f32)__msa_bnegi_w((v4u32)_v, 31);
    _v = exp_ps(_v);
    _v = __msa_fadd_w(_v, _one);
    return __msa_fdiv_w(_one, _v);
}

static inline v4f32 atan2_ps(v4f32 a, v4f32 b)
{
    //TODO msa optimize
    float tmpx[4];
    float tmpy[4];
    __msa_st_w((v4i32)a, tmpx, 0);
    __msa_st_w((v4i32)b, tmpy, 0);
    tmpx[0] = atan2(tmpx[0], tmpy[0]);
    tmpx[1] = atan2(tmpx[1], tmpy[1]);
    tmpx[2] = atan2(tmpx[2], tmpy[2]);
    tmpx[3] = atan2(tmpx[3], tmpy[3]);
    return (v4f32)__msa_ld_w(tmpx, 0);
}

#endif // MSA_MATHFUN_H
