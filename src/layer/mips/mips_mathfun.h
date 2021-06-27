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

#ifndef LAYER_MIPS_MATHFUN_H
#define LAYER_MIPS_MATHFUN_H

#include "mips_common.h"

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

    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p1.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p2.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p3.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p4.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p5.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p6.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p7.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_log_p8.i));
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
    tmp = __msa_ffint_s_w(__msa_ftrunc_s_w(fx));

    /* if greater, substract 1 */
    v4i32_w mask = __msa_fslt_w(fx, tmp);
    mask = (v4i32_w)__msa_and_v((v16u8)mask, (v16u8)one);

    fx = __msa_fsub_w(tmp, (v4f32)mask);

    tmp = __msa_fmul_w(fx, (v4f32)__msa_fill_w(c_cephes_exp_C1.i));
    v4f32 z = __msa_fmul_w(fx, (v4f32)__msa_fill_w(c_cephes_exp_C2.i));
    x = __msa_fsub_w(x, tmp);
    x = __msa_fsub_w(x, z);

    v4f32 y = (v4f32)__msa_fill_w(c_cephes_exp_p0.i);

    y = __msa_fmul_w(y, x);
    z = __msa_fmul_w(x, x);

    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_exp_p1.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_exp_p2.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_exp_p3.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_exp_p4.i));
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_exp_p5.i));

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

_MIPS_FLOAT_CONST(c_cephes_HALFMAXLOGF, 44.014845935754205f);
_MIPS_FLOAT_CONST(c_cephes_tanh_C1, 0.625f);

_MIPS_FLOAT_CONST(c_cephes_tanh_p0, -5.70498872745E-3);
_MIPS_FLOAT_CONST(c_cephes_tanh_p1, +2.06390887954E-2);
_MIPS_FLOAT_CONST(c_cephes_tanh_p2, -5.37397155531E-2);
_MIPS_FLOAT_CONST(c_cephes_tanh_p3, +1.33314422036E-1);
_MIPS_FLOAT_CONST(c_cephes_tanh_p4, -3.33332819422E-1);

/* tanh() computed for 4 float at once */
static inline v4f32 tanh_ps(v4f32 x)
{
    v4f32 x2 = (v4f32)__msa_bclri_w((v4u32)x, 31);

    v4i32_w mask_l = __msa_fclt_w((v4f32)__msa_fill_w(c_cephes_tanh_C1.i), x2);
    v4i32_w mask_l2 = __msa_fcle_w((v4f32)__msa_fill_w(c_cephes_HALFMAXLOGF.i), x2);

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    v4f32 _one = (v4f32)__msa_fill_w(c_1.i);
    v4f32 _two = (v4f32)__msa_fill_w(c_2.i);
    v4f32 exp_x_x = exp_ps(__msa_fadd_w(x, x));
    v4f32 y0 = __msa_fsub_w(_one, __msa_fdiv_w(_two, __msa_fadd_w(exp_x_x, _one)));

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
    v4f32 y = (v4f32)__msa_fill_w(c_cephes_tanh_p0.i);

    v4f32 z = __msa_fmul_w(x, x);

    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_tanh_p1.i));
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_tanh_p2.i));
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_tanh_p3.i));
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_cephes_tanh_p4.i));

    y = __msa_fmul_w(y, z);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    v4i32_w mask_pos = __msa_fcle_w((v4f32)__msa_fill_w(0), x);
    v4f32 y1 = (v4f32)__msa_bsel_v((v16u8)mask_pos, (v16u8)__msa_fill_w(c_n1.i), (v16u8)__msa_fill_w(c_1.i));

    y = (v4f32)__msa_bsel_v((v16u8)mask_l, (v16u8)y, (v16u8)y0);
    y = (v4f32)__msa_bsel_v((v16u8)mask_l2, (v16u8)y, (v16u8)y1);
    return y;
}

#endif // LAYER_MIPS_MATHFUN_H
