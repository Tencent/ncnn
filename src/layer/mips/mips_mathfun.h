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
    v4f32 c1 = (v4f32)__msa_fill_w(c_cephes_exp_p1.i);
    v4f32 c2 = (v4f32)__msa_fill_w(c_cephes_exp_p2.i);
    v4f32 c3 = (v4f32)__msa_fill_w(c_cephes_exp_p3.i);
    v4f32 c4 = (v4f32)__msa_fill_w(c_cephes_exp_p4.i);
    v4f32 c5 = (v4f32)__msa_fill_w(c_cephes_exp_p5.i);

    y = __msa_fmul_w(y, x);
    z = __msa_fmul_w(x, x);

    y = __msa_fadd_w(y, c1);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, c2);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, c3);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, c4);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, c5);

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
    v4f32 c1 = (v4f32)__msa_fill_w(c_cephes_tanh_p1.i);
    v4f32 c2 = (v4f32)__msa_fill_w(c_cephes_tanh_p2.i);
    v4f32 c3 = (v4f32)__msa_fill_w(c_cephes_tanh_p3.i);
    v4f32 c4 = (v4f32)__msa_fill_w(c_cephes_tanh_p4.i);

    v4f32 z = __msa_fmul_w(x, x);

    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, c1);
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, c2);
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, c3);
    y = __msa_fmul_w(y, z);
    y = __msa_fadd_w(y, c4);

    y = __msa_fmul_w(y, z);
    y = __msa_fmul_w(y, x);
    y = __msa_fadd_w(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    v4i32_w mask_pos = __msa_fcle_w((v4f32)__msa_fill_w(0), x2);
    v4f32 y1 = (v4f32)__msa_bsel_v((v16u8)mask_pos, (v16u8)__msa_fill_w(c_1.i), (v16u8)__msa_fill_w(c_n1.i));

    y = (v4f32)__msa_bsel_v((v16u8)mask_l, (v16u8)y0, (v16u8)y);
    y = (v4f32)__msa_bsel_v((v16u8)mask_l2, (v16u8)y1, (v16u8)y);
    return y;
}

#endif // LAYER_MIPS_MATHFUN_H
