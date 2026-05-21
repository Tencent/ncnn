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

_MIPS_FLOAT_CONST(c_0, 0.0f);
_MIPS_FLOAT_CONST(c_1, 1.0f);
_MIPS_FLOAT_CONST(c_2, 2.0f);
_MIPS_FLOAT_CONST(c_n1, -1.0f);
_MIPS_FLOAT_CONST(c_n3, -3.0f);
_MIPS_FLOAT_CONST(c_0p5, 0.5f);
_MIPS_FLOAT_CONST(c_eps, 1E-8f);

#define c_inv_mant_mask_msa ~0x7f800000u
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
static NCNN_FORCEINLINE v4f32 log_ps(v4f32 x)
{
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);

    x = __msa_fmax_w(x, (v4f32)__msa_fill_w(0)); /* force flush to zero on denormal values */
    v4i32 invalid_mask = __msa_fcle_w(x, (v4f32)__msa_fill_w(0));

    v4i32 ux = (v4i32)(x);

    v4i32 emm0 = __msa_srl_w(ux, (v4i32)__msa_fill_w(23));

    /* keep only the fractional part */
    ux = (v4i32)__msa_and_v((v16u8)ux, (v16u8)__msa_fill_w(c_inv_mant_mask_msa));
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

    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p1.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p2.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p3.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p4.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p5.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p6.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p7.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_log_p8.i), y, x);
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
static NCNN_FORCEINLINE v4f32 exp_ps(v4f32 x)
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

    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p1.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p2.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p3.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p4.i), y, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_exp_p5.i), y, x);

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
static NCNN_FORCEINLINE v4f32 tanh_ps(v4f32 x)
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
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_11.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_9.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_7.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_5.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_3.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_alpha_1.i), y, z);
    y = __msa_fmul_w(y, x2);

    // evaluate the denominator polynomial w.
    v4f32 w = (v4f32)__msa_fill_w(c_tanh_beta_6.i);
    w = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_4.i), w, z);
    w = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_2.i), w, z);
    w = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_tanh_beta_0.i), w, z);

    // divide the numerator by the denominator.
    y = __msa_fdiv_w(y, w);

    // reinstate the sign.
    y = (v4f32)__msa_binsli_w((v4u32)y, (v4u32)x, 0);

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = (v4f32)__msa_bsel_v((v16u8)tiny_mask, (v16u8)y, (v16u8)x);

    return y;
}

_MIPS_FLOAT_CONST(c_minus_cephes_DP1, -0.78515625f);
_MIPS_FLOAT_CONST(c_minus_cephes_DP2, -2.4187564849853515625e-4f);
_MIPS_FLOAT_CONST(c_minus_cephes_DP3, -3.77489497744594108e-8f);
_MIPS_FLOAT_CONST(c_cephes_sin_p0, -1.9515295891E-4f);
_MIPS_FLOAT_CONST(c_cephes_sin_p1, 8.3321608736E-3f);
_MIPS_FLOAT_CONST(c_cephes_sin_p2, -1.6666654611E-1f);
_MIPS_FLOAT_CONST(c_cephes_cos_p0, 2.443315711809948E-005f);
_MIPS_FLOAT_CONST(c_cephes_cos_p1, -1.388731625493765E-003f);
_MIPS_FLOAT_CONST(c_cephes_cos_p2, 4.166664568298827E-002f);
_MIPS_FLOAT_CONST(c_cephes_FOPI, 1.27323954473516f); // 4/PI

static NCNN_FORCEINLINE v4f32 sin_ps(v4f32 x)
{
    v4f32 y;
    v4i32 swap_sign_bit, poly_mask, sign_bit;
    v4f32 n0p5 = (v4f32)__msa_fill_w_f32(-0.5f);
    v4i32 all_ones = __msa_fill_w(-1);

    sign_bit = (v4i32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x80000000));
    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    y = __msa_fmul_w(x, (v4f32)__msa_fill_w(c_cephes_FOPI.i));

    poly_mask = __msa_ftrunc_s_w(y);
    poly_mask = __msa_addv_w(poly_mask, __msa_fill_w(1));
    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(~1));
    y = __msa_ffint_s_w(poly_mask);

    swap_sign_bit = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(4));
    swap_sign_bit = (v4i32)__msa_slli_w(swap_sign_bit, 29);

    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(2));
    poly_mask = __msa_ceqi_w(poly_mask, 0);

    sign_bit = (v4i32)__msa_xor_v((v16u8)sign_bit, (v16u8)swap_sign_bit);

    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP1.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP2.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP3.i));

    y = (v4f32)__msa_fill_w(c_cephes_cos_p0.i);
    v4f32 z = __msa_fmul_w(x, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p1.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p2.i), y, z);
    y = __msa_fmul_w(y, z);
    y = __msa_fmul_w(y, z);
    y = __ncnn_msa_fmadd_w(y, z, n0p5);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_1.i));

    v4f32 y2 = (v4f32)__msa_fill_w(c_cephes_sin_p0.i);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p2.i), y2, z);
    y2 = __msa_fmul_w(y2, z);
    y2 = __ncnn_msa_fmadd_w(x, y2, x);

    y2 = (v4f32)__msa_and_v((v16u8)y2, (v16u8)poly_mask);
    y = (v4f32)__msa_and_v((v16u8)__msa_xor_v((v16u8)poly_mask, (v16u8)all_ones), (v16u8)y);
    y = __msa_fadd_w(y, y2);
    y = (v4f32)__msa_xor_v((v16u8)y, (v16u8)sign_bit);

    return y;
}

static NCNN_FORCEINLINE v4f32 cos_ps(v4f32 x)
{
    v4f32 y;
    v4i32 swap_sign_bit, poly_mask, sign_bit;
    v4f32 n0p5 = (v4f32)__msa_fill_w_f32(-0.5f);
    v4i32 all_ones = __msa_fill_w(-1);

    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    y = __msa_fmul_w(x, (v4f32)__msa_fill_w(c_cephes_FOPI.i));

    poly_mask = __msa_ftrunc_s_w(y);
    poly_mask = __msa_addv_w(poly_mask, __msa_fill_w(1));
    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(~1));
    y = __msa_ffint_s_w(poly_mask);
    poly_mask = __msa_subv_w(poly_mask, __msa_fill_w(2));

    swap_sign_bit = (v4i32)__msa_and_v((v16u8)__msa_xor_v((v16u8)poly_mask, (v16u8)all_ones), (v16u8)__msa_fill_w(4));
    swap_sign_bit = (v4i32)__msa_slli_w(swap_sign_bit, 29);

    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(2));
    poly_mask = __msa_ceqi_w(poly_mask, 0);

    sign_bit = swap_sign_bit;

    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP1.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP2.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP3.i));

    y = (v4f32)__msa_fill_w(c_cephes_cos_p0.i);
    v4f32 z = __msa_fmul_w(x, x);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p1.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p2.i), y, z);
    y = __msa_fmul_w(y, z);
    y = __msa_fmul_w(y, z);
    y = __ncnn_msa_fmadd_w(y, z, n0p5);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_1.i));

    v4f32 y2 = (v4f32)__msa_fill_w(c_cephes_sin_p0.i);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p2.i), y2, z);
    y2 = __msa_fmul_w(y2, z);
    y2 = __ncnn_msa_fmadd_w(x, y2, x);

    y2 = (v4f32)__msa_and_v((v16u8)y2, (v16u8)poly_mask);
    y = (v4f32)__msa_and_v((v16u8)__msa_xor_v((v16u8)poly_mask, (v16u8)all_ones), (v16u8)y);
    y = __msa_fadd_w(y, y2);
    y = (v4f32)__msa_xor_v((v16u8)y, (v16u8)sign_bit);

    return y;
}

static NCNN_FORCEINLINE void sincos_ps(v4f32 x, v4f32& s, v4f32& c)
{
    v4f32 y;
    v4i32 swap_sign_bit_cos, swap_sign_bit_sin, poly_mask, sign_bit_sin, sign_bit_cos;
    v4f32 n0p5 = (v4f32)__msa_fill_w_f32(-0.5f);
    v4i32 all_ones = __msa_fill_w(-1);

    sign_bit_sin = (v4i32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x80000000));
    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    y = __msa_fmul_w(x, (v4f32)__msa_fill_w(c_cephes_FOPI.i));

    poly_mask = __msa_ftrunc_s_w(y);
    poly_mask = __msa_addv_w(poly_mask, __msa_fill_w(1));
    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(~1));
    y = __msa_ffint_s_w(poly_mask);

    swap_sign_bit_cos = __msa_subv_w(poly_mask, __msa_fill_w(2));
    swap_sign_bit_cos = (v4i32)__msa_and_v((v16u8)__msa_xor_v((v16u8)swap_sign_bit_cos, (v16u8)all_ones), (v16u8)__msa_fill_w(4));
    swap_sign_bit_cos = (v4i32)__msa_slli_w(swap_sign_bit_cos, 29);

    swap_sign_bit_sin = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(4));
    swap_sign_bit_sin = (v4i32)__msa_slli_w(swap_sign_bit_sin, 29);

    poly_mask = (v4i32)__msa_and_v((v16u8)poly_mask, (v16u8)__msa_fill_w(2));
    poly_mask = __msa_ceqi_w(poly_mask, 0);

    sign_bit_sin = (v4i32)__msa_xor_v((v16u8)sign_bit_sin, (v16u8)swap_sign_bit_sin);
    sign_bit_cos = swap_sign_bit_cos;

    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP1.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP2.i));
    x = __ncnn_msa_fmadd_w(x, y, (v4f32)__msa_fill_w(c_minus_cephes_DP3.i));

    v4f32 z = __msa_fmul_w(x, x);
    y = (v4f32)__msa_fill_w(c_cephes_cos_p0.i);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p1.i), y, z);
    y = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_cos_p2.i), y, z);
    y = __msa_fmul_w(y, z);
    y = __msa_fmul_w(y, z);
    y = __ncnn_msa_fmadd_w(y, z, n0p5);
    y = __msa_fadd_w(y, (v4f32)__msa_fill_w(c_1.i));

    v4f32 y2 = (v4f32)__msa_fill_w(c_cephes_sin_p0.i);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_sin_p2.i), y2, z);
    y2 = __msa_fmul_w(y2, z);
    y2 = __ncnn_msa_fmadd_w(x, y2, x);

    v4f32 ysin1 = (v4f32)__msa_and_v((v16u8)__msa_xor_v((v16u8)poly_mask, (v16u8)all_ones), (v16u8)y);
    v4f32 ysin2 = (v4f32)__msa_and_v((v16u8)poly_mask, (v16u8)y2);
    y2 = __msa_fsub_w(y2, ysin2);
    y = __msa_fsub_w(y, ysin1);

    ysin1 = __msa_fadd_w(ysin1, ysin2);
    y = __msa_fadd_w(y, y2);

    s = (v4f32)__msa_xor_v((v16u8)ysin1, (v16u8)sign_bit_sin);
    c = (v4f32)__msa_xor_v((v16u8)y, (v16u8)sign_bit_cos);
}

static NCNN_FORCEINLINE v4f32 tan_ps(v4f32 x)
{
    v4f32 ysin, ycos;
    v4f32 eps = (v4f32)__msa_fill_w(c_eps.i);
    sincos_ps(x, ysin, ycos);
    v4i32 mask = __msa_fceq_w(ycos, (v4f32)__msa_fill_w(c_0.i));
    mask = (v4i32)__msa_and_v((v16u8)mask, (v16u8)eps);
    ycos = __msa_fadd_w(ycos, (v4f32)mask);
    return __msa_fdiv_w(ysin, ycos);
}

_MIPS_FLOAT_CONST(c_erf_threshold, 0.927734375f);
_MIPS_FLOAT_CONST(c_erf_c0, -1.72853470e-5f);
_MIPS_FLOAT_CONST(c_erf_c1, 3.83197126e-4f);
_MIPS_FLOAT_CONST(c_erf_c2, -3.88396438e-3f);
_MIPS_FLOAT_CONST(c_erf_c3, 2.42546219e-2f);
_MIPS_FLOAT_CONST(c_erf_c4, -1.06777877e-1f);
_MIPS_FLOAT_CONST(c_erf_c5, -6.34846687e-1f);
_MIPS_FLOAT_CONST(c_erf_c6, -1.28717512e-1f);
_MIPS_FLOAT_CONST(c_erf_p0, -5.96761703e-4f);
_MIPS_FLOAT_CONST(c_erf_p1, 4.99119423e-3f);
_MIPS_FLOAT_CONST(c_erf_p2, -2.67681349e-2f);
_MIPS_FLOAT_CONST(c_erf_p3, 1.12819925e-1f);
_MIPS_FLOAT_CONST(c_erf_p4, -3.76125336e-1f);
_MIPS_FLOAT_CONST(c_erf_p5, 1.28379166e-1f);

static NCNN_FORCEINLINE v4f32 erf_ps(v4f32 a)
{
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);

    v4f32 t = (v4f32)__msa_bclri_w((v4u32)a, 31);
    v4f32 s = __msa_fmul_w(a, a);

    v4i32 mask = __msa_fclt_w((v4f32)__msa_fill_w(c_erf_threshold.i), t);

    v4f32 r1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_c1.i), (v4f32)__msa_fill_w(c_erf_c0.i), t);
    v4f32 u = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_c3.i), (v4f32)__msa_fill_w(c_erf_c2.i), t);
    r1 = __ncnn_msa_fmadd_w(u, r1, s);
    r1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_c4.i), r1, t);
    r1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_c5.i), r1, t);
    r1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_c6.i), r1, t);
    v4f32 neg_t = (v4f32)__msa_bnegi_w((v4u32)t, 31);
    r1 = __ncnn_msa_fmadd_w(neg_t, r1, t);
    r1 = __msa_fsub_w(one, exp_ps(r1));
    r1 = (v4f32)__msa_binsli_w((v4u32)r1, (v4u32)a, 0);

    v4f32 r2 = (v4f32)__msa_fill_w(c_erf_p0.i);
    r2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_p1.i), r2, s);
    r2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_p2.i), r2, s);
    r2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_p3.i), r2, s);
    r2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_p4.i), r2, s);
    r2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_erf_p5.i), r2, s);
    r2 = __ncnn_msa_fmadd_w(a, r2, a);

    v4f32 r = (v4f32)__msa_bsel_v((v16u8)mask, (v16u8)r2, (v16u8)r1);

    return r;
}

static NCNN_FORCEINLINE v4f32 pow_ps(v4f32 a, v4f32 b)
{
    // pow(x, m) = exp(m * log(x))
    return exp_ps(__msa_fmul_w(b, log_ps(a)));
}

static NCNN_FORCEINLINE v4f32 sigmoid_ps(v4f32 _v)
{
    v4f32 _one = __msa_fill_w_f32(1.f);
    _v = (v4f32)__msa_bnegi_w((v4u32)_v, 31);
    _v = exp_ps(_v);
    _v = __msa_fadd_w(_v, _one);
    return __msa_fdiv_w(_one, _v);
}

_MIPS_FLOAT_CONST(c_cephes_asin_a4, 0.023994016f);
_MIPS_FLOAT_CONST(c_cephes_asin_a5, 0.042417344f);
_MIPS_FLOAT_CONST(c_cephes_asin_a2, 0.07494697f);
_MIPS_FLOAT_CONST(c_cephes_asin_a3, 0.045520633f);
_MIPS_FLOAT_CONST(c_cephes_asin_a0, 1.0f);
_MIPS_FLOAT_CONST(c_cephes_asin_a1, 0.166667819f);
_MIPS_FLOAT_CONST(c_cephes_asin_half_pi, 1.5707964f);
_MIPS_FLOAT_CONST(c_cephes_asin_pi, 3.1415927f);

static NCNN_FORCEINLINE v4f32 asin_ps(v4f32 x)
{
    v4f32 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    v4f32 is_big_input_one, output_approx, final_approx;
    v4f32 tmp1, tmp2, tmp3, tmp4;
    v4i32 mask, is_small_input, is_big_input;
    v4i32 all_ones = __msa_fill_w(-1);

    mask = (v4i32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x80000000));
    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    is_small_input = __msa_fcle_w(x, (v4f32)__msa_fill_w(c_0p5.i));
    is_big_input = (v4i32)__msa_xor_v((v16u8)is_small_input, (v16u8)all_ones);
    is_big_input_one = (v4f32)__msa_and_v((v16u8)__msa_fill_w(c_1.i), (v16u8)is_big_input);

    big_input_approx = __msa_fsub_w((v4f32)__msa_fill_w(c_1.i), x);
    big_input_approx = __msa_fmul_w((v4f32)__msa_fill_w(c_0p5.i), big_input_approx);
    big_input_approx = __msa_fsqrt_w(big_input_approx);

    input_approx = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)x);
    input_approx = (v4f32)__msa_or_v((v16u8)input_approx, __msa_and_v((v16u8)is_big_input, (v16u8)big_input_approx));

    square_of_input_approx = __msa_fmul_w(input_approx, input_approx);
    fourth_power_of_input_approx = __msa_fmul_w(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a2.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_asin_a4.i));
    tmp2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a3.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_asin_a5.i));
    tmp3 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a0.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a1.i), fourth_power_of_input_approx, tmp2);
    output_approx = __ncnn_msa_fmadd_w(tmp3, square_of_input_approx, tmp4);

    tmp1 = __msa_fmul_w((v4f32)__msa_fill_w(c_cephes_asin_half_pi.i), is_big_input_one);
    tmp2 = __msa_fmul_w(output_approx, input_approx);
    tmp3 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_1.i), (v4f32)__msa_fill_w(c_n3.i), is_big_input_one);

    final_approx = __ncnn_msa_fmadd_w(tmp1, tmp2, tmp3);
    final_approx = (v4f32)__msa_or_v((v16u8)final_approx, (v16u8)mask);

    return final_approx;
}

static NCNN_FORCEINLINE v4f32 acos_ps(v4f32 x)
{
    v4f32 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    v4f32 output_approx, final_approx, small_final_approx, big_final_approx;
    v4f32 tmp1, tmp2, tmp3, tmp4;
    v4i32 mask, is_small_input, is_big_input, lt_zero;
    v4i32 all_ones = __msa_fill_w(-1);

    lt_zero = __msa_fclt_w(x, (v4f32)__msa_fill_w(c_0.i));
    mask = (v4i32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x80000000));
    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    is_small_input = __msa_fcle_w(x, (v4f32)__msa_fill_w(c_0p5.i));
    is_big_input = (v4i32)__msa_xor_v((v16u8)is_small_input, (v16u8)all_ones);

    big_input_approx = __msa_fsub_w((v4f32)__msa_fill_w(c_1.i), x);
    big_input_approx = __msa_fmul_w((v4f32)__msa_fill_w(c_0p5.i), big_input_approx);
    big_input_approx = __msa_fsqrt_w(big_input_approx);

    input_approx = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)x);
    input_approx = (v4f32)__msa_or_v((v16u8)input_approx, __msa_and_v((v16u8)is_big_input, (v16u8)big_input_approx));

    square_of_input_approx = __msa_fmul_w(input_approx, input_approx);
    fourth_power_of_input_approx = __msa_fmul_w(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a2.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_asin_a4.i));
    tmp2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a3.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_asin_a5.i));
    tmp3 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a0.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_asin_a1.i), fourth_power_of_input_approx, tmp2);
    output_approx = __ncnn_msa_fmadd_w(tmp3, square_of_input_approx, tmp4);

    tmp1 = __msa_fmul_w(input_approx, output_approx);

    small_final_approx = (v4f32)__msa_or_v((v16u8)tmp1, (v16u8)mask);
    small_final_approx = __msa_fsub_w((v4f32)__msa_fill_w(c_cephes_asin_half_pi.i), small_final_approx);

    big_final_approx = (v4f32)__msa_and_v((v16u8)lt_zero, (v16u8)__msa_fill_w(c_cephes_asin_pi.i));
    tmp1 = __msa_fadd_w(tmp1, tmp1);
    tmp1 = (v4f32)__msa_or_v((v16u8)tmp1, (v16u8)mask);
    big_final_approx = __msa_fadd_w(big_final_approx, tmp1);

    final_approx = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)small_final_approx);
    final_approx = (v4f32)__msa_or_v((v16u8)final_approx, __msa_and_v((v16u8)is_big_input, (v16u8)big_final_approx));

    return final_approx;
}

_MIPS_FLOAT_CONST(c_cephes_atan_x0, 1.0f);
_MIPS_FLOAT_CONST(c_cephes_atan_x1, -0.33333072f);
_MIPS_FLOAT_CONST(c_cephes_atan_x2, 0.1999262f);
_MIPS_FLOAT_CONST(c_cephes_atan_x3, -0.14203644f);
_MIPS_FLOAT_CONST(c_cephes_atan_x4, 0.10640934f);
_MIPS_FLOAT_CONST(c_cephes_atan_x5, -0.07504295f);
_MIPS_FLOAT_CONST(c_cephes_atan_x6, 0.04269152f);
_MIPS_FLOAT_CONST(c_cephes_atan_x7, -0.01606863f);
_MIPS_FLOAT_CONST(c_cephes_atan_x8, 0.0028498897f);

static NCNN_FORCEINLINE v4f32 atan_ps(v4f32 x)
{
    v4i32 mask, is_small_input, is_big_input;
    v4f32 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, input_approx, output_approx;
    v4f32 square_of_input_approx, fourth_power_of_input_approx;
    v4i32 all_ones = __msa_fill_w(-1);

    mask = (v4i32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x80000000));
    x = (v4f32)__msa_and_v((v16u8)x, (v16u8)__msa_fill_w(0x7fffffff));

    is_small_input = __msa_fclt_w((v4f32)__msa_fill_w(c_1.i), x);
    is_big_input = (v4i32)__msa_xor_v((v16u8)is_small_input, (v16u8)all_ones);

    tmp1 = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)__msa_fill_w(c_n1.i));
    tmp1 = (v4f32)__msa_or_v(__msa_and_v((v16u8)is_big_input, (v16u8)x), (v16u8)tmp1);

    tmp2 = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)x);
    tmp2 = (v4f32)__msa_or_v(__msa_and_v((v16u8)is_big_input, (v16u8)__msa_fill_w(c_1.i)), (v16u8)tmp2);

    input_approx = __msa_fdiv_w(tmp1, tmp2);
    square_of_input_approx = __msa_fmul_w(input_approx, input_approx);
    fourth_power_of_input_approx = __msa_fmul_w(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x5.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_atan_x7.i));
    tmp2 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x6.i), fourth_power_of_input_approx, (v4f32)__msa_fill_w(c_cephes_atan_x8.i));
    tmp3 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x3.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x4.i), fourth_power_of_input_approx, tmp2);
    tmp5 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x1.i), fourth_power_of_input_approx, tmp3);
    tmp6 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x2.i), fourth_power_of_input_approx, tmp4);
    tmp7 = __ncnn_msa_fmadd_w((v4f32)__msa_fill_w(c_cephes_atan_x0.i), fourth_power_of_input_approx, tmp6);
    output_approx = __ncnn_msa_fmadd_w(tmp7, square_of_input_approx, tmp5);

    tmp1 = __msa_fmul_w(input_approx, output_approx);
    tmp2 = (v4f32)__msa_and_v((v16u8)is_small_input, (v16u8)__msa_fill_w(c_cephes_asin_half_pi.i));
    tmp1 = __msa_fadd_w(tmp1, tmp2);
    tmp1 = (v4f32)__msa_xor_v((v16u8)mask, (v16u8)tmp1);
    return tmp1;
}

static NCNN_FORCEINLINE v4f32 atan2_ps(v4f32 a, v4f32 b)
{
    //TODO msa optimize
    float tmpx[4];
    float tmpy[4];
    __msa_st_w((v4i32)a, tmpx, 0);
    __msa_st_w((v4i32)b, tmpy, 0);
    tmpx[0] = atan2f(tmpx[0], tmpy[0]);
    tmpx[1] = atan2f(tmpx[1], tmpy[1]);
    tmpx[2] = atan2f(tmpx[2], tmpy[2]);
    tmpx[3] = atan2f(tmpx[3], tmpy[3]);
    return (v4f32)__msa_ld_w(tmpx, 0);
}

static NCNN_FORCEINLINE v4f32 fmod_ps(v4f32 a, v4f32 b)
{
    // fmod(a,b) = a - trunc(a/b)*b   (trunc toward 0)
    v4f32 q = __msa_fdiv_w(a, b);
    v4i32 qi = __msa_ftrunc_s_w(q); // trunc toward zero (independent of RM)
    v4f32 qf = __msa_ffint_s_w(qi);
    return __msa_fsub_w(a, __msa_fmul_w(qf, b));
}

static NCNN_FORCEINLINE v4f32 round_ps(v4f32 x)
{
    v4f32 half = (v4f32)__msa_fill_w(c_0p5.i);
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);
    v4i32 sign_mask = __msa_fclt_w(x, (v4f32)__msa_fill_w(0));
    v4f32 abs_x = (v4f32)__msa_bclri_w((v4u32)x, 31);
    v4i32 xi = __msa_ftrunc_s_w(abs_x);
    v4f32 xf = __msa_ffint_s_w(xi);
    v4f32 diff = __msa_fsub_w(abs_x, xf);
    v4i32 diff_gt_half = __msa_fclt_w(half, diff);
    v4i32 diff_eq_half = __msa_fceq_w(diff, half);
    v4i32 xi_and_1 = (v4i32)__msa_and_v((v16u8)xi, (v16u8)__msa_fill_w(1));
    v4i32 is_odd = __msa_ceqi_w(xi_and_1, 1);
    v4i32 round_up = (v4i32)__msa_or_v((v16u8)diff_gt_half, __msa_and_v((v16u8)diff_eq_half, (v16u8)is_odd));
    v4f32 rounded = __msa_fadd_w(xf, (v4f32)__msa_and_v((v16u8)one, (v16u8)round_up));
    return (v4f32)__msa_bsel_v((v16u8)sign_mask, (v16u8)rounded, (v16u8)__msa_bnegi_w((v4u32)rounded, 31));
}

static NCNN_FORCEINLINE v4f32 logaddexp_ps(v4f32 a, v4f32 b)
{
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);
    v4f32 max_xy = __msa_fmax_w(a, b);
    v4f32 min_xy = __msa_fmin_w(a, b);
    v4f32 diff = __msa_fsub_w(min_xy, max_xy);
    v4f32 exp_diff = exp_ps(diff);
    v4f32 one_plus_exp = __msa_fadd_w(one, exp_diff);
    v4f32 log_result = log_ps(one_plus_exp);
    return __msa_fadd_w(max_xy, log_result);
}

static NCNN_FORCEINLINE v4f32 floor_ps(v4f32 x)
{
    v4i32 xi = __msa_ftrunc_s_w(x);
    v4f32 xf = __msa_ffint_s_w(xi);
    v4i32 need_adjust = __msa_fclt_w(x, xf);
    v4f32 one = (v4f32)__msa_fill_w(c_1.i);
    return __msa_fsub_w(xf, (v4f32)__msa_and_v((v16u8)one, (v16u8)need_adjust));
}

static NCNN_FORCEINLINE v4f32 floor_divide_ps(v4f32 a, v4f32 b)
{
    v4f32 q = __msa_fdiv_w(a, b);
    return floor_ps(q);
}

static NCNN_FORCEINLINE v4f32 remainder_ps(v4f32 a, v4f32 b)
{
    v4f32 q = __msa_fdiv_w(a, b);
    v4f32 rq = round_ps(q);
    return __msa_fsub_w(a, __msa_fmul_w(rq, b));
}

#endif // MSA_MATHFUN_H
