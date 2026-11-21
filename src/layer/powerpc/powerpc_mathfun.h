/* PowerPC implementation of exp
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 *
 *   Ported from the MIPS MSA version (src/layer/mips/msa_mathfun.h)
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

#ifndef POWERPC_MATHFUN_H
#define POWERPC_MATHFUN_H

#include "powerpc_usability.h"

#include <math.h>

#if __ALTIVEC__

/* msa-style shuffle mask helper: bits [2i+1:2i] of the mask select element i */
#define _MSA_SHUFFLE(z, y, x, w) ((w) | ((x) << 2) | ((y) << 4) | ((z) << 6))

static NCNN_FORCEINLINE v16u8 __ncnn_altivec_shuffle_perm(int e0, int e1, int e2, int e3)
{
    const v16u8 _perm = {
        (unsigned char)(e0 * 4 + 0), (unsigned char)(e0 * 4 + 1), (unsigned char)(e0 * 4 + 2), (unsigned char)(e0 * 4 + 3),
        (unsigned char)(e1 * 4 + 0), (unsigned char)(e1 * 4 + 1), (unsigned char)(e1 * 4 + 2), (unsigned char)(e1 * 4 + 3),
        (unsigned char)(e2 * 4 + 0), (unsigned char)(e2 * 4 + 1), (unsigned char)(e2 * 4 + 2), (unsigned char)(e2 * 4 + 3),
        (unsigned char)(e3 * 4 + 0), (unsigned char)(e3 * 4 + 1), (unsigned char)(e3 * 4 + 2), (unsigned char)(e3 * 4 + 3)
    };
    return _perm;
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_shuffle(v4f32 _v, int _m)
{
    return vec_perm(_v, _v, __ncnn_altivec_shuffle_perm(_m & 3, (_m >> 2) & 3, (_m >> 4) & 3, (_m >> 6) & 3));
}

// copy the sign bit of _x into _y, per lane
// used for the sign reinstatement of tanh/erf (msa binsli_w(y, x, 31) intent)
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_reinst_sign(v4f32 _y, v4f32 _x)
{
    // copy the sign bit of _x onto the magnitude of _y
    // note vec_sel() with a vector bool mask selects whole elements, a bitwise blend is required here
    v4u32 _mag = vec_and((v4u32)_y, (v4u32)vec_splats(0x7fffffffu));
    v4u32 _sign = vec_and((v4u32)_x, (v4u32)vec_splats(0x80000000u));
    return (v4f32)vec_or(_mag, _sign);
}

_POWERPC_FLOAT_CONST(c_0, 0.0f);
_POWERPC_FLOAT_CONST(c_1, 1.0f);
_POWERPC_FLOAT_CONST(c_2, 2.0f);
_POWERPC_FLOAT_CONST(c_n1, -1.0f);
_POWERPC_FLOAT_CONST(c_n3, -3.0f);
_POWERPC_FLOAT_CONST(c_0p5, 0.5f);
_POWERPC_FLOAT_CONST(c_eps, 1E-8f);

#define c_inv_mant_mask_msa ~0x7f800000u
_POWERPC_FLOAT_CONST(c_cephes_SQRTHF, 0.707106781186547524);
_POWERPC_FLOAT_CONST(c_cephes_log_p0, 7.0376836292E-2);
_POWERPC_FLOAT_CONST(c_cephes_log_p1, -1.1514610310E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p2, 1.1676998740E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p3, -1.2420140846E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p4, +1.4249322787E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p5, -1.6668057665E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p6, +2.0000714765E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p7, -2.4999993993E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_p8, +3.3333331174E-1);
_POWERPC_FLOAT_CONST(c_cephes_log_q1, -2.12194440e-4);
_POWERPC_FLOAT_CONST(c_cephes_log_q2, 0.693359375);

/* natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
static NCNN_FORCEINLINE v4f32 log_ps(v4f32 x)
{
    v4f32 one = (v4f32)vec_splats(c_1.i);

    x = vec_max(x, (v4f32)vec_splats(0)); /* force flush to zero on denormal values */
    v4i32 invalid_mask = (v4i32)vec_cmpge((v4f32)vec_splats(0), x);
    // log(0) is -inf. the polynomial below turns a flushed zero into a large
    // finite value, which the scalar logf never does, so replace those lanes.
    x = __ncnn_altivec_sel(invalid_mask, vec_splats(1.f), x);

    v4i32 ux = (v4i32)(x);

    v4i32 emm0 = (v4i32)vec_sr((v4u32)ux, vec_splats((unsigned)23));

    /* keep only the fractional part */
    ux = __ncnn_altivec_and(ux, (v4i32)vec_splats((int)c_inv_mant_mask_msa));
    ux = __ncnn_altivec_or(ux, (v4i32)vec_splats(c_0p5.i));
    x = (v4f32)(ux);

    emm0 = vec_sub(emm0, vec_splats(0x7f));
    v4f32 e = __ncnn_altivec_ffint_s(emm0);

    e = vec_add(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    v4i32 mask = (v4i32)vec_cmpgt((v4f32)vec_splats(c_cephes_SQRTHF.i), x);
    v4f32 tmp = (v4f32)(__ncnn_altivec_and((v4i32)(x), mask));
    x = vec_sub(x, one);
    e = vec_sub(e, (v4f32)(__ncnn_altivec_and((v4i32)(one), mask)));
    x = vec_add(x, tmp);

    v4f32 z = __ncnn_altivec_mul(x, x);

    v4f32 y = (v4f32)vec_splats(c_cephes_log_p0.i);

    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p1.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p2.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p3.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p4.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p5.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p6.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p7.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_log_p8.i), y, x);
    y = __ncnn_altivec_mul(y, x);

    y = __ncnn_altivec_mul(y, z);

    tmp = __ncnn_altivec_mul(e, (v4f32)vec_splats(c_cephes_log_q1.i));
    y = vec_add(y, tmp);

    tmp = __ncnn_altivec_mul(z, (v4f32)vec_splats(c_0p5.i));
    y = vec_sub(y, tmp);

    tmp = __ncnn_altivec_mul(e, (v4f32)vec_splats(c_cephes_log_q2.i));
    x = vec_add(x, y);
    x = vec_add(x, tmp);
    // a non-positive argument is -inf, matching logf(0). or-ing the mask in
    // made it a NaN, which the bf16 round trip then stored as a NaN.
    x = __ncnn_altivec_sel(invalid_mask, vec_splats(-__builtin_inff()), x);
    return x;
}

_POWERPC_FLOAT_CONST(c_exp_hi, 88.3762626647949f);
_POWERPC_FLOAT_CONST(c_exp_lo, -88.3762626647949f);

_POWERPC_FLOAT_CONST(c_cephes_LOG2EF, 1.44269504088896341);
_POWERPC_FLOAT_CONST(c_cephes_exp_C1, 0.693359375);
_POWERPC_FLOAT_CONST(c_cephes_exp_C2, -2.12194440e-4);

_POWERPC_FLOAT_CONST(c_cephes_exp_p0, 1.9875691500E-4);
_POWERPC_FLOAT_CONST(c_cephes_exp_p1, 1.3981999507E-3);
_POWERPC_FLOAT_CONST(c_cephes_exp_p2, 8.3334519073E-3);
_POWERPC_FLOAT_CONST(c_cephes_exp_p3, 4.1665795894E-2);
_POWERPC_FLOAT_CONST(c_cephes_exp_p4, 1.6666665459E-1);
_POWERPC_FLOAT_CONST(c_cephes_exp_p5, 5.0000001201E-1);

/* exp() computed for 4 float at once */
static NCNN_FORCEINLINE v4f32 exp_ps(v4f32 x)
{
    v4f32 tmp, fx;

    v4f32 one = (v4f32)vec_splats(c_1.i);
    x = vec_min(x, (v4f32)vec_splats(c_exp_hi.i));
    x = vec_max(x, (v4f32)vec_splats(c_exp_lo.i));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __ncnn_altivec_mul(x, (v4f32)vec_splats(c_cephes_LOG2EF.i));
    fx = vec_add(fx, (v4f32)vec_splats(c_0p5.i));

    /* perform a floorf */
    tmp = __ncnn_altivec_ffint_s(__ncnn_altivec_ftint_s(fx));

    /* if greater, substract 1 */
    v4i32 mask = (v4i32)vec_cmpgt(tmp, fx);
    mask = __ncnn_altivec_and(mask, (v4i32)one);

    fx = vec_sub(tmp, (v4f32)mask);

    tmp = __ncnn_altivec_mul(fx, (v4f32)vec_splats(c_cephes_exp_C1.i));
    v4f32 z = __ncnn_altivec_mul(fx, (v4f32)vec_splats(c_cephes_exp_C2.i));
    x = vec_sub(x, tmp);
    x = vec_sub(x, z);

    v4f32 y = (v4f32)vec_splats(c_cephes_exp_p0.i);

    z = __ncnn_altivec_mul(x, x);

    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_exp_p1.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_exp_p2.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_exp_p3.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_exp_p4.i), y, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_exp_p5.i), y, x);

    y = __ncnn_altivec_mul(y, z);
    y = vec_add(y, x);
    y = vec_add(y, one);

    /* build 2^n */
    v4i32 mm;
    mm = __ncnn_altivec_ftrunc_s(fx);
    mm = vec_add(mm, vec_splats(0x7f));
    mm = (v4i32)vec_sl((v4u32)mm, vec_splats((unsigned)23));

    y = __ncnn_altivec_mul(y, (v4f32)mm);
    return y;
}

_POWERPC_FLOAT_CONST(c_tanh_tiny, 1e-4f);
_POWERPC_FLOAT_CONST(c_tanh_hi, 9.0f);
// The monomial coefficients of the numerator polynomial (odd).
_POWERPC_FLOAT_CONST(c_tanh_alpha_1, 4.89352455891786e-3f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_3, 6.37261928875436e-4f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_5, 1.48572235717979e-5f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_7, 5.12229709037114e-8f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_9, -8.60467152213735e-11f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_11, 2.00018790482477e-13f);
_POWERPC_FLOAT_CONST(c_tanh_alpha_13, -2.76076847742355e-16f);
// The monomial coefficients of the denominator polynomial (even).
_POWERPC_FLOAT_CONST(c_tanh_beta_0, 4.89352518554385e-3f);
_POWERPC_FLOAT_CONST(c_tanh_beta_2, 2.26843463243900e-3f);
_POWERPC_FLOAT_CONST(c_tanh_beta_4, 1.18534705686654e-4f);
_POWERPC_FLOAT_CONST(c_tanh_beta_6, 1.19825839466702e-6f);

/* tanh() computed for 4 float at once */
static NCNN_FORCEINLINE v4f32 tanh_ps(v4f32 x)
{
    v4f32 x2 = __ncnn_altivec_clear_sign(x);
    v4i32 tiny_mask = (v4i32)vec_cmpgt((v4f32)vec_splats(c_tanh_tiny.i), x2);

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = vec_min(x2, (v4f32)vec_splats(c_tanh_hi.i));

    // since the polynomials are odd/even, we need x**2.
    v4f32 z = __ncnn_altivec_mul(x2, x2);

    // evaluate the numerator polynomial y.
    v4f32 y = (v4f32)vec_splats(c_tanh_alpha_13.i);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_11.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_9.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_7.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_5.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_3.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_alpha_1.i), y, z);
    y = __ncnn_altivec_mul(y, x2);

    // evaluate the denominator polynomial w.
    v4f32 w = (v4f32)vec_splats(c_tanh_beta_6.i);
    w = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_beta_4.i), w, z);
    w = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_beta_2.i), w, z);
    w = __ncnn_altivec_fmadd((v4f32)vec_splats(c_tanh_beta_0.i), w, z);

    // divide the numerator by the denominator.
    y = __ncnn_altivec_div2(y, w);

    // reinstate the sign.
    y = __ncnn_altivec_reinst_sign(y, x);

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = __ncnn_altivec_sel(tiny_mask, x, y);

    return y;
}

_POWERPC_FLOAT_CONST(c_minus_cephes_DP1, -0.78515625f);
_POWERPC_FLOAT_CONST(c_minus_cephes_DP2, -2.4187564849853515625e-4f);
_POWERPC_FLOAT_CONST(c_minus_cephes_DP3, -3.77489497744594108e-8f);
_POWERPC_FLOAT_CONST(c_cephes_sin_p0, -1.9515295891E-4f);
_POWERPC_FLOAT_CONST(c_cephes_sin_p1, 8.3321608736E-3f);
_POWERPC_FLOAT_CONST(c_cephes_sin_p2, -1.6666654611E-1f);
_POWERPC_FLOAT_CONST(c_cephes_cos_p0, 2.443315711809948E-005f);
_POWERPC_FLOAT_CONST(c_cephes_cos_p1, -1.388731625493765E-003f);
_POWERPC_FLOAT_CONST(c_cephes_cos_p2, 4.166664568298827E-002f);
_POWERPC_FLOAT_CONST(c_cephes_FOPI, 1.27323954473516f); // 4/PI

static NCNN_FORCEINLINE v4f32 sin_ps(v4f32 x)
{
    v4f32 y;
    v4i32 swap_sign_bit, poly_mask, sign_bit;
    v4f32 n0p5 = __ncnn_altivec_fill_f32(-0.5f);
    v4i32 all_ones = vec_splats(-1);

    sign_bit = __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000));
    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    y = __ncnn_altivec_mul(x, (v4f32)vec_splats(c_cephes_FOPI.i));

    poly_mask = __ncnn_altivec_ftrunc_s(y);
    poly_mask = vec_add(poly_mask, vec_splats(1));
    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats((int)~1));
    y = __ncnn_altivec_ffint_s(poly_mask);

    swap_sign_bit = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats(4));
    swap_sign_bit = (v4i32)vec_sl((v4u32)swap_sign_bit, vec_splats((unsigned)29));

    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats(2));
    poly_mask = (v4i32)vec_cmpeq(poly_mask, vec_splats(0));

    sign_bit = __ncnn_altivec_xor(sign_bit, swap_sign_bit);

    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP1.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP2.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP3.i));

    y = (v4f32)vec_splats(c_cephes_cos_p0.i);
    v4f32 z = __ncnn_altivec_mul(x, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p1.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p2.i), y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_fmadd(y, z, n0p5);
    y = vec_add(y, (v4f32)vec_splats(c_1.i));

    v4f32 y2 = (v4f32)vec_splats(c_cephes_sin_p0.i);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p2.i), y2, z);
    y2 = __ncnn_altivec_mul(y2, z);
    y2 = __ncnn_altivec_fmadd(x, y2, x);

    y2 = (v4f32)__ncnn_altivec_and((v4i32)y2, poly_mask);
    y = (v4f32)__ncnn_altivec_and(__ncnn_altivec_xor(poly_mask, all_ones), (v4i32)y);
    y = vec_add(y, y2);
    y = (v4f32)__ncnn_altivec_xor((v4i32)y, sign_bit);

    return y;
}

static NCNN_FORCEINLINE v4f32 cos_ps(v4f32 x)
{
    v4f32 y;
    v4i32 swap_sign_bit, poly_mask, sign_bit;
    v4f32 n0p5 = __ncnn_altivec_fill_f32(-0.5f);
    v4i32 all_ones = vec_splats(-1);

    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    y = __ncnn_altivec_mul(x, (v4f32)vec_splats(c_cephes_FOPI.i));

    poly_mask = __ncnn_altivec_ftrunc_s(y);
    poly_mask = vec_add(poly_mask, vec_splats(1));
    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats((int)~1));
    y = __ncnn_altivec_ffint_s(poly_mask);
    poly_mask = vec_sub(poly_mask, vec_splats(2));

    swap_sign_bit = __ncnn_altivec_and(__ncnn_altivec_xor(poly_mask, all_ones), (v4i32)vec_splats(4));
    swap_sign_bit = (v4i32)vec_sl((v4u32)swap_sign_bit, vec_splats((unsigned)29));

    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats(2));
    poly_mask = (v4i32)vec_cmpeq(poly_mask, vec_splats(0));

    sign_bit = swap_sign_bit;

    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP1.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP2.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP3.i));

    y = (v4f32)vec_splats(c_cephes_cos_p0.i);
    v4f32 z = __ncnn_altivec_mul(x, x);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p1.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p2.i), y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_fmadd(y, z, n0p5);
    y = vec_add(y, (v4f32)vec_splats(c_1.i));

    v4f32 y2 = (v4f32)vec_splats(c_cephes_sin_p0.i);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p2.i), y2, z);
    y2 = __ncnn_altivec_mul(y2, z);
    y2 = __ncnn_altivec_fmadd(x, y2, x);

    y2 = (v4f32)__ncnn_altivec_and((v4i32)y2, poly_mask);
    y = (v4f32)__ncnn_altivec_and(__ncnn_altivec_xor(poly_mask, all_ones), (v4i32)y);
    y = vec_add(y, y2);
    y = (v4f32)__ncnn_altivec_xor((v4i32)y, sign_bit);

    return y;
}

static NCNN_FORCEINLINE void sincos_ps(v4f32 x, v4f32& s, v4f32& c)
{
    v4f32 y;
    v4i32 swap_sign_bit_cos, swap_sign_bit_sin, poly_mask, sign_bit_sin, sign_bit_cos;
    v4f32 n0p5 = __ncnn_altivec_fill_f32(-0.5f);
    v4i32 all_ones = vec_splats(-1);

    sign_bit_sin = __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000));
    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    y = __ncnn_altivec_mul(x, (v4f32)vec_splats(c_cephes_FOPI.i));

    poly_mask = __ncnn_altivec_ftrunc_s(y);
    poly_mask = vec_add(poly_mask, vec_splats(1));
    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats((int)~1));
    y = __ncnn_altivec_ffint_s(poly_mask);

    swap_sign_bit_cos = vec_sub(poly_mask, vec_splats(2));
    swap_sign_bit_cos = __ncnn_altivec_and(__ncnn_altivec_xor(swap_sign_bit_cos, all_ones), (v4i32)vec_splats(4));
    swap_sign_bit_cos = (v4i32)vec_sl((v4u32)swap_sign_bit_cos, vec_splats((unsigned)29));

    swap_sign_bit_sin = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats(4));
    swap_sign_bit_sin = (v4i32)vec_sl((v4u32)swap_sign_bit_sin, vec_splats((unsigned)29));

    poly_mask = __ncnn_altivec_and(poly_mask, (v4i32)vec_splats(2));
    poly_mask = (v4i32)vec_cmpeq(poly_mask, vec_splats(0));

    sign_bit_sin = __ncnn_altivec_xor(sign_bit_sin, swap_sign_bit_sin);
    sign_bit_cos = swap_sign_bit_cos;

    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP1.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP2.i));
    x = __ncnn_altivec_fmadd(x, y, (v4f32)vec_splats(c_minus_cephes_DP3.i));

    v4f32 z = __ncnn_altivec_mul(x, x);
    y = (v4f32)vec_splats(c_cephes_cos_p0.i);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p1.i), y, z);
    y = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_cos_p2.i), y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_mul(y, z);
    y = __ncnn_altivec_fmadd(y, z, n0p5);
    y = vec_add(y, (v4f32)vec_splats(c_1.i));

    v4f32 y2 = (v4f32)vec_splats(c_cephes_sin_p0.i);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p1.i), y2, z);
    y2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_sin_p2.i), y2, z);
    y2 = __ncnn_altivec_mul(y2, z);
    y2 = __ncnn_altivec_fmadd(x, y2, x);

    v4f32 ysin1 = (v4f32)__ncnn_altivec_and(__ncnn_altivec_xor(poly_mask, all_ones), (v4i32)y);
    v4f32 ysin2 = (v4f32)__ncnn_altivec_and(poly_mask, (v4i32)y2);
    y2 = vec_sub(y2, ysin2);
    y = vec_sub(y, ysin1);

    ysin1 = vec_add(ysin1, ysin2);
    y = vec_add(y, y2);

    s = (v4f32)__ncnn_altivec_xor((v4i32)ysin1, sign_bit_sin);
    c = (v4f32)__ncnn_altivec_xor((v4i32)y, sign_bit_cos);
}

static NCNN_FORCEINLINE v4f32 tan_ps(v4f32 x)
{
    v4f32 ysin, ycos;
    v4f32 eps = (v4f32)vec_splats(c_eps.i);
    sincos_ps(x, ysin, ycos);
    v4i32 mask = (v4i32)vec_cmpeq(ycos, (v4f32)vec_splats(c_0.i));
    mask = __ncnn_altivec_and(mask, (v4i32)eps);
    ycos = vec_add(ycos, (v4f32)mask);
    return __ncnn_altivec_div2(ysin, ycos);
}

_POWERPC_FLOAT_CONST(c_erf_threshold, 0.927734375f);
_POWERPC_FLOAT_CONST(c_erf_c0, -1.72853470e-5f);
_POWERPC_FLOAT_CONST(c_erf_c1, 3.83197126e-4f);
_POWERPC_FLOAT_CONST(c_erf_c2, -3.88396438e-3f);
_POWERPC_FLOAT_CONST(c_erf_c3, 2.42546219e-2f);
_POWERPC_FLOAT_CONST(c_erf_c4, -1.06777877e-1f);
_POWERPC_FLOAT_CONST(c_erf_c5, -6.34846687e-1f);
_POWERPC_FLOAT_CONST(c_erf_c6, -1.28717512e-1f);
_POWERPC_FLOAT_CONST(c_erf_p0, -5.96761703e-4f);
_POWERPC_FLOAT_CONST(c_erf_p1, 4.99119423e-3f);
_POWERPC_FLOAT_CONST(c_erf_p2, -2.67681349e-2f);
_POWERPC_FLOAT_CONST(c_erf_p3, 1.12819925e-1f);
_POWERPC_FLOAT_CONST(c_erf_p4, -3.76125336e-1f);
_POWERPC_FLOAT_CONST(c_erf_p5, 1.28379166e-1f);

static NCNN_FORCEINLINE v4f32 erf_ps(v4f32 a)
{
    v4f32 one = (v4f32)vec_splats(c_1.i);

    v4f32 t = __ncnn_altivec_clear_sign(a);
    v4f32 s = __ncnn_altivec_mul(a, a);

    v4i32 mask = (v4i32)vec_cmpgt(t, (v4f32)vec_splats(c_erf_threshold.i));

    v4f32 r1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_c1.i), (v4f32)vec_splats(c_erf_c0.i), t);
    v4f32 u = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_c3.i), (v4f32)vec_splats(c_erf_c2.i), t);
    r1 = __ncnn_altivec_fmadd(u, r1, s);
    r1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_c4.i), r1, t);
    r1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_c5.i), r1, t);
    r1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_c6.i), r1, t);
    v4f32 neg_t = __ncnn_altivec_flip_sign(t);
    r1 = __ncnn_altivec_fmadd(neg_t, r1, t);
    r1 = vec_sub(one, exp_ps(r1));
    r1 = __ncnn_altivec_reinst_sign(r1, a);

    v4f32 r2 = (v4f32)vec_splats(c_erf_p0.i);
    r2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_p1.i), r2, s);
    r2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_p2.i), r2, s);
    r2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_p3.i), r2, s);
    r2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_p4.i), r2, s);
    r2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_erf_p5.i), r2, s);
    r2 = __ncnn_altivec_fmadd(a, r2, a);

    v4f32 r = __ncnn_altivec_sel(mask, r1, r2);

    return r;
}

static NCNN_FORCEINLINE v4f32 pow_ps(v4f32 a, v4f32 b)
{
    // pow(x, m) = exp(m * log(x)) holds for a positive base only. zero and
    // negative bases, including an integral power of a negative number, go
    // through scalar powf for the lanes that need it.
    v4f32 r = exp_ps(__ncnn_altivec_mul(b, log_ps(a)));

    float ab[4];
    float bb[4];
    float rb[4];
    __ncnn_altivec_store(ab, a);
    __ncnn_altivec_store(bb, b);
    __ncnn_altivec_store(rb, r);
    for (int i = 0; i < 4; i++)
    {
        if (!(ab[i] > 0.f))
            rb[i] = powf(ab[i], bb[i]);
    }
    return __ncnn_altivec_load(rb);
}

static NCNN_FORCEINLINE v4f32 sigmoid_ps(v4f32 _v)
{
    v4f32 _one = __ncnn_altivec_fill_f32(1.f);
    _v = __ncnn_altivec_flip_sign(_v);
    _v = exp_ps(_v);
    _v = vec_add(_v, _one);
    return __ncnn_altivec_div2(_one, _v);
}

_POWERPC_FLOAT_CONST(c_cephes_asin_a4, 0.023994016f);
_POWERPC_FLOAT_CONST(c_cephes_asin_a5, 0.042417344f);
_POWERPC_FLOAT_CONST(c_cephes_asin_a2, 0.07494697f);
_POWERPC_FLOAT_CONST(c_cephes_asin_a3, 0.045520633f);
_POWERPC_FLOAT_CONST(c_cephes_asin_a0, 1.0f);
_POWERPC_FLOAT_CONST(c_cephes_asin_a1, 0.166667819f);
_POWERPC_FLOAT_CONST(c_cephes_asin_half_pi, 1.5707964f);
_POWERPC_FLOAT_CONST(c_cephes_asin_pi, 3.1415927f);

static NCNN_FORCEINLINE v4f32 asin_ps(v4f32 x)
{
    v4f32 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    v4f32 is_big_input_one, output_approx, final_approx;
    v4f32 tmp1, tmp2, tmp3, tmp4;
    v4i32 mask, is_small_input, is_big_input;
    v4i32 all_ones = vec_splats(-1);

    mask = __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000));
    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    is_small_input = (v4i32)vec_cmpge((v4f32)vec_splats(c_0p5.i), x);
    is_big_input = __ncnn_altivec_xor(is_small_input, all_ones);
    is_big_input_one = (v4f32)__ncnn_altivec_and((v4i32)vec_splats(c_1.i), is_big_input);

    big_input_approx = vec_sub((v4f32)vec_splats(c_1.i), x);
    big_input_approx = __ncnn_altivec_mul((v4f32)vec_splats(c_0p5.i), big_input_approx);
    big_input_approx = __ncnn_altivec_sqrt2(big_input_approx);

    input_approx = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)x);
    input_approx = (v4f32)__ncnn_altivec_or((v4i32)input_approx, __ncnn_altivec_and(is_big_input, (v4i32)big_input_approx));

    square_of_input_approx = __ncnn_altivec_mul(input_approx, input_approx);
    fourth_power_of_input_approx = __ncnn_altivec_mul(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a2.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_asin_a4.i));
    tmp2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a3.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_asin_a5.i));
    tmp3 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a0.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a1.i), fourth_power_of_input_approx, tmp2);
    output_approx = __ncnn_altivec_fmadd(tmp3, square_of_input_approx, tmp4);

    tmp1 = __ncnn_altivec_mul((v4f32)vec_splats(c_cephes_asin_half_pi.i), is_big_input_one);
    tmp2 = __ncnn_altivec_mul(output_approx, input_approx);
    tmp3 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_1.i), (v4f32)vec_splats(c_n3.i), is_big_input_one);

    final_approx = __ncnn_altivec_fmadd(tmp1, tmp2, tmp3);
    final_approx = (v4f32)__ncnn_altivec_or((v4i32)final_approx, mask);

    return final_approx;
}

static NCNN_FORCEINLINE v4f32 acos_ps(v4f32 x)
{
    v4f32 big_input_approx, input_approx, square_of_input_approx, fourth_power_of_input_approx;
    v4f32 output_approx, final_approx, small_final_approx, big_final_approx;
    v4f32 tmp1, tmp2, tmp3, tmp4;
    v4i32 mask, is_small_input, is_big_input, lt_zero;
    v4i32 all_ones = vec_splats(-1);

    lt_zero = (v4i32)vec_cmpgt((v4f32)vec_splats(c_0.i), x);
    mask = __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000));
    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    is_small_input = (v4i32)vec_cmpge((v4f32)vec_splats(c_0p5.i), x);
    is_big_input = __ncnn_altivec_xor(is_small_input, all_ones);

    big_input_approx = vec_sub((v4f32)vec_splats(c_1.i), x);
    big_input_approx = __ncnn_altivec_mul((v4f32)vec_splats(c_0p5.i), big_input_approx);
    big_input_approx = __ncnn_altivec_sqrt2(big_input_approx);

    input_approx = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)x);
    input_approx = (v4f32)__ncnn_altivec_or((v4i32)input_approx, __ncnn_altivec_and(is_big_input, (v4i32)big_input_approx));

    square_of_input_approx = __ncnn_altivec_mul(input_approx, input_approx);
    fourth_power_of_input_approx = __ncnn_altivec_mul(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a2.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_asin_a4.i));
    tmp2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a3.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_asin_a5.i));
    tmp3 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a0.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_asin_a1.i), fourth_power_of_input_approx, tmp2);
    output_approx = __ncnn_altivec_fmadd(tmp3, square_of_input_approx, tmp4);

    tmp1 = __ncnn_altivec_mul(input_approx, output_approx);

    small_final_approx = (v4f32)__ncnn_altivec_or((v4i32)tmp1, mask);
    small_final_approx = vec_sub((v4f32)vec_splats(c_cephes_asin_half_pi.i), small_final_approx);

    big_final_approx = (v4f32)__ncnn_altivec_and(lt_zero, (v4i32)vec_splats(c_cephes_asin_pi.i));
    tmp1 = vec_add(tmp1, tmp1);
    tmp1 = (v4f32)__ncnn_altivec_or((v4i32)tmp1, mask);
    big_final_approx = vec_add(big_final_approx, tmp1);

    final_approx = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)small_final_approx);
    final_approx = (v4f32)__ncnn_altivec_or((v4i32)final_approx, __ncnn_altivec_and(is_big_input, (v4i32)big_final_approx));

    return final_approx;
}

_POWERPC_FLOAT_CONST(c_cephes_atan_x0, 1.0f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x1, -0.33333072f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x2, 0.1999262f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x3, -0.14203644f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x4, 0.10640934f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x5, -0.07504295f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x6, 0.04269152f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x7, -0.01606863f);
_POWERPC_FLOAT_CONST(c_cephes_atan_x8, 0.0028498897f);

static NCNN_FORCEINLINE v4f32 atan_ps(v4f32 x)
{
    v4i32 mask, is_small_input, is_big_input;
    v4f32 tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, input_approx, output_approx;
    v4f32 square_of_input_approx, fourth_power_of_input_approx;
    v4i32 all_ones = vec_splats(-1);

    mask = __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000));
    x = (v4f32)__ncnn_altivec_and((v4i32)x, (v4i32)vec_splats(0x7fffffff));

    is_small_input = (v4i32)vec_cmpgt(x, (v4f32)vec_splats(c_1.i));
    is_big_input = __ncnn_altivec_xor(is_small_input, all_ones);

    tmp1 = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)vec_splats(c_n1.i));
    tmp1 = (v4f32)__ncnn_altivec_or(__ncnn_altivec_and(is_big_input, (v4i32)x), (v4i32)tmp1);

    tmp2 = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)x);
    tmp2 = (v4f32)__ncnn_altivec_or(__ncnn_altivec_and(is_big_input, (v4i32)vec_splats(c_1.i)), (v4i32)tmp2);

    input_approx = __ncnn_altivec_div2(tmp1, tmp2);
    square_of_input_approx = __ncnn_altivec_mul(input_approx, input_approx);
    fourth_power_of_input_approx = __ncnn_altivec_mul(square_of_input_approx, square_of_input_approx);

    tmp1 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x5.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_atan_x7.i));
    tmp2 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x6.i), fourth_power_of_input_approx, (v4f32)vec_splats(c_cephes_atan_x8.i));
    tmp3 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x3.i), fourth_power_of_input_approx, tmp1);
    tmp4 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x4.i), fourth_power_of_input_approx, tmp2);
    tmp5 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x1.i), fourth_power_of_input_approx, tmp3);
    tmp6 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x2.i), fourth_power_of_input_approx, tmp4);
    tmp7 = __ncnn_altivec_fmadd((v4f32)vec_splats(c_cephes_atan_x0.i), fourth_power_of_input_approx, tmp6);
    output_approx = __ncnn_altivec_fmadd(tmp7, square_of_input_approx, tmp5);

    tmp1 = __ncnn_altivec_mul(input_approx, output_approx);
    tmp2 = (v4f32)__ncnn_altivec_and(is_small_input, (v4i32)vec_splats(c_cephes_asin_half_pi.i));
    tmp1 = vec_add(tmp1, tmp2);
    tmp1 = (v4f32)__ncnn_altivec_xor(mask, (v4i32)tmp1);
    return tmp1;
}

static NCNN_FORCEINLINE v4f32 atan2_ps(v4f32 a, v4f32 b)
{
    //TODO altivec optimize
    float tmpx[4];
    float tmpy[4];
    __ncnn_altivec_store(tmpx, a);
    __ncnn_altivec_store(tmpy, b);
    tmpx[0] = atan2f(tmpx[0], tmpy[0]);
    tmpx[1] = atan2f(tmpx[1], tmpy[1]);
    tmpx[2] = atan2f(tmpx[2], tmpy[2]);
    tmpx[3] = atan2f(tmpx[3], tmpy[3]);
    return __ncnn_altivec_load(tmpx);
}

static NCNN_FORCEINLINE v4f32 fmod_ps(v4f32 a, v4f32 b)
{
    // fmod(a,b) = a - trunc(a/b)*b   (trunc toward 0)
    v4f32 q = __ncnn_altivec_div2(a, b);
    v4i32 qi = __ncnn_altivec_ftrunc_s(q); // trunc toward zero (independent of RM)
    v4f32 qf = __ncnn_altivec_ffint_s(qi);
    return vec_sub(a, __ncnn_altivec_mul(qf, b));
}

static NCNN_FORCEINLINE v4f32 round_ps(v4f32 x)
{
    v4f32 half = (v4f32)vec_splats(c_0p5.i);
    v4f32 one = (v4f32)vec_splats(c_1.i);
    v4i32 sign_mask = (v4i32)vec_cmpgt((v4f32)vec_splats(0), x);
    v4f32 abs_x = __ncnn_altivec_clear_sign(x);
    v4i32 xi = __ncnn_altivec_ftrunc_s(abs_x);
    v4f32 xf = __ncnn_altivec_ffint_s(xi);
    v4f32 diff = vec_sub(abs_x, xf);
    v4i32 diff_gt_half = (v4i32)vec_cmpgt(diff, half);
    v4i32 diff_eq_half = (v4i32)vec_cmpeq(diff, half);
    v4i32 xi_and_1 = __ncnn_altivec_and(xi, (v4i32)vec_splats(1));
    v4i32 is_odd = (v4i32)vec_cmpeq(xi_and_1, vec_splats(1));
    v4i32 round_up = __ncnn_altivec_or(diff_gt_half, __ncnn_altivec_and(diff_eq_half, is_odd));
    v4f32 rounded = vec_add(xf, (v4f32)__ncnn_altivec_and((v4i32)one, round_up));
    return __ncnn_altivec_sel(sign_mask, __ncnn_altivec_flip_sign(rounded), rounded);
}

static NCNN_FORCEINLINE v4f32 logaddexp_ps(v4f32 a, v4f32 b)
{
    v4f32 one = (v4f32)vec_splats(c_1.i);
    v4f32 max_xy = vec_max(a, b);
    v4f32 min_xy = vec_min(a, b);
    v4f32 diff = vec_sub(min_xy, max_xy);
    v4f32 exp_diff = exp_ps(diff);
    v4f32 one_plus_exp = vec_add(one, exp_diff);
    v4f32 log_result = log_ps(one_plus_exp);
    return vec_add(max_xy, log_result);
}

static NCNN_FORCEINLINE v4f32 abs_ps(v4f32 x)
{
    return __ncnn_altivec_clear_sign(x);
}

static NCNN_FORCEINLINE v4f32 expm1_ps(v4f32 x)
{
    v4f32 x2 = __ncnn_altivec_mul(x, x);
    v4f32 poly = vec_add(x, __ncnn_altivec_mul(x2, vec_add(__ncnn_altivec_fill_f32(0.5f), __ncnn_altivec_mul(x, __ncnn_altivec_fill_f32(1.0f / 6.0f)))));
    v4f32 y = vec_sub(exp_ps(x), __ncnn_altivec_fill_f32(1.f));
    v4i32 mask = (v4i32)vec_cmpgt(__ncnn_altivec_fill_f32(1e-4f), abs_ps(x));
    return __ncnn_altivec_sel(mask, poly, y);
}

static NCNN_FORCEINLINE v4f32 log1p_ps(v4f32 x)
{
    v4f32 x2 = __ncnn_altivec_mul(x, x);
    v4f32 poly = vec_add(x, __ncnn_altivec_mul(x2, vec_add(__ncnn_altivec_fill_f32(-0.5f), __ncnn_altivec_mul(x, __ncnn_altivec_fill_f32(1.0f / 3.0f)))));
    v4f32 y = log_ps(vec_add(__ncnn_altivec_fill_f32(1.f), x));
    v4i32 mask = (v4i32)vec_cmpgt(__ncnn_altivec_fill_f32(1e-4f), abs_ps(x));
    return __ncnn_altivec_sel(mask, poly, y);
}

static NCNN_FORCEINLINE v4f32 sinh_ps(v4f32 x)
{
    v4f32 expm1_x = expm1_ps(x);
    v4f32 expm1_neg_x = expm1_ps(vec_sub(__ncnn_altivec_fill_f32(0.f), x));
    return __ncnn_altivec_mul(vec_sub(expm1_x, expm1_neg_x), __ncnn_altivec_fill_f32(0.5f));
}

static NCNN_FORCEINLINE v4f32 asinh_ps(v4f32 x)
{
    v4f32 ax = abs_ps(x);
    v4f32 x2 = __ncnn_altivec_mul(ax, ax);
    v4f32 y = log_ps(vec_add(ax, __ncnn_altivec_sqrt2(vec_add(x2, __ncnn_altivec_fill_f32(1.f)))));
    v4f32 y_large = vec_add(log_ps(ax), __ncnn_altivec_fill_f32(0.6931471805599453f));
    v4i32 mask = (v4i32)vec_cmpgt(ax, __ncnn_altivec_fill_f32(1e19f));
    y = __ncnn_altivec_sel(mask, y_large, y);
    return (v4f32)__ncnn_altivec_or((v4i32)y, __ncnn_altivec_and((v4i32)x, (v4i32)vec_splats((int)0x80000000)));
}

static NCNN_FORCEINLINE v4f32 cosh_ps(v4f32 x)
{
    v4f32 exp_x = exp_ps(x);
    v4f32 exp_neg_x = exp_ps(vec_sub(__ncnn_altivec_fill_f32(0.f), x));
    return __ncnn_altivec_mul(vec_add(exp_x, exp_neg_x), __ncnn_altivec_fill_f32(0.5f));
}

static NCNN_FORCEINLINE v4f32 acosh_ps(v4f32 x)
{
    v4f32 one = __ncnn_altivec_fill_f32(1.f);
    v4f32 y = log_ps(vec_add(x, __ncnn_altivec_mul(__ncnn_altivec_sqrt2(vec_sub(x, one)), __ncnn_altivec_sqrt2(vec_add(x, one)))));
    v4f32 y_large = vec_add(log_ps(x), __ncnn_altivec_fill_f32(0.6931471805599453f));
    v4i32 mask = (v4i32)vec_cmpgt(x, __ncnn_altivec_fill_f32(1e19f));
    return __ncnn_altivec_sel(mask, y_large, y);
}

static NCNN_FORCEINLINE v4f32 atanh_ps(v4f32 x)
{
    v4f32 log_pos = log1p_ps(x);
    v4f32 log_neg = log1p_ps(vec_sub(__ncnn_altivec_fill_f32(0.f), x));
    return __ncnn_altivec_mul(vec_sub(log_pos, log_neg), __ncnn_altivec_fill_f32(0.5f));
}

static NCNN_FORCEINLINE v4f32 floor_ps(v4f32 x)
{
    v4i32 xi = __ncnn_altivec_ftrunc_s(x);
    v4f32 xf = __ncnn_altivec_ffint_s(xi);
    v4i32 need_adjust = (v4i32)vec_cmpgt(xf, x);
    v4f32 one = (v4f32)vec_splats(c_1.i);
    return vec_sub(xf, (v4f32)__ncnn_altivec_and((v4i32)one, need_adjust));
}

static NCNN_FORCEINLINE v4f32 floor_divide_ps(v4f32 a, v4f32 b)
{
    v4f32 q = __ncnn_altivec_div2(a, b);
    return floor_ps(q);
}

static NCNN_FORCEINLINE v4f32 remainder_ps(v4f32 a, v4f32 b)
{
    v4f32 q = __ncnn_altivec_div2(a, b);
    v4f32 rq = round_ps(q);
    return vec_sub(a, __ncnn_altivec_mul(rq, b));
}

#endif // __ALTIVEC__

#endif // POWERPC_MATHFUN_H
