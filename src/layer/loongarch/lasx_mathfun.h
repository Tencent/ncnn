// AtomAlpaca is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 AtomAlpaca <atal@anche.no>. All rights reserved.
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
//
//

#ifndef LASX_MATHFUN_H
#define LASX_MATHFUN_H

#include "loongarch_usability.h"

#include <lasxintrin.h>

_LOONGARCH_FLOAT_CONST_PS256(c_1, 1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_2, 2.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_n1, -1.0f);
_LOONGARCH_FLOAT_CONST_PS256(c_0p5, 0.5f);

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

_LOONGARCH_FLOAT_CONST_PS256(c_tanh_tiny, 1e-4f);
_LOONGARCH_FLOAT_CONST_PS256(c_tanh_hi, 9.0f);
// The monomial coefficients of the numerator polynomial (odd).
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

static inline __m256 atan2256_ps(__m256 a, __m256 b)
{
    //TODO lsx optimize
    float tmpx[4];
    float tmpy[4];
    __lasx_xvst(a, tmpx, 0);
    __lasx_xvst(b, tmpy, 0);
    tmpx[0] = atan2f(tmpx[0], tmpy[0]);
    tmpx[1] = atan2f(tmpx[1], tmpy[1]);
    tmpx[2] = atan2f(tmpx[2], tmpy[2]);
    tmpx[3] = atan2f(tmpx[3], tmpy[3]);
    return (__m256)__lasx_xvld(tmpx, 0);
}

#endif // LASX_MATHFUN_H
