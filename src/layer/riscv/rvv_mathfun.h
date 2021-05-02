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

#ifndef RVV_MATHFUN_H
#define RVV_MATHFUN_H

#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

static inline vfloat32m8_t exp_ps(vfloat32m8_t x, word_type vl)
{
    vfloat32m8_t tmp, fx;

    x = vfmin_vf_f32m8(x, c_exp_hi, vl);
    x = vfmax_vf_f32m8(x, c_exp_lo, vl);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfmacc_vf_f32m8(vfmv_v_f_f32m8(0.5f, vl), c_cephes_LOG2EF, x, vl);

    /* perform a floorf */
    tmp = vfcvt_f_x_v_f32m8(vfcvt_x_f_v_i32m8(fx, vl), vl);

    /* if greater, substract 1 */
    vbool4_t mask = vmfgt_vv_f32m8_b4(tmp, fx, vl);
    fx = vfsub_vf_f32m8_m(mask, tmp, tmp, 1.f, vl);

    tmp = vfmul_vf_f32m8(fx, c_cephes_exp_C1, vl);
    vfloat32m8_t z = vfmul_vf_f32m8(fx, c_cephes_exp_C2, vl);
    x = vfsub_vv_f32m8(x, tmp, vl);
    x = vfsub_vv_f32m8(x, z, vl);

    vfloat32m8_t y = vfmul_vf_f32m8(x, c_cephes_exp_p0, vl);
    z = vfmul_vv_f32m8(x, x, vl);

    y = vfadd_vf_f32m8(y, c_cephes_exp_p1, vl);
    y = vfmul_vv_f32m8(y, x, vl);
    y = vfadd_vf_f32m8(y, c_cephes_exp_p2, vl);
    y = vfmul_vv_f32m8(y, x, vl);
    y = vfadd_vf_f32m8(y, c_cephes_exp_p3, vl);
    y = vfmul_vv_f32m8(y, x, vl);
    y = vfadd_vf_f32m8(y, c_cephes_exp_p4, vl);
    y = vfmul_vv_f32m8(y, x, vl);
    y = vfadd_vf_f32m8(y, c_cephes_exp_p5, vl);

    y = vfmul_vv_f32m8(y, z, vl);
    y = vfadd_vv_f32m8(y, x, vl);
    y = vfadd_vf_f32m8(y, 1.f, vl);

    /* build 2^n */
    vint32m8_t mm = vfcvt_x_f_v_i32m8(fx, vl);
    mm = vadd_vx_i32m8(mm, 0x7f, vl);
    mm = vsll_vx_i32m8(mm, 23, vl);
    vfloat32m8_t pow2n = vreinterpret_v_i32m8_f32m8(mm);

    y = vfmul_vv_f32m8(y, pow2n, vl);
    return y;
}

static inline vfloat32m8_t sigmoid_ps(vfloat32m8_t _v, word_type vl)
{
    _v = vfneg_v_f32m8(_v, vl);
    _v = exp_ps(_v, vl);
    _v = vfadd_vf_f32m8(_v, 1.f, vl);
    return vfrdiv_vf_f32m8(_v, 1.f, vl);
}

#endif // RVV_MATHFUN_H
