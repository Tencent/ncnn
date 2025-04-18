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

#ifndef RVV_MATHFUN_FP16S_H
#define RVV_MATHFUN_FP16S_H

#include <riscv_vector.h>

#define c_inv_mant_mask_f16 -31745 // ~0x7c00u
#define c_cephes_SQRTHF     0.707106781186547524
#define c_cephes_log_p0     7.0376836292E-2
#define c_cephes_log_p1     -1.1514610310E-1
#define c_cephes_log_p2     1.1676998740E-1
#define c_cephes_log_p3     -1.2420140846E-1
#define c_cephes_log_p4     +1.4249322787E-1
#define c_cephes_log_p5     -1.6668057665E-1
#define c_cephes_log_p6     +2.0000714765E-1
#define c_cephes_log_p7     -2.4999993993E-1
#define c_cephes_log_p8     +3.3333331174E-1
#define c_cephes_log_q1     -2.12194440e-4
#define c_cephes_log_q2     0.693359375

#define _RVV_FLOAT16_LOG_OP(LMUL, MLEN)                                                                                           \
    static inline vfloat16m##LMUL##_t log_ps(vfloat16m##LMUL##_t x, size_t vl)                                                    \
    {                                                                                                                             \
        x = __riscv_vfmax_vf_f16m##LMUL(x, (__fp16)0.f, vl); /* force flush to zero on denormal values */                         \
        vbool##MLEN##_t invalid_mask = __riscv_vmfle_vf_f16m##LMUL##_b##MLEN(x, (__fp16)0.f, vl);                                 \
                                                                                                                                  \
        vint16m##LMUL##_t ux = __riscv_vreinterpret_v_f16m##LMUL##_i16m##LMUL(x);                                                 \
                                                                                                                                  \
        vint16m##LMUL##_t emm0 = __riscv_vsra_vx_i16m##LMUL(ux, 10, vl);                                                          \
                                                                                                                                  \
        /* keep only the fractional part */                                                                                       \
        ux = __riscv_vand_vx_i16m##LMUL(ux, c_inv_mant_mask_f16, vl);                                                             \
        ux = __riscv_vor_vx_i16m##LMUL(ux, 14336 /* reinterpret_cast<short>((__fp16)0.5) */, vl);                                 \
        x = __riscv_vreinterpret_v_i16m##LMUL##_f16m##LMUL(ux);                                                                   \
                                                                                                                                  \
        emm0 = __riscv_vsub_vx_i16m##LMUL(emm0, 0xf, vl);                                                                         \
        vfloat16m##LMUL##_t e = __riscv_vfcvt_f_x_v_f16m##LMUL(emm0, vl);                                                         \
                                                                                                                                  \
        e = __riscv_vfadd_vf_f16m##LMUL(e, (__fp16)1.f, vl);                                                                      \
                                                                                                                                  \
        /* part2:                      */                                                                                         \
        /*     if( x < SQRTHF ) {      */                                                                                         \
        /*       e -= 1;               */                                                                                         \
        /*       x = x + x - 1.0;      */                                                                                         \
        /*     } else { x = x - 1.0; } */                                                                                         \
        vbool##MLEN##_t mask = __riscv_vmflt_vf_f16m##LMUL##_b##MLEN(x, (__fp16)c_cephes_SQRTHF, vl);                             \
        x = __riscv_vfadd_vv_f16m##LMUL##_mu(mask, x, x, x, vl);                                                                  \
        x = __riscv_vfsub_vf_f16m##LMUL(x, (__fp16)1.f, vl);                                                                      \
        e = __riscv_vfsub_vf_f16m##LMUL##_mu(mask, e, e, (__fp16)1.f, vl);                                                        \
                                                                                                                                  \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vv_f16m##LMUL(x, x, vl);                                                            \
                                                                                                                                  \
        vfloat16m##LMUL##_t y = __riscv_vfmul_vf_f16m##LMUL(x, (__fp16)c_cephes_log_p0, vl);                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p1, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p2, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p3, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p4, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p5, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p6, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p7, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_log_p8, vl);                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                                \
                                                                                                                                  \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                                                \
                                                                                                                                  \
        vfloat16m##LMUL##_t tmp = __riscv_vfmul_vf_f16m##LMUL(e, (__fp16)c_cephes_log_q1, vl);                                    \
        y = __riscv_vfadd_vv_f16m##LMUL(y, tmp, vl);                                                                              \
                                                                                                                                  \
        tmp = __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)0.5f, vl);                                                                   \
        y = __riscv_vfsub_vv_f16m##LMUL(y, tmp, vl);                                                                              \
                                                                                                                                  \
        tmp = __riscv_vfmul_vf_f16m##LMUL(e, (__fp16)c_cephes_log_q2, vl);                                                        \
        x = __riscv_vfadd_vv_f16m##LMUL(x, y, vl);                                                                                \
        x = __riscv_vfadd_vv_f16m##LMUL(x, tmp, vl);                                                                              \
        /* negative arg will be NAN */                                                                                            \
        vuint16m##LMUL##_t xtmp = __riscv_vreinterpret_v_f16m##LMUL##_u16m##LMUL(x);                                              \
        x = __riscv_vreinterpret_v_u16m##LMUL##_f16m##LMUL(__riscv_vor_vx_u16m##LMUL##_mu(invalid_mask, xtmp, xtmp, 0xffff, vl)); \
        return x;                                                                                                                 \
    }

_RVV_FLOAT16_LOG_OP(1, 16)
_RVV_FLOAT16_LOG_OP(2, 8)
_RVV_FLOAT16_LOG_OP(4, 4)
_RVV_FLOAT16_LOG_OP(8, 2)

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#define _RVV_FLOAT16_EXP_OP(LMUL, MLEN)                                                                                   \
    static inline vfloat16m##LMUL##_t exp_ps(vfloat16m##LMUL##_t x, size_t vl)                                            \
    {                                                                                                                     \
        vfloat16m##LMUL##_t tmp, fx;                                                                                      \
                                                                                                                          \
        x = __riscv_vfmin_vf_f16m##LMUL(x, (__fp16)c_exp_hi_f16, vl);                                                     \
        x = __riscv_vfmax_vf_f16m##LMUL(x, (__fp16)c_exp_lo_f16, vl);                                                     \
                                                                                                                          \
        /* express exp(x) as exp(g + n*log(2)) */                                                                         \
        fx = __riscv_vfmacc_vf_f16m##LMUL(__riscv_vfmv_v_f_f16m##LMUL((__fp16)0.5f, vl), (__fp16)c_cephes_LOG2EF, x, vl); \
                                                                                                                          \
        /* perform a floorf */                                                                                            \
        tmp = __riscv_vfcvt_f_x_v_f16m##LMUL(__riscv_vfcvt_x_f_v_i16m##LMUL(fx, vl), vl);                                 \
                                                                                                                          \
        /* if greater, substract 1 */                                                                                     \
        vbool##MLEN##_t mask = __riscv_vmfgt_vv_f16m##LMUL##_b##MLEN(tmp, fx, vl);                                        \
        fx = __riscv_vfsub_vf_f16m##LMUL##_mu(mask, tmp, tmp, (__fp16)1.f, vl);                                           \
                                                                                                                          \
        tmp = __riscv_vfmul_vf_f16m##LMUL(fx, (__fp16)c_cephes_exp_C1, vl);                                               \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vf_f16m##LMUL(fx, (__fp16)c_cephes_exp_C2, vl);                             \
        x = __riscv_vfsub_vv_f16m##LMUL(x, tmp, vl);                                                                      \
        x = __riscv_vfsub_vv_f16m##LMUL(x, z, vl);                                                                        \
                                                                                                                          \
        vfloat16m##LMUL##_t y = __riscv_vfmul_vf_f16m##LMUL(x, (__fp16)c_cephes_exp_p0, vl);                              \
        z = __riscv_vfmul_vv_f16m##LMUL(x, x, vl);                                                                        \
                                                                                                                          \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_exp_p1, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_exp_p2, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_exp_p3, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_exp_p4, vl);                                                  \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_cephes_exp_p5, vl);                                                  \
                                                                                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                                        \
        y = __riscv_vfadd_vv_f16m##LMUL(y, x, vl);                                                                        \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)1.f, vl);                                                              \
                                                                                                                          \
        /* build 2^n */                                                                                                   \
        vint16m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i16m##LMUL(fx, vl);                                                    \
        mm = __riscv_vadd_vx_i16m##LMUL(mm, 0xf, vl);                                                                     \
        mm = __riscv_vsll_vx_i16m##LMUL(mm, 10, vl);                                                                      \
        vfloat16m##LMUL##_t pow2n = __riscv_vreinterpret_v_i16m##LMUL##_f16m##LMUL(mm);                                   \
                                                                                                                          \
        y = __riscv_vfmul_vv_f16m##LMUL(y, pow2n, vl);                                                                    \
        return y;                                                                                                         \
    }

_RVV_FLOAT16_EXP_OP(1, 16)
_RVV_FLOAT16_EXP_OP(2, 8)
_RVV_FLOAT16_EXP_OP(4, 4)
_RVV_FLOAT16_EXP_OP(8, 2)

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0        -1.9515295891E-4
#define c_sincof_p1        8.3321608736E-3
#define c_sincof_p2        -1.6666654611E-1
#define c_coscof_p0        2.443315711809948E-005
#define c_coscof_p1        -1.388731625493765E-003
#define c_coscof_p2        4.166664568298827E-002
#define c_cephes_FOPI      1.27323954473516 // 4 / M_PI

#define _RVV_FLOAT16_SINCOS_OP(LMUL, MLEN)                                                                                                                  \
    static inline void sincos_ps(vfloat16m##LMUL##_t x, vfloat16m##LMUL##_t* ysin, vfloat16m##LMUL##_t* ycos, size_t vl)                                    \
    {                                                                                                                                                       \
        /* any x */                                                                                                                                         \
        vfloat16m##LMUL##_t xmm1, xmm2, xmm3, y;                                                                                                            \
                                                                                                                                                            \
        vuint16m##LMUL##_t emm2;                                                                                                                            \
                                                                                                                                                            \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                                                                                                       \
        sign_mask_sin = __riscv_vmflt_vf_f16m##LMUL##_b##MLEN(x, (__fp16)0.f, vl);                                                                          \
        x = __riscv_vfsgnj_vf_f16m##LMUL(x, (__fp16)1.f, vl);                                                                                               \
                                                                                                                                                            \
        /* scale by 4/Pi */                                                                                                                                 \
        y = __riscv_vfmul_vf_f16m##LMUL(x, (__fp16)c_cephes_FOPI, vl);                                                                                      \
                                                                                                                                                            \
        /* store the integer part of y in mm0 */                                                                                                            \
        emm2 = __riscv_vfcvt_xu_f_v_u16m##LMUL(y, vl);                                                                                                      \
        /* j=(j+1) & (~1) (see the cephes sources) */                                                                                                       \
        emm2 = __riscv_vadd_vx_u16m##LMUL(emm2, 1, vl);                                                                                                     \
        emm2 = __riscv_vand_vx_u16m##LMUL(emm2, ~1, vl);                                                                                                    \
        y = __riscv_vfcvt_f_xu_v_f16m##LMUL(emm2, vl);                                                                                                      \
                                                                                                                                                            \
        /* get the polynom selection mask              */                                                                                                   \
        /*     there is one polynom for 0 <= x <= Pi/4 */                                                                                                   \
        /*     and another one for Pi/4<x<=Pi/2        */                                                                                                   \
        /*                                             */                                                                                                   \
        /*     Both branches will be computed.         */                                                                                                   \
        vbool##MLEN##_t poly_mask = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(emm2, 2, vl), 0, vl);                                  \
                                                                                                                                                            \
        /* The magic pass: "Extended precision modular arithmetic" */                                                                                       \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */                                                                                       \
        xmm1 = __riscv_vfmul_vf_f16m##LMUL(y, (__fp16)c_minus_cephes_DP1, vl);                                                                              \
        xmm2 = __riscv_vfmul_vf_f16m##LMUL(y, (__fp16)c_minus_cephes_DP2, vl);                                                                              \
        xmm3 = __riscv_vfmul_vf_f16m##LMUL(y, (__fp16)c_minus_cephes_DP3, vl);                                                                              \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm1, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm2, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f16m##LMUL(x, xmm3, vl);                                                                                                       \
                                                                                                                                                            \
        sign_mask_sin = __riscv_vmxor_mm_b##MLEN(sign_mask_sin, __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(emm2, 4, vl), 0, vl), vl); \
        sign_mask_cos = __riscv_vmsne_vx_u16m##LMUL##_b##MLEN(__riscv_vand_vx_u16m##LMUL(__riscv_vsub_vx_u16m##LMUL(emm2, 2, vl), 4, vl), 0, vl);           \
                                                                                                                                                            \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */                                                                                           \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */                                                                                           \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vv_f16m##LMUL(x, x, vl);                                                                                      \
        vfloat16m##LMUL##_t y1, y2;                                                                                                                         \
                                                                                                                                                            \
        y1 = __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)c_coscof_p0, vl);                                                                                       \
        y2 = __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)c_sincof_p0, vl);                                                                                       \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (__fp16)c_coscof_p1, vl);                                                                                      \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (__fp16)c_sincof_p1, vl);                                                                                      \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (__fp16)c_coscof_p2, vl);                                                                                      \
        y2 = __riscv_vfadd_vf_f16m##LMUL(y2, (__fp16)c_sincof_p2, vl);                                                                                      \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfmul_vv_f16m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f16m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfsub_vv_f16m##LMUL(y1, __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)0.5f, vl), vl);                                                         \
        y2 = __riscv_vfadd_vv_f16m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f16m##LMUL(y1, (__fp16)1.f, vl);                                                                                              \
                                                                                                                                                            \
        /* select the correct result from the two polynoms */                                                                                               \
        vfloat16m##LMUL##_t ys = __riscv_vmerge_vvm_f16m##LMUL(y2, y1, poly_mask, vl);                                                                      \
        vfloat16m##LMUL##_t yc = __riscv_vmerge_vvm_f16m##LMUL(y1, y2, poly_mask, vl);                                                                      \
        *ysin = __riscv_vmerge_vvm_f16m##LMUL(ys, __riscv_vfneg_v_f16m##LMUL(ys, vl), sign_mask_sin, vl);                                                   \
        *ycos = __riscv_vmerge_vvm_f16m##LMUL(__riscv_vfneg_v_f16m##LMUL(yc, vl), yc, sign_mask_cos, vl);                                                   \
    }

_RVV_FLOAT16_SINCOS_OP(1, 16)
_RVV_FLOAT16_SINCOS_OP(2, 8)
_RVV_FLOAT16_SINCOS_OP(4, 4)
_RVV_FLOAT16_SINCOS_OP(8, 2)

#define _RVV_FLOAT16_SIN_OP(LMUL, MLEN)                                        \
    static inline vfloat16m##LMUL##_t sin_ps(vfloat16m##LMUL##_t x, size_t vl) \
    {                                                                          \
        vfloat16m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ysin;                                                           \
    }

_RVV_FLOAT16_SIN_OP(1, 16)
_RVV_FLOAT16_SIN_OP(2, 8)
_RVV_FLOAT16_SIN_OP(4, 4)
_RVV_FLOAT16_SIN_OP(8, 2)

#define _RVV_FLOAT16_COS_OP(LMUL, MLEN)                                        \
    static inline vfloat16m##LMUL##_t cos_ps(vfloat16m##LMUL##_t x, size_t vl) \
    {                                                                          \
        vfloat16m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ycos;                                                           \
    }

_RVV_FLOAT16_COS_OP(1, 16)
_RVV_FLOAT16_COS_OP(2, 8)
_RVV_FLOAT16_COS_OP(4, 4)
_RVV_FLOAT16_COS_OP(8, 2)

#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

#define _RVV_FLOAT16_TANH_OP(LMUL, MLEN)                                                                \
    static inline vfloat16m##LMUL##_t tanh_ps(vfloat16m##LMUL##_t x, size_t vl)                         \
    {                                                                                                   \
        vfloat16m##LMUL##_t x2 = __riscv_vfsgnj_vf_f16m##LMUL(x, (__fp16)1.f, vl);                      \
                                                                                                        \
        vbool##MLEN##_t tiny_mask = __riscv_vmfge_vf_f16m##LMUL##_b##MLEN(x2, (__fp16)c_tanh_tiny, vl); \
                                                                                                        \
        /* clamp the inputs to the range [-9, 9] since anything outside */                              \
        /* this range is -/+1.0f in single-precision.                   */                              \
        x2 = __riscv_vfmin_vf_f16m##LMUL(x2, (__fp16)c_tanh_hi, vl);                                    \
                                                                                                        \
        /* since the polynomials are odd/even, we need x**2. */                                         \
        vfloat16m##LMUL##_t z = __riscv_vfmul_vv_f16m##LMUL(x2, x2, vl);                                \
                                                                                                        \
        /* evaluate the numerator polynomial y. */                                                      \
        vfloat16m##LMUL##_t y = __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)c_tanh_alpha_13, vl);            \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_11, vl);                                \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_9, vl);                                 \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_7, vl);                                 \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_5, vl);                                 \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_3, vl);                                 \
        y = __riscv_vfmul_vv_f16m##LMUL(y, z, vl);                                                      \
        y = __riscv_vfadd_vf_f16m##LMUL(y, (__fp16)c_tanh_alpha_1, vl);                                 \
        y = __riscv_vfmul_vv_f16m##LMUL(y, x2, vl);                                                     \
                                                                                                        \
        /* evaluate the denominator polynomial w. */                                                    \
        vfloat16m##LMUL##_t w = __riscv_vfmul_vf_f16m##LMUL(z, (__fp16)c_tanh_beta_6, vl);              \
        w = __riscv_vfadd_vf_f16m##LMUL(w, (__fp16)c_tanh_beta_4, vl);                                  \
        w = __riscv_vfmul_vv_f16m##LMUL(w, z, vl);                                                      \
        w = __riscv_vfadd_vf_f16m##LMUL(w, (__fp16)c_tanh_beta_2, vl);                                  \
        w = __riscv_vfmul_vv_f16m##LMUL(w, z, vl);                                                      \
        w = __riscv_vfadd_vf_f16m##LMUL(w, (__fp16)c_tanh_beta_0, vl);                                  \
                                                                                                        \
        /* divide the numerator by the denominator. */                                                  \
        y = __riscv_vfdiv_vv_f16m##LMUL(y, w, vl);                                                      \
                                                                                                        \
        /* reinstate the sign.  */                                                                      \
        y = __riscv_vfsgnj_vv_f16m##LMUL(y, x, vl);                                                     \
                                                                                                        \
        /* when the argument is very small in magnitude it's more accurate to just return it. */        \
        y = __riscv_vmerge_vvm_f16m##LMUL(x, y, tiny_mask, vl);                                         \
                                                                                                        \
        return y;                                                                                       \
    }

_RVV_FLOAT16_TANH_OP(1, 16)
_RVV_FLOAT16_TANH_OP(2, 8)
_RVV_FLOAT16_TANH_OP(4, 4)
_RVV_FLOAT16_TANH_OP(8, 2)

#define _RVV_FLOAT16_POW_OP(LMUL, MLEN)                                                               \
    static inline vfloat16m##LMUL##_t pow_ps(vfloat16m##LMUL##_t a, vfloat16m##LMUL##_t b, size_t vl) \
    {                                                                                                 \
        /* pow(x, m) = exp(m * log(x)) */                                                             \
        return exp_ps(__riscv_vfmul_vv_f16m##LMUL(b, log_ps(a, vl), vl), vl);                         \
    }

_RVV_FLOAT16_POW_OP(1, 16)
_RVV_FLOAT16_POW_OP(2, 8)
_RVV_FLOAT16_POW_OP(4, 4)
_RVV_FLOAT16_POW_OP(8, 2)

#if __riscv_xtheadvector
#define _RVV_FLOAT16_SIGMOID_OP(LMUL, MLEN)                                                  \
    static inline vfloat16m##LMUL##_t sigmoid_ps(vfloat16m##LMUL##_t _v, size_t vl)          \
    {                                                                                        \
        _v = __riscv_vfneg_v_f16m##LMUL(_v, vl);                                             \
        _v = exp_ps(_v, vl);                                                                 \
        _v = __riscv_vfadd_vf_f16m##LMUL(_v, (__fp16)1.f, vl);                               \
        vfloat16m##LMUL##_t _reciprocal = __riscv_vfrdiv_vf_f16m##LMUL(_v, (__fp16)1.f, vl); \
        return _reciprocal;                                                                  \
    }
#else // __riscv_xtheadvector
#define _RVV_FLOAT16_SIGMOID_OP(LMUL, MLEN)                                                                                                                                \
    static inline vfloat16m##LMUL##_t sigmoid_ps(vfloat16m##LMUL##_t _v, size_t vl)                                                                                        \
    {                                                                                                                                                                      \
        _v = __riscv_vfneg_v_f16m##LMUL(_v, vl);                                                                                                                           \
        _v = exp_ps(_v, vl);                                                                                                                                               \
        _v = __riscv_vfadd_vf_f16m##LMUL(_v, (__fp16)1.f, vl);                                                                                                             \
        vfloat16m##LMUL##_t _reciprocal = __riscv_vfrec7_v_f16m##LMUL(_v, vl);                                                                                             \
        _reciprocal = __riscv_vfmul_vv_f16m##LMUL(__riscv_vfrsub_vf_f16m##LMUL(__riscv_vfmul_vv_f16m##LMUL(_v, _reciprocal, vl), (__fp16)2.f, vl), _reciprocal, vl);       \
        /* _reciprocal = __riscv_vfmul_vv_f16m##LMUL(__riscv_vfrsub_vf_f16m##LMUL(__riscv_vfmul_vv_f16m##LMUL(_v, _reciprocal, vl), (__fp16)2.f, vl), _reciprocal, vl); */ \
        return _reciprocal;                                                                                                                                                \
    }
#endif // __riscv_xtheadvector

_RVV_FLOAT16_SIGMOID_OP(1, 16)
_RVV_FLOAT16_SIGMOID_OP(2, 8)
_RVV_FLOAT16_SIGMOID_OP(4, 4)
_RVV_FLOAT16_SIGMOID_OP(8, 2)

//TODO rvv optimize
#define _RVV_FLOAT16_ATAN2_OP(LMUL, MLEN)                                                               \
    static inline vfloat16m##LMUL##_t atan2_ps(vfloat16m##LMUL##_t a, vfloat16m##LMUL##_t b, size_t vl) \
    {                                                                                                   \
        std::vector<__fp16> tmpx(vl);                                                                   \
        std::vector<__fp16> tmpy(vl);                                                                   \
        __riscv_vse16_v_f16m##LMUL(tmpx.data(), a, vl);                                                 \
        __riscv_vse16_v_f16m##LMUL(tmpy.data(), b, vl);                                                 \
        for (size_t i = 0; i < vl; i++)                                                                 \
        {                                                                                               \
            tmpx[i] = (__fp16)atan2((float)tmpx[i], (float)tmpy[i]);                                    \
        }                                                                                               \
        return __riscv_vle16_v_f16m##LMUL(tmpx.data(), vl);                                             \
    }

_RVV_FLOAT16_ATAN2_OP(1, 16)
_RVV_FLOAT16_ATAN2_OP(2, 8)
_RVV_FLOAT16_ATAN2_OP(4, 4)
_RVV_FLOAT16_ATAN2_OP(8, 2)

#endif // RVV_MATHFUN_FP16S_H
