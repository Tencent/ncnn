// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef RVV_MATHFUN_H
#define RVV_MATHFUN_H

#include <riscv_vector.h>

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                                                                               \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x, size_t vl)                                                        \
    {                                                                                                                                 \
        x = __riscv_vfmax_vf_f32m##LMUL(x, 0.f, vl); /* force flush to zero on denormal values */                                     \
        vbool##MLEN##_t invalid_mask = __riscv_vmfle_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                                             \
                                                                                                                                      \
        vint32m##LMUL##_t ux = __riscv_vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);                                                     \
                                                                                                                                      \
        vint32m##LMUL##_t emm0 = __riscv_vsra_vx_i32m##LMUL(ux, 23, vl);                                                              \
                                                                                                                                      \
        /* keep only the fractional part */                                                                                           \
        ux = __riscv_vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);                                                                     \
        ux = __riscv_vor_vx_i32m##LMUL(ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);                                          \
        x = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                                                                       \
                                                                                                                                      \
        emm0 = __riscv_vsub_vx_i32m##LMUL(emm0, 0x7f, vl);                                                                            \
        vfloat32m##LMUL##_t e = __riscv_vfcvt_f_x_v_f32m##LMUL(emm0, vl);                                                             \
                                                                                                                                      \
        e = __riscv_vfadd_vf_f32m##LMUL(e, 1.f, vl);                                                                                  \
                                                                                                                                      \
        /* part2:                      */                                                                                             \
        /*     if( x < SQRTHF ) {      */                                                                                             \
        /*       e -= 1;               */                                                                                             \
        /*       x = x + x - 1.0;      */                                                                                             \
        /*     } else { x = x - 1.0; } */                                                                                             \
        vbool##MLEN##_t mask = __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);                                         \
        x = __riscv_vfadd_vv_f32m##LMUL##_mu(mask, x, x, x, vl);                                                                      \
        x = __riscv_vfsub_vf_f32m##LMUL(x, 1.f, vl);                                                                                  \
        e = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, e, e, 1.f, vl);                                                                    \
                                                                                                                                      \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                                \
                                                                                                                                      \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_log_p0, vl);                                                  \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p1, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p2, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p3, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p4, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p5, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p6, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p7, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p8, vl);                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                                    \
                                                                                                                                      \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                                                                    \
                                                                                                                                      \
        vfloat32m##LMUL##_t tmp = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q1, vl);                                                \
        y = __riscv_vfadd_vv_f32m##LMUL(y, tmp, vl);                                                                                  \
                                                                                                                                      \
        tmp = __riscv_vfmul_vf_f32m##LMUL(z, 0.5f, vl);                                                                               \
        y = __riscv_vfsub_vv_f32m##LMUL(y, tmp, vl);                                                                                  \
                                                                                                                                      \
        tmp = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q2, vl);                                                                    \
        x = __riscv_vfadd_vv_f32m##LMUL(x, y, vl);                                                                                    \
        x = __riscv_vfadd_vv_f32m##LMUL(x, tmp, vl);                                                                                  \
        /* negative arg will be NAN */                                                                                                \
        vuint32m##LMUL##_t xtmp = __riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(x);                                                  \
        x = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(__riscv_vor_vx_u32m##LMUL##_mu(invalid_mask, xtmp, xtmp, 0xffffffff, vl)); \
        return x;                                                                                                                     \
    }

_RVV_FLOAT32_LOG_OP(1, 32)
_RVV_FLOAT32_LOG_OP(2, 16)
_RVV_FLOAT32_LOG_OP(4, 8)
_RVV_FLOAT32_LOG_OP(8, 4)

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

#define _RVV_FLOAT32_EXP_OP(LMUL, MLEN)                                                                   \
    static inline vfloat32m##LMUL##_t exp_ps(vfloat32m##LMUL##_t x, size_t vl)                            \
    {                                                                                                     \
        vfloat32m##LMUL##_t tmp, fx;                                                                      \
                                                                                                          \
        x = __riscv_vfmin_vf_f32m##LMUL(x, c_exp_hi, vl);                                                 \
        x = __riscv_vfmax_vf_f32m##LMUL(x, c_exp_lo, vl);                                                 \
                                                                                                          \
        /* express exp(x) as exp(g + n*log(2)) */                                                         \
        fx = __riscv_vfmacc_vf_f32m##LMUL(__riscv_vfmv_v_f_f32m##LMUL(0.5f, vl), c_cephes_LOG2EF, x, vl); \
                                                                                                          \
        /* perform a floorf */                                                                            \
        tmp = __riscv_vfcvt_f_x_v_f32m##LMUL(__riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl), vl);                 \
                                                                                                          \
        /* if greater, substract 1 */                                                                     \
        vbool##MLEN##_t mask = __riscv_vmfgt_vv_f32m##LMUL##_b##MLEN(tmp, fx, vl);                        \
        fx = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, tmp, tmp, 1.f, vl);                                   \
                                                                                                          \
        tmp = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C1, vl);                                       \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C2, vl);                     \
        x = __riscv_vfsub_vv_f32m##LMUL(x, tmp, vl);                                                      \
        x = __riscv_vfsub_vv_f32m##LMUL(x, z, vl);                                                        \
                                                                                                          \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_exp_p0, vl);                      \
        z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                        \
                                                                                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p1, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p2, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p3, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p4, vl);                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p5, vl);                                          \
                                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                                        \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                                                        \
        y = __riscv_vfadd_vf_f32m##LMUL(y, 1.f, vl);                                                      \
                                                                                                          \
        /* build 2^n */                                                                                   \
        vint32m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl);                                    \
        mm = __riscv_vadd_vx_i32m##LMUL(mm, 0x7f, vl);                                                    \
        mm = __riscv_vsll_vx_i32m##LMUL(mm, 23, vl);                                                      \
        vfloat32m##LMUL##_t pow2n = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(mm);                   \
                                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, pow2n, vl);                                                    \
        return y;                                                                                         \
    }

_RVV_FLOAT32_EXP_OP(1, 32)
_RVV_FLOAT32_EXP_OP(2, 16)
_RVV_FLOAT32_EXP_OP(4, 8)
_RVV_FLOAT32_EXP_OP(8, 4)

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

#define _RVV_FLOAT32_SINCOS_OP(LMUL, MLEN)                                                                                                                  \
    static inline void sincos_ps(vfloat32m##LMUL##_t x, vfloat32m##LMUL##_t* ysin, vfloat32m##LMUL##_t* ycos, size_t vl)                                    \
    {                                                                                                                                                       \
        /* any x */                                                                                                                                         \
        vfloat32m##LMUL##_t xmm1, xmm2, xmm3, y;                                                                                                            \
                                                                                                                                                            \
        vuint32m##LMUL##_t emm2;                                                                                                                            \
                                                                                                                                                            \
        vbool##MLEN##_t sign_mask_sin, sign_mask_cos;                                                                                                       \
        sign_mask_sin = __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                                                                                  \
        x = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.f, vl);                                                                                                       \
                                                                                                                                                            \
        /* scale by 4/Pi */                                                                                                                                 \
        y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_FOPI, vl);                                                                                              \
                                                                                                                                                            \
        /* store the integer part of y in mm0 */                                                                                                            \
        emm2 = __riscv_vfcvt_xu_f_v_u32m##LMUL(y, vl);                                                                                                      \
        /* j=(j+1) & (~1) (see the cephes sources) */                                                                                                       \
        emm2 = __riscv_vadd_vx_u32m##LMUL(emm2, 1, vl);                                                                                                     \
        emm2 = __riscv_vand_vx_u32m##LMUL(emm2, ~1, vl);                                                                                                    \
        y = __riscv_vfcvt_f_xu_v_f32m##LMUL(emm2, vl);                                                                                                      \
                                                                                                                                                            \
        /* get the polynom selection mask              */                                                                                                   \
        /*     there is one polynom for 0 <= x <= Pi/4 */                                                                                                   \
        /*     and another one for Pi/4<x<=Pi/2        */                                                                                                   \
        /*                                             */                                                                                                   \
        /*     Both branches will be computed.         */                                                                                                   \
        vbool##MLEN##_t poly_mask = __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(__riscv_vand_vx_u32m##LMUL(emm2, 2, vl), 0, vl);                                  \
                                                                                                                                                            \
        /* The magic pass: "Extended precision modular arithmetic" */                                                                                       \
        /*     x = ((x - y * DP1) - y * DP2) - y * DP3;            */                                                                                       \
        xmm1 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP1, vl);                                                                                      \
        xmm2 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP2, vl);                                                                                      \
        xmm3 = __riscv_vfmul_vf_f32m##LMUL(y, c_minus_cephes_DP3, vl);                                                                                      \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm1, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm2, vl);                                                                                                       \
        x = __riscv_vfadd_vv_f32m##LMUL(x, xmm3, vl);                                                                                                       \
                                                                                                                                                            \
        sign_mask_sin = __riscv_vmxor_mm_b##MLEN(sign_mask_sin, __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(__riscv_vand_vx_u32m##LMUL(emm2, 4, vl), 0, vl), vl); \
        sign_mask_cos = __riscv_vmsne_vx_u32m##LMUL##_b##MLEN(__riscv_vand_vx_u32m##LMUL(__riscv_vsub_vx_u32m##LMUL(emm2, 2, vl), 4, vl), 0, vl);           \
                                                                                                                                                            \
        /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1, */                                                                                           \
        /*     and the second polynom  (Pi/4 <= x <= 0) in y2  */                                                                                           \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                                                      \
        vfloat32m##LMUL##_t y1, y2;                                                                                                                         \
                                                                                                                                                            \
        y1 = __riscv_vfmul_vf_f32m##LMUL(z, c_coscof_p0, vl);                                                                                               \
        y2 = __riscv_vfmul_vf_f32m##LMUL(z, c_sincof_p0, vl);                                                                                               \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, c_coscof_p1, vl);                                                                                              \
        y2 = __riscv_vfadd_vf_f32m##LMUL(y2, c_sincof_p1, vl);                                                                                              \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, c_coscof_p2, vl);                                                                                              \
        y2 = __riscv_vfadd_vf_f32m##LMUL(y2, c_sincof_p2, vl);                                                                                              \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, z, vl);                                                                                                        \
        y1 = __riscv_vfmul_vv_f32m##LMUL(y1, z, vl);                                                                                                        \
        y2 = __riscv_vfmul_vv_f32m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfsub_vv_f32m##LMUL(y1, __riscv_vfmul_vf_f32m##LMUL(z, 0.5f, vl), vl);                                                                 \
        y2 = __riscv_vfadd_vv_f32m##LMUL(y2, x, vl);                                                                                                        \
        y1 = __riscv_vfadd_vf_f32m##LMUL(y1, 1.f, vl);                                                                                                      \
                                                                                                                                                            \
        /* select the correct result from the two polynoms */                                                                                               \
        vfloat32m##LMUL##_t ys = __riscv_vmerge_vvm_f32m##LMUL(y2, y1, poly_mask, vl);                                                                      \
        vfloat32m##LMUL##_t yc = __riscv_vmerge_vvm_f32m##LMUL(y1, y2, poly_mask, vl);                                                                      \
        *ysin = __riscv_vmerge_vvm_f32m##LMUL(ys, __riscv_vfneg_v_f32m##LMUL(ys, vl), sign_mask_sin, vl);                                                   \
        *ycos = __riscv_vmerge_vvm_f32m##LMUL(__riscv_vfneg_v_f32m##LMUL(yc, vl), yc, sign_mask_cos, vl);                                                   \
    }

_RVV_FLOAT32_SINCOS_OP(1, 32)
_RVV_FLOAT32_SINCOS_OP(2, 16)
_RVV_FLOAT32_SINCOS_OP(4, 8)
_RVV_FLOAT32_SINCOS_OP(8, 4)

#define _RVV_FLOAT32_SIN_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t sin_ps(vfloat32m##LMUL##_t x, size_t vl) \
    {                                                                          \
        vfloat32m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ysin;                                                           \
    }

_RVV_FLOAT32_SIN_OP(1, 32)
_RVV_FLOAT32_SIN_OP(2, 16)
_RVV_FLOAT32_SIN_OP(4, 8)
_RVV_FLOAT32_SIN_OP(8, 4)

#define _RVV_FLOAT32_COS_OP(LMUL, MLEN)                                        \
    static inline vfloat32m##LMUL##_t cos_ps(vfloat32m##LMUL##_t x, size_t vl) \
    {                                                                          \
        vfloat32m##LMUL##_t ysin, ycos;                                        \
        sincos_ps(x, &ysin, &ycos, vl);                                        \
        return ycos;                                                           \
    }

_RVV_FLOAT32_COS_OP(1, 32)
_RVV_FLOAT32_COS_OP(2, 16)
_RVV_FLOAT32_COS_OP(4, 8)
_RVV_FLOAT32_COS_OP(8, 4)

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

#define _RVV_FLOAT32_TANH_OP(LMUL, MLEN)                                                         \
    static inline vfloat32m##LMUL##_t tanh_ps(vfloat32m##LMUL##_t x, size_t vl)                  \
    {                                                                                            \
        vfloat32m##LMUL##_t x2 = __riscv_vfsgnj_vf_f32m##LMUL(x, 1.f, vl);                       \
                                                                                                 \
        vbool##MLEN##_t tiny_mask = __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x2, c_tanh_tiny, vl);  \
                                                                                                 \
        /* clamp the inputs to the range [-9, 9] since anything outside */                       \
        /* this range is -/+1.0f in single-precision.                   */                       \
        x2 = __riscv_vfmin_vf_f32m##LMUL(x2, c_tanh_hi, vl);                                     \
                                                                                                 \
        /* since the polynomials are odd/even, we need x**2. */                                  \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x2, x2, vl);                         \
                                                                                                 \
        /* evaluate the numerator polynomial y. */                                               \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(z, c_tanh_alpha_13, vl);             \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_11, vl);                                 \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_9, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_7, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_5, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_3, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                               \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_tanh_alpha_1, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x2, vl);                                              \
                                                                                                 \
        /* evaluate the denominator polynomial w. */                                             \
        vfloat32m##LMUL##_t w = __riscv_vfmul_vf_f32m##LMUL(z, c_tanh_beta_6, vl);               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_4, vl);                                   \
        w = __riscv_vfmul_vv_f32m##LMUL(w, z, vl);                                               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_2, vl);                                   \
        w = __riscv_vfmul_vv_f32m##LMUL(w, z, vl);                                               \
        w = __riscv_vfadd_vf_f32m##LMUL(w, c_tanh_beta_0, vl);                                   \
                                                                                                 \
        /* divide the numerator by the denominator. */                                           \
        y = __riscv_vfdiv_vv_f32m##LMUL(y, w, vl);                                               \
                                                                                                 \
        /* reinstate the sign.  */                                                               \
        y = __riscv_vfsgnj_vv_f32m##LMUL(y, x, vl);                                              \
                                                                                                 \
        /* when the argument is very small in magnitude it's more accurate to just return it. */ \
        y = __riscv_vmerge_vvm_f32m##LMUL(x, y, tiny_mask, vl);                                  \
                                                                                                 \
        return y;                                                                                \
    }

_RVV_FLOAT32_TANH_OP(1, 32)
_RVV_FLOAT32_TANH_OP(2, 16)
_RVV_FLOAT32_TANH_OP(4, 8)
_RVV_FLOAT32_TANH_OP(8, 4)

#define _RVV_FLOAT32_POW_OP(LMUL, MLEN)                                                               \
    static inline vfloat32m##LMUL##_t pow_ps(vfloat32m##LMUL##_t a, vfloat32m##LMUL##_t b, size_t vl) \
    {                                                                                                 \
        /* pow(x, m) = exp(m * log(x)) */                                                             \
        return exp_ps(__riscv_vfmul_vv_f32m##LMUL(b, log_ps(a, vl), vl), vl);                         \
    }

_RVV_FLOAT32_POW_OP(1, 32)
_RVV_FLOAT32_POW_OP(2, 16)
_RVV_FLOAT32_POW_OP(4, 8)
_RVV_FLOAT32_POW_OP(8, 4)

#if __riscv_xtheadvector
#define _RVV_FLOAT32_SIGMOID_OP(LMUL, MLEN)                                          \
    static inline vfloat32m##LMUL##_t sigmoid_ps(vfloat32m##LMUL##_t _v, size_t vl)  \
    {                                                                                \
        _v = __riscv_vfneg_v_f32m##LMUL(_v, vl);                                     \
        _v = exp_ps(_v, vl);                                                         \
        _v = __riscv_vfadd_vf_f32m##LMUL(_v, 1.f, vl);                               \
        vfloat32m##LMUL##_t _reciprocal = __riscv_vfrdiv_vf_f32m##LMUL(_v, 1.f, vl); \
        return _reciprocal;                                                          \
    }
#else // __riscv_xtheadvector
#define _RVV_FLOAT32_SIGMOID_OP(LMUL, MLEN)                                                                                                                        \
    static inline vfloat32m##LMUL##_t sigmoid_ps(vfloat32m##LMUL##_t _v, size_t vl)                                                                                \
    {                                                                                                                                                              \
        _v = __riscv_vfneg_v_f32m##LMUL(_v, vl);                                                                                                                   \
        _v = exp_ps(_v, vl);                                                                                                                                       \
        _v = __riscv_vfadd_vf_f32m##LMUL(_v, 1.f, vl);                                                                                                             \
        vfloat32m##LMUL##_t _reciprocal = __riscv_vfrec7_v_f32m##LMUL(_v, vl);                                                                                     \
        _reciprocal = __riscv_vfmul_vv_f32m##LMUL(__riscv_vfrsub_vf_f32m##LMUL(__riscv_vfmul_vv_f32m##LMUL(_v, _reciprocal, vl), 2.f, vl), _reciprocal, vl);       \
        /* _reciprocal = __riscv_vfmul_vv_f32m##LMUL(__riscv_vfrsub_vf_f32m##LMUL(__riscv_vfmul_vv_f32m##LMUL(_v, _reciprocal, vl), 2.f, vl), _reciprocal, vl); */ \
        return _reciprocal;                                                                                                                                        \
    }
#endif // __riscv_xtheadvector

_RVV_FLOAT32_SIGMOID_OP(1, 32)
_RVV_FLOAT32_SIGMOID_OP(2, 16)
_RVV_FLOAT32_SIGMOID_OP(4, 8)
_RVV_FLOAT32_SIGMOID_OP(8, 4)

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
#define c_erfc_erx_f 8.4506291151e-01f /* 0x3f58560b */
// Coefficients for approximation to  erf on [00.84375]
#define c_erfc_efx  1.2837916613e-01f /* 0x3e0375d4 */
#define c_erfc_efx8 1.0270333290e+00f /* 0x3f8375d4 */

#define c_erfc_pp0 1.2837916613e-01f  /* 0x3e0375d4 */
#define c_erfc_pp1 -3.2504209876e-01f /* 0xbea66beb */
#define c_erfc_pp2 -2.8481749818e-02f /* 0xbce9528f */
#define c_erfc_pp3 -5.7702702470e-03f /* 0xbbbd1489 */
#define c_erfc_pp4 -2.3763017452e-05f /* 0xb7c756b1 */
#define c_erfc_qq1 3.9791721106e-01f  /* 0x3ecbbbce */
#define c_erfc_qq2 6.5022252500e-02f  /* 0x3d852a63 */
#define c_erfc_qq3 5.0813062117e-03f  /* 0x3ba68116 */
#define c_erfc_qq4 1.3249473704e-04f  /* 0x390aee49 */
#define c_erfc_qq5 -3.9602282413e-06f /* 0xb684e21a */

// Coefficients for approximation to  erf  in [0.843751.25]
#define c_erfc_pa0 -2.3621185683e-03f /* 0xbb1acdc6 */
#define c_erfc_pa1 4.1485610604e-01f  /* 0x3ed46805 */
#define c_erfc_pa2 -3.7220788002e-01f /* 0xbebe9208 */
#define c_erfc_pa3 3.1834661961e-01f  /* 0x3ea2fe54 */
#define c_erfc_pa4 -1.1089469492e-01f /* 0xbde31cc2 */
#define c_erfc_pa5 3.5478305072e-02f  /* 0x3d1151b3 */
#define c_erfc_pa6 -2.1663755178e-03f /* 0xbb0df9c0 */
#define c_erfc_qa1 1.0642088205e-01f  /* 0x3dd9f331 */
#define c_erfc_qa2 5.4039794207e-01f  /* 0x3f0a5785 */
#define c_erfc_qa3 7.1828655899e-02f  /* 0x3d931ae7 */
#define c_erfc_qa4 1.2617121637e-01f  /* 0x3e013307 */
#define c_erfc_qa5 1.3637083583e-02f  /* 0x3c5f6e13 */
#define c_erfc_qa6 1.1984500103e-02f  /* 0x3c445aa3 */

// Coefficients for approximation to  erfc in [1.251/0.35]
#define c_erfc_ra0 -9.8649440333e-03f /* 0xbc21a093 */
#define c_erfc_ra1 -6.9385856390e-01f /* 0xbf31a0b7 */
#define c_erfc_ra2 -1.0558626175e+01f /* 0xc128f022 */
#define c_erfc_ra3 -6.2375331879e+01f /* 0xc2798057 */
#define c_erfc_ra4 -1.6239666748e+02f /* 0xc322658c */
#define c_erfc_ra5 -1.8460508728e+02f /* 0xc3389ae7 */
#define c_erfc_ra6 -8.1287437439e+01f /* 0xc2a2932b */
#define c_erfc_ra7 -9.8143291473e+00f /* 0xc11d077e */
#define c_erfc_sa1 1.9651271820e+01f  /* 0x419d35ce */
#define c_erfc_sa2 1.3765776062e+02f  /* 0x4309a863 */
#define c_erfc_sa3 4.3456588745e+02f  /* 0x43d9486f */
#define c_erfc_sa4 6.4538726807e+02f  /* 0x442158c9 */
#define c_erfc_sa5 4.2900814819e+02f  /* 0x43d6810b */
#define c_erfc_sa6 1.0863500214e+02f  /* 0x42d9451f */
#define c_erfc_sa7 6.5702495575e+00f  /* 0x40d23f7c */
#define c_erfc_sa8 -6.0424413532e-02f /* 0xbd777f97 */

// Coefficients for approximation to  erfc in [1/.3528]

#define c_erfc_rb0 -9.8649431020e-03f /* 0xbc21a092 */
#define c_erfc_rb1 -7.9928326607e-01f /* 0xbf4c9dd4 */
#define c_erfc_rb2 -1.7757955551e+01f /* 0xc18e104b */
#define c_erfc_rb3 -1.6063638306e+02f /* 0xc320a2ea */
#define c_erfc_rb4 -6.3756646729e+02f /* 0xc41f6441 */
#define c_erfc_rb5 -1.0250950928e+03f /* 0xc480230b */
#define c_erfc_rb6 -4.8351919556e+02f /* 0xc3f1c275 */
#define c_erfc_sb1 3.0338060379e+01f  /* 0x41f2b459 */
#define c_erfc_sb2 3.2579251099e+02f  /* 0x43a2e571 */
#define c_erfc_sb3 1.5367296143e+03f  /* 0x44c01759 */
#define c_erfc_sb4 3.1998581543e+03f  /* 0x4547fdbb */
#define c_erfc_sb5 2.5530502930e+03f  /* 0x451f90ce */
#define c_erfc_sb6 4.7452853394e+02f  /* 0x43ed43a7 */
#define c_erfc_sb7 -2.2440952301e+01f /* 0xc1b38712 */

#define _RVV_FLOAT32_FMA_HELPER(LMUL)                                                                             \
    static inline vfloat32m##LMUL##_t __riscv_vfmadd_vff_f32m##LMUL(vfloat32m##LMUL##_t a, float b,               \
                                                                    float c, size_t vl)                           \
    {                                                                                                             \
        vfloat32m##LMUL##_t ret = __riscv_vfmul_vf_f32m##LMUL(a, b, vl);                                          \
        ret = __riscv_vfadd_vf_f32m##LMUL(ret, c, vl);                                                            \
        return ret;                                                                                               \
    }                                                                                                             \
                                                                                                                  \
    static inline vfloat32m##LMUL##_t __riscv_vfmadd_vvf_f32m##LMUL(vfloat32m##LMUL##_t a, vfloat32m##LMUL##_t b, \
                                                                    float c, size_t vl)                           \
    {                                                                                                             \
        vfloat32m##LMUL##_t ret = __riscv_vfmul_vv_f32m##LMUL(a, b, vl);                                          \
        ret = __riscv_vfadd_vf_f32m##LMUL(ret, c, vl);                                                            \
        return ret;                                                                                               \
    }

_RVV_FLOAT32_FMA_HELPER(8)
_RVV_FLOAT32_FMA_HELPER(4)
_RVV_FLOAT32_FMA_HELPER(2)
_RVV_FLOAT32_FMA_HELPER(1)

#define _RVV_FLOAT32_ERFC_OP(LMUL, MLEN)                                                                                                                                                                                                                                                                                                                                                                   \
    static inline vfloat32m##LMUL##_t erfc_ps(vfloat32m##LMUL##_t x, size_t vl)                                                                                                                                                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                                                                                                                                                                                      \
        /* Argument for polys */                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t absx = __riscv_vfsgnjx_vv_f32m##LMUL(x, x, vl);                                                                                                                                                                                                                                                                                                                                \
        vfloat32m##LMUL##_t x2 = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                                                                                                                                                                                                                                                                                                    \
        vfloat32m##LMUL##_t t = __riscv_vfrdiv_vf_f32m##LMUL(x2, 1.0f, vl);                                                                                                                                                                                                                                                                                                                                \
        vfloat32m##LMUL##_t tt = __riscv_vfsub_vf_f32m##LMUL(absx, 1.0f, vl);                                                                                                                                                                                                                                                                                                                              \
        /* absx < 1.25f ? tt:t */                                                                                                                                                                                                                                                                                                                                                                          \
        t = __riscv_vmerge_vvm_f32m##LMUL(tt, t, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl), vl);                                                                                                                                                                                                                                                                                              \
        /* absx < 0.84375f ? x2:t*/                                                                                                                                                                                                                                                                                                                                                                        \
        t = __riscv_vmerge_vvm_f32m##LMUL(x2, t, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl);                                                                                                                                                                                                                                                                                           \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t u = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_rb6, c_erfc_rb5, vl), c_erfc_rb4, vl), c_erfc_rb3, vl), c_erfc_rb2, vl), c_erfc_rb1, vl), c_erfc_rb0, vl);                                                    \
        vfloat32m##LMUL##_t v = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_sb7, c_erfc_sb6, vl), c_erfc_sb5, vl), c_erfc_sb4, vl), c_erfc_sb3, vl), c_erfc_sb2, vl), c_erfc_sb1, vl);                                                    \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_ra7, c_erfc_ra6, vl), c_erfc_ra5, vl), c_erfc_ra4, vl), c_erfc_ra3, vl), c_erfc_ra2, vl), c_erfc_ra1, vl), c_erfc_ra0, vl); \
        vfloat32m##LMUL##_t tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_sa8, c_erfc_sa7, vl), c_erfc_sa6, vl), c_erfc_sa5, vl), c_erfc_sa4, vl), c_erfc_sa3, vl), c_erfc_sa2, vl), c_erfc_sa1, vl); \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 2.857143f, vl), vl); /* u = absx < 0x1.6db6dap+1f ? tu : u;*/                                                                                                                                                                                                                                                 \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 2.857143f, vl), vl); /* v = absx < 0x1.6db6dap+1f ? tv : v;*/                                                                                                                                                                                                                                                 \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_pa6, c_erfc_pa5, vl), c_erfc_pa4, vl), c_erfc_pa3, vl), c_erfc_pa2, vl), c_erfc_pa1, vl), c_erfc_pa0, vl);                                                                       \
        tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_qa6, c_erfc_qa5, vl), c_erfc_qa4, vl), c_erfc_qa3, vl), c_erfc_qa2, vl), c_erfc_qa1, vl);                                                                                                                         \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl),                                                                                                                                                                                                                                                                                                   \
                                          vl); /* absx < 1.25f ? tu : u */                                                                                                                                                                                                                                                                                                                                 \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl),                                                                                                                                                                                                                                                                                                   \
                                          vl); /* absx < 1.25f ? tv : v  */                                                                                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        tu = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_pp4, c_erfc_pp3, vl), c_erfc_pp2, vl), c_erfc_pp1, vl), c_erfc_pp0, vl);                                                                                                                                                                           \
        tv = __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vvf_f32m##LMUL(t, __riscv_vfmadd_vff_f32m##LMUL(t, c_erfc_qq5, c_erfc_qq4, vl), c_erfc_qq3, vl), c_erfc_qq2, vl), c_erfc_qq1, vl);                                                                                                                                                                           \
        u = __riscv_vmerge_vvm_f32m##LMUL(tu, u, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl); /* absx < 0.84375f ? tu : u */                                                                                                                                                                                                                                                            \
        v = __riscv_vmerge_vvm_f32m##LMUL(tv, v, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl); /* absx <  0.84375f ? tv : v */                                                                                                                                                                                                                                                           \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        v = __riscv_vfmadd_vvf_f32m##LMUL(t, v, 1.f, vl);                                                                                                                                                                                                                                                                                                                                                  \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t q = __riscv_vfdiv_vv_f32m##LMUL(u, v, vl);                                                                                                                                                                                                                                                                                                                                     \
        vfloat32m##LMUL##_t ret = __riscv_vfmv_v_f_f32m##LMUL(0.f, vl);                                                                                                                                                                                                                                                                                                                                    \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t z = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(__riscv_vand_vx_u32m##LMUL(__riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(absx), 0xfffff000, vl));                                                                                                                                                                                                                          \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        vfloat32m##LMUL##_t r = __riscv_vfmul_vv_f32m##LMUL(exp_ps(__riscv_vfmadd_vvf_f32m##LMUL(__riscv_vfneg_v_f32m##LMUL(z, vl), z, -0.5625f, vl), vl), exp_ps(__riscv_vfmadd_vv_f32m##LMUL(__riscv_vfsub_vv_f32m##LMUL(z, absx, vl), __riscv_vfadd_vv_f32m##LMUL(z, absx, vl), q, vl), vl), vl);                                                                                                       \
        r = __riscv_vfdiv_vv_f32m##LMUL(r, absx, vl);                                                                                                                                                                                                                                                                                                                                                      \
        t = __riscv_vfrsub_vf_f32m##LMUL(r, 2.f, vl);                                                                                                                                                                                                                                                                                                                                                      \
        r = __riscv_vmerge_vvm_f32m##LMUL(t, r, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl), vl);         /* x < 0.f ? t:r  */                                                                                                                                                                                                                                                                       \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 28.f, vl), vl); /*  abs < 28.f ? r : ret  */                                                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        r = __riscv_vfrsub_vf_f32m##LMUL(q, 1.f - c_erfc_erx_f, vl);                                                                                                                                                                                                                                                                                                                                       \
        t = __riscv_vfadd_vf_f32m##LMUL(q, 1.f + c_erfc_erx_f, vl);                                                                                                                                                                                                                                                                                                                                        \
        r = __riscv_vmerge_vvm_f32m##LMUL(t, r, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl), vl);          /*  x < 0.f ? t:r*/                                                                                                                                                                                                                                                                       \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 1.25f, vl), vl); /*  absx < 1.25f ? r : ret*/                                                                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        r = __riscv_vfrsub_vf_f32m##LMUL(__riscv_vfmadd_vv_f32m##LMUL(x, q, __riscv_vfsub_vf_f32m##LMUL(x, 0.5f, vl), vl), .5, vl);                                                                                                                                                                                                                                                                        \
        ret = __riscv_vmerge_vvm_f32m##LMUL(r, ret, __riscv_vmfge_vf_f32m##LMUL##_b##MLEN(absx, 0.84375f, vl), vl); /*  absx < 0.84375f ? r : ret*/                                                                                                                                                                                                                                                        \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        ret = __riscv_vfmerge_vfm_f32m##LMUL(ret, 2.f, __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, -6.0f, vl), vl); /*  x< -6.0f ? 2.0f: ret*/                                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        ret = __riscv_vmerge_vvm_f32m##LMUL(x, ret, __riscv_vmfeq_vv_f32m##LMUL##_b##MLEN(x, x, vl), vl); /*  erfc(NaN) = NaN*/                                                                                                                                                                                                                                                                            \
                                                                                                                                                                                                                                                                                                                                                                                                           \
        return ret;                                                                                                                                                                                                                                                                                                                                                                                        \
    }

_RVV_FLOAT32_ERFC_OP(1, 32)
_RVV_FLOAT32_ERFC_OP(2, 16)
_RVV_FLOAT32_ERFC_OP(4, 8)
_RVV_FLOAT32_ERFC_OP(8, 4)

//TODO rvv optimize
#define _RVV_FLOAT32_ATAN2_OP(LMUL, MLEN)                                                                        \
    static inline vfloat32m##LMUL##_t atan2_ps(vfloat32m##LMUL##_t a, vfloat32m##LMUL##_t b, volatile size_t vl) \
    {                                                                                                            \
        std::vector<float> tmpx(vl);                                                                             \
        std::vector<float> tmpy(vl);                                                                             \
        __riscv_vse32_v_f32m##LMUL(tmpx.data(), a, vl);                                                          \
        __riscv_vse32_v_f32m##LMUL(tmpy.data(), b, vl);                                                          \
        for (size_t i = 0; i < vl; i++)                                                                          \
        {                                                                                                        \
            tmpx[i] = atan2f(tmpx[i], tmpy[i]);                                                                  \
        }                                                                                                        \
        return __riscv_vle32_v_f32m##LMUL(tmpx.data(), vl);                                                      \
    }

_RVV_FLOAT32_ATAN2_OP(1, 32)
_RVV_FLOAT32_ATAN2_OP(2, 16)
_RVV_FLOAT32_ATAN2_OP(4, 8)
_RVV_FLOAT32_ATAN2_OP(8, 4)

#endif // RVV_MATHFUN_H
