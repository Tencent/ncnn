// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef RISCV_ACTIVATION_H
#define RISCV_ACTIVATION_H

#include "fused_activation.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#if __riscv_zvfh
#include "rvv_mathfun_fp16s.h"
#endif

#define _RVV_FLOAT_ACTIVATION_PS(SEW, LMUL, MLEN, STYPE)                                                                                                                      \
    static inline vfloat##SEW##m##LMUL##_t activation_ps(vfloat##SEW##m##LMUL##_t _v, int activation_type, const ncnn::Mat& activation_params, size_t vl)                     \
    {                                                                                                                                                                         \
        if (activation_type == 1)                                                                                                                                             \
        {                                                                                                                                                                     \
            _v = __riscv_vfmax_vf_f##SEW##m##LMUL(_v, (STYPE)0.f, vl);                                                                                                        \
        }                                                                                                                                                                     \
        else if (activation_type == 2)                                                                                                                                        \
        {                                                                                                                                                                     \
            vbool##MLEN##_t _lemask = __riscv_vmfle_vf_f##SEW##m##LMUL##_b##MLEN(_v, (STYPE)0.f, vl);                                                                         \
            _v = __riscv_vfmul_vf_f##SEW##m##LMUL##_mu(_lemask, _v, _v, (STYPE)activation_params[0], vl);                                                                     \
        }                                                                                                                                                                     \
        else if (activation_type == 3)                                                                                                                                        \
        {                                                                                                                                                                     \
            _v = __riscv_vfmax_vf_f##SEW##m##LMUL(_v, (STYPE)activation_params[0], vl);                                                                                       \
            _v = __riscv_vfmin_vf_f##SEW##m##LMUL(_v, (STYPE)activation_params[1], vl);                                                                                       \
        }                                                                                                                                                                     \
        else if (activation_type == 4)                                                                                                                                        \
        {                                                                                                                                                                     \
            _v = sigmoid_ps(_v, vl);                                                                                                                                          \
        }                                                                                                                                                                     \
        else if (activation_type == 5)                                                                                                                                        \
        {                                                                                                                                                                     \
            _v = __riscv_vfmul_vv_f##SEW##m##LMUL(_v, tanh_ps(log_ps(__riscv_vfadd_vf_f##SEW##m##LMUL(exp_ps(_v, vl), (STYPE)1.f, vl), vl), vl), vl);                         \
        }                                                                                                                                                                     \
        else if (activation_type == 6)                                                                                                                                        \
        {                                                                                                                                                                     \
            const float alpha = activation_params[0];                                                                                                                         \
            const float beta = activation_params[1];                                                                                                                          \
            const float lower = -beta / alpha;                                                                                                                                \
            const float upper = (1.f / alpha) + lower;                                                                                                                        \
            vbool##MLEN##_t _lower = __riscv_vmflt_vf_f##SEW##m##LMUL##_b##MLEN(_v, (STYPE)lower, vl);                                                                        \
            vbool##MLEN##_t _higher = __riscv_vmfgt_vf_f##SEW##m##LMUL##_b##MLEN(_v, (STYPE)upper, vl);                                                                       \
            vbool##MLEN##_t _apply = __riscv_vmnor_mm_b##MLEN(_lower, _higher, vl);                                                                                           \
            _v = __riscv_vfmerge_vfm_f##SEW##m##LMUL(_v, (STYPE).0f, _lower, vl);                                                                                             \
                                                                                                                                                                              \
            vfloat##SEW##m##LMUL##_t _p0 = __riscv_vfadd_vf_f##SEW##m##LMUL##_m(_apply, __riscv_vfmul_vf_f##SEW##m##LMUL##_m(_apply, _v, (STYPE)alpha, vl), (STYPE)beta, vl); \
            _v = __riscv_vfmul_vv_f##SEW##m##LMUL##_mu(_apply, _v, _v, _p0, vl);                                                                                              \
        }                                                                                                                                                                     \
        else if (activation_type == 7)                                                                                                                                        \
        {                                                                                                                                                                     \
            int fast_gelu = activation_params.row<int>(0)[0];                                                                                                                 \
            if (fast_gelu)                                                                                                                                                    \
            {                                                                                                                                                                 \
                vfloat##SEW##m##LMUL##_t _arg = __riscv_vfmul_vf_f##SEW##m##LMUL(                                                                                             \
                                                    __riscv_vfmul_vv_f##SEW##m##LMUL(__riscv_vfmul_vv_f##SEW##m##LMUL(_v, _v, vl), _v, vl), (STYPE)0.044715f, vl);            \
                _arg = __riscv_vfadd_vv_f##SEW##m##LMUL(_v, _arg, vl);                                                                                                        \
                _arg = __riscv_vfmul_vf_f##SEW##m##LMUL(_arg, (STYPE)0.79788452f, vl);                                                                                        \
                vfloat##SEW##m##LMUL##_t _tanharg = tanh_ps(_arg, vl);                                                                                                        \
                _v = __riscv_vfmul_vf_f##SEW##m##LMUL(                                                                                                                        \
                         __riscv_vfmul_vv_f##SEW##m##LMUL(_v, __riscv_vfadd_vf_f##SEW##m##LMUL(_tanharg, (STYPE)1.f, vl), vl), (STYPE).5f, vl);                               \
            }                                                                                                                                                                 \
            else                                                                                                                                                              \
            {                                                                                                                                                                 \
                _v = __riscv_vfmul_vf_f##SEW##m##LMUL(_v, (STYPE)0.f, vl);                                                                                                    \
            }                                                                                                                                                                 \
        }                                                                                                                                                                     \
        else if (activation_type == 8)                                                                                                                                        \
        {                                                                                                                                                                     \
            vfloat##SEW##m##LMUL##_t _sigmoid = sigmoid_ps(_v, vl);                                                                                                           \
            _v = __riscv_vfmul_vv_f##SEW##m##LMUL(_v, _sigmoid, vl);                                                                                                          \
        }                                                                                                                                                                     \
        else if (activation_type == 9)                                                                                                                                        \
        {                                                                                                                                                                     \
            const float alpha = activation_params[0];                                                                                                                         \
            vbool##MLEN##_t _lower = __riscv_vmflt_vf_f##SEW##m##LMUL##_b##MLEN(_v, (STYPE)0.f, vl);                                                                          \
            vfloat##SEW##m##LMUL##_t _exp_v = exp_ps(_v, vl);                                                                                                                 \
            _exp_v = __riscv_vfsub_vf_f##SEW##m##LMUL(_exp_v, (STYPE)1.f, vl);                                                                                                \
            _exp_v = __riscv_vfmul_vf_f##SEW##m##LMUL(_exp_v, (STYPE)alpha, vl);                                                                                              \
            _v = __riscv_vmerge_vvm_f##SEW##m##LMUL(_exp_v, _v, _lower, vl);                                                                                                  \
        }                                                                                                                                                                     \
        else if (activation_type == 10)                                                                                                                                       \
        {                                                                                                                                                                     \
            const float alpha = 1.67326324f;                                                                                                                                  \
            const float lambda = 1.050700987f;                                                                                                                                \
            const float alphaxlambda = alpha * lambda;                                                                                                                        \
            vbool##MLEN##_t _lower = __riscv_vmflt_vf_f##SEW##m##LMUL##_b##MLEN(_v, (STYPE)0.f, vl);                                                                          \
            vbool##MLEN##_t _higher = __riscv_vmnot_m_b##MLEN(_lower, vl);                                                                                                    \
            _v = __riscv_vfmul_vf_f##SEW##m##LMUL##_mu(_higher, _v, _v, (STYPE)lambda, vl);                                                                                   \
            vfloat##SEW##m##LMUL##_t _nps = exp_ps(_v, vl);                                                                                                                   \
            _nps = __riscv_vfsub_vf_f##SEW##m##LMUL##_mu(_lower, _nps, _nps, (STYPE)1.f, vl);                                                                                 \
            _nps = __riscv_vfmul_vf_f##SEW##m##LMUL##_mu(_lower, _v, _nps, (STYPE)alphaxlambda, vl);                                                                          \
            _v = __riscv_vmerge_vvm_f##SEW##m##LMUL(_nps, _v, _lower, vl);                                                                                                    \
        }                                                                                                                                                                     \
                                                                                                                                                                              \
        return _v;                                                                                                                                                            \
    }

#if __riscv_zvfh
_RVV_FLOAT_ACTIVATION_PS(16, 1, 16, __fp16)
_RVV_FLOAT_ACTIVATION_PS(16, 2, 8, __fp16)
_RVV_FLOAT_ACTIVATION_PS(16, 4, 4, __fp16)
_RVV_FLOAT_ACTIVATION_PS(16, 8, 2, __fp16)
#endif
_RVV_FLOAT_ACTIVATION_PS(32, 1, 32, float)
_RVV_FLOAT_ACTIVATION_PS(32, 2, 16, float)
_RVV_FLOAT_ACTIVATION_PS(32, 4, 8, float)
_RVV_FLOAT_ACTIVATION_PS(32, 8, 4, float)

#endif // __riscv_vector

#endif // RISCV_ACTIVATION_H
