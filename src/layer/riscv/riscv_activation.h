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

#ifndef RISCV_ACTIVATION_H
#define RISCV_ACTIVATION_H

static inline float activation_ss(float v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        v = std::max(v, 0.f);
    }
    else if (activation_type == 2)
    {
        float slope = activation_params[0];
        v = v > 0.f ? v : v * slope;
    }
    else if (activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (v < min)
            v = min;
        if (v > max)
            v = max;
    }
    else if (activation_type == 4)
    {
        v = 1.f / (1.f + exp(-v));
    }
    else if (activation_type == 5)
    {
        v = v * tanh(log(exp(v) + 1.f));
    }

    return v;
}

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"

#define _RVV_FLOAT_ACTIVATION_PS(SEW, LMUL, MLEN)                                                                                                            \
    static inline vfloat##SEW##m##LMUL##_t activation_ps(vfloat##SEW##m##LMUL##_t _v, int activation_type, const ncnn::Mat& activation_params, word_type vl) \
    {                                                                                                                                                        \
        if (activation_type == 1)                                                                                                                            \
        {                                                                                                                                                    \
            _v = vfmax_vf_f##SEW##m##LMUL(_v, 0.f, vl);                                                                                                      \
        }                                                                                                                                                    \
        else if (activation_type == 2)                                                                                                                       \
        {                                                                                                                                                    \
            vbool##MLEN##_t _lemask = vmfle_vf_f##SEW##m##LMUL##_b##MLEN(_v, 0.f, vl);                                                                       \
            _v = vfmul_vf_f##SEW##m##LMUL##_m(_lemask, _v, _v, activation_params[0], vl);                                                                    \
        }                                                                                                                                                    \
        else if (activation_type == 3)                                                                                                                       \
        {                                                                                                                                                    \
            _v = vfmax_vf_f##SEW##m##LMUL(_v, activation_params[0], vl);                                                                                     \
            _v = vfmin_vf_f##SEW##m##LMUL(_v, activation_params[1], vl);                                                                                     \
        }                                                                                                                                                    \
        else if (activation_type == 4)                                                                                                                       \
        {                                                                                                                                                    \
            _v = sigmoid_ps(_v, vl);                                                                                                                         \
        }                                                                                                                                                    \
        else if (activation_type == 5)                                                                                                                       \
        {                                                                                                                                                    \
            _v = vfmul_vv_f##SEW##m##LMUL(_v, tanh_ps(log_ps(vfadd_vf_f##SEW##m##LMUL(exp_ps(_v, vl), 1.f, vl), vl), vl), vl);                               \
        }                                                                                                                                                    \
                                                                                                                                                             \
        return _v;                                                                                                                                           \
    }

_RVV_FLOAT_ACTIVATION_PS(16, 1, 16)
_RVV_FLOAT_ACTIVATION_PS(16, 2, 8)
_RVV_FLOAT_ACTIVATION_PS(16, 4, 4)
_RVV_FLOAT_ACTIVATION_PS(16, 8, 2)
_RVV_FLOAT_ACTIVATION_PS(32, 1, 32)
_RVV_FLOAT_ACTIVATION_PS(32, 2, 16)
_RVV_FLOAT_ACTIVATION_PS(32, 4, 8)
_RVV_FLOAT_ACTIVATION_PS(32, 8, 4)

#endif // __riscv_vector

#endif // RISCV_ACTIVATION_H
