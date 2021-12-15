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

#ifndef MIPS_ACTIVATION_H
#define MIPS_ACTIVATION_H

#include "fused_activation.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"

static inline v4f32 activation_ps(v4f32 _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        _v = __msa_fmax_w(_v, _zero);
    }
    else if (activation_type == 2)
    {
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _slope = (v4f32)__msa_fill_w_f32(activation_params[0]);
        v4i32_w _lemask = __msa_fcle_w(_v, _zero);
        v4f32 _ps = __msa_fmul_w(_v, _slope);
        _v = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_v, (v16u8)_ps);
    }
    else if (activation_type == 3)
    {
        v4f32 _min = (v4f32)__msa_fill_w_f32(activation_params[0]);
        v4f32 _max = (v4f32)__msa_fill_w_f32(activation_params[1]);
        _v = __msa_fmax_w(_v, _min);
        _v = __msa_fmin_w(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps(_v);
    }
    else if (activation_type == 5)
    {
        _v = __msa_fmul_w(_v, tanh_ps(log_ps(__msa_fadd_w(exp_ps(_v), (v4f32)__msa_fill_w_f32(1.f)))));
    }
    else if (activation_type == 6)
    {
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(activation_params[0]);
        v4f32 _beta = (v4f32)__msa_fill_w_f32(activation_params[1]);
        v4f32 _zero = (v4f32)__msa_fill_w(0);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v4f32 _outp = __msa_fmadd_w(_beta, _v, _alpha);
        _outp = __msa_fmax_w(_outp, _zero);
        _outp = __msa_fmin_w(_outp, _one);
        _v = __msa_fmul_w(_outp, _v);
    }

    return _v;
}
#endif // __mips_msa

#endif // MIPS_ACTIVATION_H
