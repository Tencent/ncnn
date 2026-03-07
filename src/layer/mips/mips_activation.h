// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MIPS_ACTIVATION_H
#define MIPS_ACTIVATION_H

#include "fused_activation.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"

static inline v4f32 swish_ps(v4f32 _v)
{
    return __msa_fmul_w(_v, sigmoid_ps(_v));
}

static inline v4f32 elu_ps(v4f32 _v, v4f32 _alpha)
{
    v4f32 _zero = (v4f32)__msa_fill_w(0);
    v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
    v4f32 _pos = __msa_fmax_w(_v, _zero);
    v4f32 _neg = __msa_fmin_w(_v, _zero);
    _neg = __msa_fsub_w(exp_ps(_neg), _one);
    return __msa_fadd_w(_pos, __msa_fmul_w(_alpha, _neg));
}

static inline v4f32 gelu_ps(v4f32 _v, int fast_gelu)
{
    if (fast_gelu)
    {
        v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v4f32 _fast1c = (v4f32)__msa_fill_w_f32(0.79788452f);
        v4f32 _fast2c = (v4f32)__msa_fill_w_f32(0.044715f);
        v4f32 _cube = __msa_fmul_w(_v, _v);
        _cube = __msa_fmul_w(_v, _cube);
        v4f32 _blob = __msa_fmul_w(_fast2c, _cube);
        _blob = __msa_fadd_w(_v, _blob);
        _blob = __msa_fmul_w(_fast1c, _blob);
        _blob = tanh_ps(_blob);
        _blob = __msa_fadd_w(_one, _blob);
        return __msa_fmul_w(_half, __msa_fmul_w(_blob, _v));
    }
    else
    {
        v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v4f32 _inv_sqrt2 = (v4f32)__msa_fill_w_f32(0.70710678f);
        v4f32 _erf = erf_ps(__msa_fmul_w(_v, _inv_sqrt2));
        v4f32 _blob = __msa_fadd_w(_one, _erf);
        return __msa_fmul_w(_half, __msa_fmul_w(_blob, _v));
    }
}

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
    else if (activation_type == 7)
    {
        int fast_gelu = activation_params.row<int>(0)[0];
        _v = gelu_ps(_v, fast_gelu);
    }
    else if (activation_type == 8)
    {
        _v = swish_ps(_v);
    }
    else if (activation_type == 9)
    {
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(activation_params[0]);
        _v = elu_ps(_v, _alpha);
    }
    else if (activation_type == 10)
    {
        v4f32 _alpha = (v4f32)__msa_fill_w_f32(1.67326324f);
        v4f32 _lambda = (v4f32)__msa_fill_w_f32(1.050700987f);
        _v = __msa_fmul_w(_lambda, elu_ps(_v, _alpha));
    }

    return _v;
}
#endif // __mips_msa

#endif // MIPS_ACTIVATION_H
