// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MIPS_ACTIVATION_H
#define MIPS_ACTIVATION_H

#include "fused_activation.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"

static NCNN_FORCEINLINE v4f32 sigmoid_msa(v4f32 inputs)
{
    const v4f32 one = (v4f32)__msa_fill_w_f32(1.0f);
    return __msa_fdiv_w(one, __msa_fadd_w(one, exp_ps(__msa_fsub_w((v4f32)__msa_fill_w(0), inputs))));
}

static NCNN_FORCEINLINE v4f32 tanh_msa(v4f32 inputs)
{
    const v4f32 one = (v4f32)__msa_fill_w_f32(1.0f);
    const v4f32 two = (v4f32)__msa_fill_w_f32(2.0f);
    return __msa_fsub_w(__msa_fmul_w(sigmoid_msa(__msa_fmul_w(inputs, two)), two), one);
}

static NCNN_FORCEINLINE v4f32 mish_msa(v4f32 inputs)
{
    return __msa_fmul_w(inputs, tanh_msa(log_ps(__msa_fadd_w(exp_ps(inputs), (v4f32)__msa_fill_w_f32(1.f)))));
}

static NCNN_FORCEINLINE v4f32 swish_msa(v4f32 inputs)
{
    return __msa_fmul_w(inputs, sigmoid_msa(inputs));
}

static NCNN_FORCEINLINE v4f32 hardswish_msa(v4f32 inputs, v4f32 a, v4f32 b)
{
    const v4f32 one = (v4f32)__msa_fill_w_f32(1.0f);
    b = __msa_fmadd_w(b, inputs, a);
    b = __msa_fmax_w(b, (v4f32)__msa_fill_w(0));
    b = __msa_fmin_w(b, one);
    return __msa_fmul_w(b, inputs);
}

static NCNN_FORCEINLINE v4f32 lrelu_msa(v4f32 inputs, float slope)
{
    v4f32 pos = __msa_fmax_w((v4f32)__msa_fill_w(0), inputs);
    v4f32 neg = __msa_fmin_w((v4f32)__msa_fill_w(0), inputs);
    return __msa_fadd_w(pos, __msa_fmul_w((v4f32)__msa_fill_w_f32(slope), neg));
}

static NCNN_FORCEINLINE v4f32 prelu_msa(v4f32 inputs, v4f32 alphas)
{
    v4f32 pos = __msa_fmax_w((v4f32)__msa_fill_w(0), inputs);
    v4f32 neg = __msa_fmin_w((v4f32)__msa_fill_w(0), inputs);
    return __msa_fadd_w(pos, __msa_fmul_w(alphas, neg));
}

static NCNN_FORCEINLINE v4f32 elu_msa(v4f32 inputs, v4f32 alphas)
{
    v4f32 pos = __msa_fmax_w((v4f32)__msa_fill_w(0), inputs);
    v4f32 neg = __msa_fmin_w((v4f32)__msa_fill_w(0), inputs);
    neg = __msa_fsub_w(exp_ps(neg), (v4f32)__msa_fill_w_f32(1.f));
    return __msa_fadd_w(pos, __msa_fmul_w(alphas, neg));
}

static NCNN_FORCEINLINE v4f32 activation_msa(v4f32 _v, int activation_type, const ncnn::Mat& activation_params)
{
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return __msa_fmax_w(_v, (v4f32)__msa_fill_w(0));
    }
    case 2:
    {
        // Leaky relu
        return lrelu_msa(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        v4f32 min = (v4f32)__msa_fill_w_f32(activation_params[0]);
        v4f32 max = (v4f32)__msa_fill_w_f32(activation_params[1]);
        return __msa_fmin_w(__msa_fmax_w(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_msa(_v);
    }
    case 5:
    {
        // Mish
        return mish_msa(_v);
    }
    case 6:
    {
        // Hard swish
        v4f32 _a = (v4f32)__msa_fill_w_f32(activation_params[0]);
        v4f32 _b = (v4f32)__msa_fill_w_f32(activation_params[1]);
        return hardswish_msa(_v, _a, _b);
    }
    }

    return _v;
}

#endif // __mips_msa

#endif // MIPS_ACTIVATION_H
