// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LOONGARCH_ACTIVATION_H
#define LOONGARCH_ACTIVATION_H

#include "fused_activation.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"

static NCNN_FORCEINLINE __m128 sigmoid_lsx(__m128 inputs)
{
    const __m128 one = (__m128)__lsx_vreplfr2vr_s(1.0f);
    return __lsx_vfdiv_s(one, __lsx_vfadd_s(one, exp_ps(__lsx_vfsub_s((__m128)__lsx_vreplgr2vr_w(0), inputs))));
}

static NCNN_FORCEINLINE __m128 tanh_lsx(__m128 inputs)
{
    const __m128 one = (__m128)__lsx_vreplfr2vr_s(1.0f);
    const __m128 two = (__m128)__lsx_vreplfr2vr_s(2.0f);
    return __lsx_vfsub_s(__lsx_vfmul_s(sigmoid_lsx(__lsx_vfmul_s(inputs, two)), two), one);
}

static NCNN_FORCEINLINE __m128 mish_lsx(__m128 inputs)
{
    return __lsx_vfmul_s(inputs, tanh_lsx(log_ps(__lsx_vfadd_s(exp_ps(inputs), (__m128)__lsx_vreplfr2vr_s(1.f)))));
}

static NCNN_FORCEINLINE __m128 swish_lsx(__m128 inputs)
{
    return __lsx_vfmul_s(inputs, sigmoid_lsx(inputs));
}

static NCNN_FORCEINLINE __m128 hardswish_lsx(__m128 inputs, __m128 a, __m128 b)
{
    const __m128 one = (__m128)__lsx_vreplfr2vr_s(1.0f);
    b = __lsx_vfmadd_s(a, inputs, b);
    b = __lsx_vfmax_s(b, (__m128)__lsx_vreplgr2vr_w(0));
    b = __lsx_vfmin_s(b, one);
    return __lsx_vfmul_s(b, inputs);
}

static NCNN_FORCEINLINE __m128 lrelu_lsx(__m128 inputs, float slope)
{
    __m128 pos = __lsx_vfmax_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    __m128 neg = __lsx_vfmin_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    return __lsx_vfadd_s(pos, __lsx_vfmul_s((__m128)__lsx_vreplfr2vr_s(slope), neg));
}

static NCNN_FORCEINLINE __m128 prelu_lsx(__m128 inputs, __m128 alphas)
{
    __m128 pos = __lsx_vfmax_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    __m128 neg = __lsx_vfmin_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    return __lsx_vfadd_s(pos, __lsx_vfmul_s(alphas, neg));
}

static NCNN_FORCEINLINE __m128 elu_lsx(__m128 inputs, __m128 alphas)
{
    __m128 pos = __lsx_vfmax_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    __m128 neg = __lsx_vfmin_s(inputs, (__m128)__lsx_vreplgr2vr_w(0));
    neg = __lsx_vfsub_s(exp_ps(neg), (__m128)__lsx_vreplfr2vr_s(1.f));
    return __lsx_vfadd_s(pos, __lsx_vfmul_s(alphas, neg));
}

static NCNN_FORCEINLINE __m128 activation_lsx(__m128 _v, int activation_type, const ncnn::Mat& activation_params)
{
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return __lsx_vfmax_s(_v, (__m128)__lsx_vreplgr2vr_w(0));
    }
    case 2:
    {
        // Leaky relu
        return lrelu_lsx(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        __m128 min = (__m128)__lsx_vreplfr2vr_s(activation_params[0]);
        __m128 max = (__m128)__lsx_vreplfr2vr_s(activation_params[1]);
        return __lsx_vfmin_s(__lsx_vfmax_s(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_lsx(_v);
    }
    case 5:
    {
        // Mish
        return mish_lsx(_v);
    }
    case 6:
    {
        // Hard swish
        __m128 _a = (__m128)__lsx_vreplfr2vr_s(activation_params[0]);
        __m128 _b = (__m128)__lsx_vreplfr2vr_s(activation_params[1]);
        return hardswish_lsx(_v, _a, _b);
    }
    }

    return _v;
}

#endif // __loongarch_sx

#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"

static NCNN_FORCEINLINE __m256 sigmoid_lasx(__m256 inputs)
{
    const __m256 one = (__m256)__lasx_xvreplfr2vr_s(1.0f);
    __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
    return __lasx_xvfdiv_s(one, __lasx_xvfadd_s(one, exp256_ps(__lasx_xvfsub_s(_zero, inputs))));
}

static NCNN_FORCEINLINE __m256 tanh_lasx(__m256 inputs)
{
    const __m256 one = (__m256)__lasx_xvreplfr2vr_s(1.0f);
    const __m256 two = (__m256)__lasx_xvreplfr2vr_s(2.0f);
    return __lasx_xvfsub_s(__lasx_xvfmul_s(sigmoid_lasx(__lasx_xvfmul_s(inputs, two)), two), one);
}

static NCNN_FORCEINLINE __m256 mish_lasx(__m256 inputs)
{
    return __lasx_xvfmul_s(inputs, tanh_lasx(log256_ps(__lasx_xvfadd_s(exp256_ps(inputs), (__m256)__lasx_xvreplfr2vr_s(1.f)))));
}

static NCNN_FORCEINLINE __m256 swish_lasx(__m256 inputs)
{
    return __lasx_xvfmul_s(inputs, sigmoid_lasx(inputs));
}

static NCNN_FORCEINLINE __m256 hardswish_lasx(__m256 inputs, __m256 a, __m256 b)
{
    const __m256 one = (__m256)__lasx_xvreplfr2vr_s(1.0f);
    b = __lasx_xvfmadd_s(a, inputs, b);
    b = __lasx_xvfmax_s(b, (__m256)__lasx_xvreplgr2vr_w(0));
    b = __lasx_xvfmin_s(b, one);
    return __lasx_xvfmul_s(b, inputs);
}

static NCNN_FORCEINLINE __m256 lrelu_lasx(__m256 inputs, float slope)
{
    __m256 pos = __lasx_xvfmax_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    __m256 neg = __lasx_xvfmin_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    return __lasx_xvfadd_s(pos, __lasx_xvfmul_s((__m256)__lasx_xvreplfr2vr_s(slope), neg));
}

static NCNN_FORCEINLINE __m256 prelu_lasx(__m256 inputs, __m256 alphas)
{
    __m256 pos = __lasx_xvfmax_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    __m256 neg = __lasx_xvfmin_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    return __lasx_xvfadd_s(pos, __lasx_xvfmul_s(alphas, neg));
}

static NCNN_FORCEINLINE __m256 elu_lasx(__m256 inputs, __m256 alphas)
{
    __m256 pos = __lasx_xvfmax_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    __m256 neg = __lasx_xvfmin_s(inputs, (__m256)__lasx_xvreplgr2vr_w(0));
    neg = __lasx_xvfsub_s(exp256_ps(neg), (__m256)__lasx_xvreplfr2vr_s(1.f));
    return __lasx_xvfadd_s(pos, __lasx_xvfmul_s(alphas, neg));
}

static NCNN_FORCEINLINE __m256 activation_lasx(__m256 _v, int activation_type, const ncnn::Mat& activation_params)
{
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return __lasx_xvfmax_s(_v, (__m256)__lasx_xvreplgr2vr_w(0));
    }
    case 2:
    {
        // Leaky relu
        return lrelu_lasx(_v, activation_params[0]);
    }
    case 3:
    {
        // min max clip
        __m256 min = (__m256)__lasx_xvreplfr2vr_s(activation_params[0]);
        __m256 max = (__m256)__lasx_xvreplfr2vr_s(activation_params[1]);
        return __lasx_xvfmin_s(__lasx_xvfmax_s(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_lasx(_v);
    }
    case 5:
    {
        // Mish
        return mish_lasx(_v);
    }
    case 6:
    {
        // Hard swish
        __m256 _a = (__m256)__lasx_xvreplfr2vr_s(activation_params[0]);
        __m256 _b = (__m256)__lasx_xvreplfr2vr_s(activation_params[1]);
        return hardswish_lasx(_v, _a, _b);
    }
    }

    return _v;
}

#endif // __loongarch_asx

#endif // LOONGARCH_ACTIVATION_H
