// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LOONGARCH_ACTIVATION_H
#define LOONGARCH_ACTIVATION_H

#include "fused_activation.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"

static inline __m128 activation_ps(__m128 _v, int activation_type, const ncnn::Mat& activation_params)
{
    if (activation_type == 1)
    {
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        _v = __lsx_vfmax_s(_v, _zero);
    }
    else if (activation_type == 2)
    {
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _slope = (__m128)__lsx_vreplfr2vr_s(activation_params[0]);
        __m128i _lemask = __lsx_vfcmp_cle_s(_v, _zero);
        __m128 _ps = __lsx_vfmul_s(_v, _slope);
        _v = (__m128)__lsx_vbitsel_v((__m128i)_v, (__m128i)_ps, (__m128i)_lemask);
    }
    else if (activation_type == 3)
    {
        __m128 _min = (__m128)__lsx_vreplfr2vr_s(activation_params[0]);
        __m128 _max = (__m128)__lsx_vreplfr2vr_s(activation_params[1]);
        _v = __lsx_vfmax_s(_v, _min);
        _v = __lsx_vfmin_s(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps(_v);
    }
    else if (activation_type == 5)
    {
        _v = __lsx_vfmul_s(_v, tanh_ps(log_ps(__lsx_vfadd_s(exp_ps(_v), (__m128)__lsx_vreplfr2vr_s(1.f)))));
    }
    else if (activation_type == 6)
    {
        __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(activation_params[0]);
        __m128 _beta = (__m128)__lsx_vreplfr2vr_s(activation_params[1]);
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _outp = __lsx_vfmadd_s(_alpha, _v, _beta);
        _outp = __lsx_vfmax_s(_outp, _zero);
        _outp = __lsx_vfmin_s(_outp, _one);
        _v = __lsx_vfmul_s(_outp, _v);
    }

    return _v;
}
#endif // __loongarch_sx

#endif // LOONGARCH_ACTIVATION_H
