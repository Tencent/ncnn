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

#ifndef LOONGARCH_USABILITY_H
#define LOONGARCH_USABILITY_H

#if __loongarch_sx
#include <lsxintrin.h>
#endif // __loongarch_sx

#include <stdint.h>

namespace ncnn {

typedef union
{
    int32_t i;
    float f;
} FloatInt;

} // namespace ncnn

#if __loongarch_sx
/* declare some loongarch constants with union */
#define _LOONGARCH_FLOAT_CONST(Name, Val) \
    static const ncnn::FloatInt Name = {.f = Val}

/* float type data load instructions */
static NCNN_FORCEINLINE __m128 __lsx_vreplfr2vr_s(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (__m128)__lsx_vreplgr2vr_w(fi_tmpval.i);
}

static NCNN_FORCEINLINE float __lsx_reduce_fadd_s(__m128 _v)
{
    // TODO find a more efficient way
    float* _v_p = (float*)&_v;
    return _v_p[0] + _v_p[1] + _v_p[2] + _v_p[3];
}

static NCNN_FORCEINLINE int __lsx_reduce_add_w(__m128i _v)
{
    // TODO find a more efficient way
    int* _v_p = (int*)&_v;
    return _v_p[0] + _v_p[1] + _v_p[2] + _v_p[3];
}

#endif // __loongarch_sx

static NCNN_FORCEINLINE signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __loongarch_sx
static NCNN_FORCEINLINE __m128i float2int8(__m128 _v)
{
    // simulate round to nearest via +/-0.5
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _sign = __lsx_vand_v((__m128i)_v, _signmask);
    __m128 _p5s = (__m128)__lsx_vor_v((__m128i)_p5, (__m128i)_sign);
    __m128 _v5 = __lsx_vfadd_s(_v, _p5s);
    __m128i _v32 = __lsx_vftintrz_w_s(_v5);

    __m128i _v32_16 = __lsx_vsat_w(_v32, 15);
    __m128i _v16 = __lsx_vpickev_h(_v32_16, _v32_16);
    _v16 = __lsx_vmax_h(_v16, __lsx_vreplgr2vr_h(-127));
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8(__m128 _vlow, __m128 _vhigh)
{
    // simulate round to nearest via +/-0.5
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _signlow = __lsx_vand_v((__m128i)_vlow, _signmask);
    __m128i _signhigh = __lsx_vand_v((__m128i)_vhigh, _signmask);
    __m128 _p5low = (__m128)__lsx_vor_v((__m128i)_p5, _signlow);
    __m128 _p5high = (__m128)__lsx_vor_v((__m128i)_p5, _signhigh);
    __m128 _vlow5 = __lsx_vfadd_s(_vlow, _p5low);
    __m128 _vhigh5 = __lsx_vfadd_s(_vhigh, _p5high);
    __m128i _vlow32 = __lsx_vftintrz_w_s(_vlow5);
    __m128i _vhigh32 = __lsx_vftintrz_w_s(_vhigh5);

    __m128i _vlow32_16 = __lsx_vsat_w(_vlow32, 15);
    __m128i _vhigh32_16 = __lsx_vsat_w(_vhigh32, 15);
    __m128i _v16 = __lsx_vpickev_h(_vhigh32_16, _vlow32_16);
    _v16 = __lsx_vmax_h(_v16, __lsx_vreplgr2vr_h(-127));
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE __m128i float2int8relu(__m128 _v)
{
    // simulate round to nearest via +/-0.5
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _sign = __lsx_vand_v((__m128i)_v, _signmask);
    __m128 _p5s = (__m128)__lsx_vor_v((__m128i)_p5, _sign);
    __m128 _v5 = __lsx_vfadd_s(_v, _p5s);
    __m128i _v32 = __lsx_vftintrz_w_s(_v5);

    __m128i _v32_16 = __lsx_vsat_w(_v32, 15);
    __m128i _v16 = __lsx_vpickev_h(_v32_16, _v32_16);
    _v16 = __lsx_vmaxi_h(_v16, 0);
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8relu(__m128 _vlow, __m128 _vhigh)
{
    // simulate round to nearest via +/-0.5
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _signlow = __lsx_vand_v((__m128i)_vlow, _signmask);
    __m128i _signhigh = __lsx_vand_v((__m128i)_vhigh, _signmask);
    __m128 _p5low = (__m128)__lsx_vor_v((__m128i)_p5, _signlow);
    __m128 _p5high = (__m128)__lsx_vor_v((__m128i)_p5, _signhigh);
    __m128 _vlow5 = __lsx_vfadd_s(_vlow, _p5low);
    __m128 _vhigh5 = __lsx_vfadd_s(_vhigh, _p5high);
    __m128i _vlow32 = __lsx_vftintrz_w_s(_vlow5);
    __m128i _vhigh32 = __lsx_vftintrz_w_s(_vhigh5);

    __m128i _vlow32_16 = __lsx_vsat_w(_vlow32, 15);
    __m128i _vhigh32_16 = __lsx_vsat_w(_vhigh32, 15);
    __m128i _v16 = __lsx_vpickev_h(_vhigh32_16, _vlow32_16);
    _v16 = __lsx_vmaxi_h(_v16, 0);
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE __m128i float2int8leakyrelu(__m128 _v, __m128 _slope)
{
    __m128 _v_leaky = __lsx_vfmul_s(_v, _slope);

    // simulate round to nearest via +/-0.5
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _sign = __lsx_vand_v((__m128i)_v, _signmask);
    __m128 _p5s = (__m128)__lsx_vor_v((__m128i)_p5, _sign);
    __m128 _v5 = __lsx_vfadd_s(_v, _p5s);
    __m128i _v32 = __lsx_vftintrz_w_s(_v5);

    __m128i _sign_leaky = __lsx_vand_v((__m128i)_v_leaky, _signmask);
    __m128 _p5_leaky = (__m128)__lsx_vor_v((__m128i)_p5, _sign_leaky);
    __m128 _v5_leaky = __lsx_vfadd_s(_v_leaky, _p5_leaky);
    __m128i _v32_leaky = __lsx_vftintrz_w_s(_v5_leaky);

    __m128i _v32_16 = __lsx_vsat_w(_v32, 15);
    __m128i _v16 = __lsx_vpickev_h(_v32_16, _v32_16);

    __m128i _v32_16_leaky = __lsx_vsat_w(_v32_leaky, 15);
    __m128i _v16_leaky = __lsx_vpickev_h(_v32_16_leaky, _v32_16_leaky);

    _v16 = __lsx_vmax_h(_v16, _v16_leaky);
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8leakyrelu(__m128 _vlow, __m128 _vhigh, __m128 _slope)
{
    __m128 _vlow_leaky = __lsx_vfmul_s(_vlow, _slope);
    __m128 _vhigh_leaky = __lsx_vfmul_s(_vhigh, _slope);

    // simulate round to nearest via +/-0.5
    __m128i _p5 = (__m128i)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _signlow = __lsx_vand_v((__m128i)_vlow, _signmask);
    __m128i _signhigh = __lsx_vand_v((__m128i)_vhigh, _signmask);
    __m128 _p5low = (__m128)__lsx_vor_v(_p5, _signlow);
    __m128 _p5high = (__m128)__lsx_vor_v(_p5, _signhigh);
    __m128 _vlow5 = __lsx_vfadd_s(_vlow, _p5low);
    __m128 _vhigh5 = __lsx_vfadd_s(_vhigh, _p5high);
    __m128i _vlow32 = __lsx_vftintrz_w_s(_vlow5);
    __m128i _vhigh32 = __lsx_vftintrz_w_s(_vhigh5);

    __m128i _signlow_leaky = __lsx_vand_v((__m128i)_vlow_leaky, _signmask);
    __m128i _signhigh_leaky = __lsx_vand_v((__m128i)_vhigh_leaky, _signmask);
    __m128 _p5low_leaky = (__m128)__lsx_vor_v(_p5, _signlow_leaky);
    __m128 _p5high_leaky = (__m128)__lsx_vor_v(_p5, _signhigh_leaky);
    __m128 _vlow5_leaky = __lsx_vfadd_s(_vlow_leaky, _p5low_leaky);
    __m128 _vhigh5_leaky = __lsx_vfadd_s(_vhigh_leaky, _p5high_leaky);
    __m128i _vlow32_leaky = __lsx_vftintrz_w_s(_vlow5_leaky);
    __m128i _vhigh32_leaky = __lsx_vftintrz_w_s(_vhigh5_leaky);

    __m128i _vlow32_16 = __lsx_vsat_w(_vlow32, 15);
    __m128i _vhigh32_16 = __lsx_vsat_w(_vhigh32, 15);
    __m128i _v16 = __lsx_vpickev_h(_vhigh32_16, _vlow32_16);

    __m128i _vlow32_16_leaky = __lsx_vsat_w(_vlow32_leaky, 15);
    __m128i _vhigh32_16_leaky = __lsx_vsat_w(_vhigh32_leaky, 15);
    __m128i _v16_leaky = __lsx_vpickev_h(_vhigh32_16_leaky, _vlow32_16_leaky);

    _v16 = __lsx_vmax_h(_v16, _v16_leaky);
    __m128i _v16_8 = __lsx_vsat_h(_v16, 7);
    __m128i _v8 = __lsx_vpickev_b(_v16_8, _v16_8);

    return _v8[0];
}
#endif // __loongarch_sx

#endif // LOONGARCH_USABILITY_H
