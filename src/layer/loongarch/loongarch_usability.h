// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LOONGARCH_USABILITY_H
#define LOONGARCH_USABILITY_H

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
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
#endif

#if __loongarch_asx
/* declare some loongarch constants with union */
#define _LOONGARCH_FLOAT_CONST_PS256(Name, Val) \
    static const ncnn::FloatInt _ps256_##Name = {.f = Val}
#endif

#if __loongarch_sx
/* float type data load instructions */
static NCNN_FORCEINLINE __m128 __lsx_vreplfr2vr_s(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (__m128)__lsx_vreplgr2vr_w(fi_tmpval.i);
}

static NCNN_FORCEINLINE float __lsx_reduce_fadd_s(__m128 _v)
{
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)_v, (__m128i)_v);
    __m128 sum64 = (__m128)__lsx_vfadd_s((__m128i)hi64, (__m128i)_v);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)sum64);
    __m128 sum32 = (__m128)__lsx_vfadd_s((__m128i)sum64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)sum32, 0, 0);
}

static NCNN_FORCEINLINE int __lsx_reduce_add_w(__m128i _v)
{
    __m128i hi64 = __lsx_vilvl_d(_v, _v);
    __m128i sum64 = __lsx_vadd_w(hi64, _v);
    __m128i hi32 = __lsx_vilvr_w(hi64, sum64);
    __m128i sum32 = __lsx_vadd_w(sum64, hi32);
    return __lsx_vpickve_w_f(sum32, 0, 0);
}

#endif // __loongarch_sx

#if __loongarch_asx
/* float type data load instructions */
static NCNN_FORCEINLINE __m256 __lasx_xvreplfr2vr_s(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (__m256)__lasx_xvreplgr2vr_w(fi_tmpval.i);
}

static NCNN_FORCEINLINE float __lasx_reduce_fadd_s(__m256 _v)
{
    __m128 lo = __lasx_xvpickve_w_f((__m256i)_v, 0, 0);
    __m128 hi = __lasx_xvpickve_w_f((__m256i)_v, 4, 0);
    __m128 sum = (__m128)__lsx_vfadd_s((__m128i)lo, (__m128i)hi);
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)sum, (__m128i)sum);
    __m128 sum64 = (__m128)__lsx_vfadd_s((__m128i)hi64, (__m128i)sum);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)sum64);
    __m128 sum32 = (__m128)__lsx_vfadd_s((__m128i)sum64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)sum32, 0, 0);
}

static NCNN_FORCEINLINE int __lasx_reduce_add_w(__m256i _v)
{
    __m128i lo = __lasx_xvpickve_w_f(_v, 0, 0);
    __m128i hi = __lasx_xvpickve_w_f(_v, 4, 0);
    __m128i sum = __lsx_vadd_w(lo, hi);
    __m128i hi64 = __lsx_vilvl_d(sum, sum);
    __m128i sum64 = __lsx_vadd_w(hi64, sum);
    __m128i hi32 = __lsx_vilvr_w(hi64, sum64);
    __m128i sum32 = __lsx_vadd_w(sum64, hi32);

    return __lsx_vpickve_w_f(sum32, 0, 0);
}
#endif // __loongarch_asx

static NCNN_FORCEINLINE signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __loongarch_sx
static NCNN_FORCEINLINE __m128i round(__m128 _v)
{
    __m128 _p5 = (__m128)__lsx_vreplfr2vr_s(0.5f);
    __m128i _signmask = __lsx_vreplgr2vr_w(1 << 31);

    __m128i _sign = __lsx_vand_v((__m128i)_v, _signmask);
    __m128 _p5s = (__m128)__lsx_vor_v((__m128i)_p5, (__m128i)_sign);
    __m128 _v5 = __lsx_vfadd_s(_v, _p5s);
    __m128i _v32 = __lsx_vftintrz_w_s(_v5);

    return _v32;
}

static NCNN_FORCEINLINE __m128i float2int8(__m128 _v)
{
    __m128i _v32 = round(_v);

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
    __m128i _v32 = round(_v);

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

#if __loongarch_asx
static NCNN_FORCEINLINE __m256i round(__m256 _v)
{
    __m256 _p5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
    __m256i _signmask = __lasx_xvreplgr2vr_w(1 << 31);

    __m256i _sign = __lasx_xvand_v((__m256i)_v, _signmask);
    __m256 _p5s = (__m256)__lasx_xvor_v((__m256i)_p5, (__m256i)_sign);
    __m256 _v5 = __lasx_xvfadd_s(_v, _p5s);
    __m256i _v32 = __lasx_xvftintrz_w_s(_v5);

    return _v32;
}

static NCNN_FORCEINLINE __m256i float2int8(__m256 _v)
{
    __m256i _v32 = round(_v);

    __m256i _v32_16 = __lasx_xvsat_w(_v32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_v32_16, _v32_16);
    _v16 = __lasx_xvmax_h(_v16, __lasx_xvreplgr2vr_h(-127));
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8(__m256 _vlow, __m256 _vhigh)
{
    // simulate round to nearest via +/-0.5
    __m256 _p5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
    __m256i _signmask = __lasx_xvreplgr2vr_w(1 << 31);

    __m256i _signlow = __lasx_xvand_v((__m256i)_vlow, _signmask);
    __m256i _signhigh = __lasx_xvand_v((__m256i)_vhigh, _signmask);
    __m256 _p5low = (__m256)__lasx_xvor_v((__m256i)_p5, _signlow);
    __m256 _p5high = (__m256)__lasx_xvor_v((__m256i)_p5, _signhigh);
    __m256 _vlow5 = __lasx_xvfadd_s(_vlow, _p5low);
    __m256 _vhigh5 = __lasx_xvfadd_s(_vhigh, _p5high);
    __m256i _vlow32 = __lasx_xvftintrz_w_s(_vlow5);
    __m256i _vhigh32 = __lasx_xvftintrz_w_s(_vhigh5);

    __m256i _vlow32_16 = __lasx_xvsat_w(_vlow32, 15);
    __m256i _vhigh32_16 = __lasx_xvsat_w(_vhigh32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_vhigh32_16, _vlow32_16);
    _v16 = __lasx_xvmax_h(_v16, __lasx_xvreplgr2vr_h(-127));
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE __m256i float2int8relu(__m256 _v)
{
    __m256i _v32 = round(_v);

    __m256i _v32_16 = __lasx_xvsat_w(_v32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_v32_16, _v32_16);
    _v16 = __lasx_xvmaxi_h(_v16, 0);
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8relu(__m256 _vlow, __m256 _vhigh)
{
    // simulate round to nearest via +/-0.5
    __m256 _p5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
    __m256i _signmask = __lasx_xvreplgr2vr_w(1 << 31);

    __m256i _signlow = __lasx_xvand_v((__m256i)_vlow, _signmask);
    __m256i _signhigh = __lasx_xvand_v((__m256i)_vhigh, _signmask);
    __m256 _p5low = (__m256)__lasx_xvor_v((__m256i)_p5, _signlow);
    __m256 _p5high = (__m256)__lasx_xvor_v((__m256i)_p5, _signhigh);
    __m256 _vlow5 = __lasx_xvfadd_s(_vlow, _p5low);
    __m256 _vhigh5 = __lasx_xvfadd_s(_vhigh, _p5high);
    __m256i _vlow32 = __lasx_xvftintrz_w_s(_vlow5);
    __m256i _vhigh32 = __lasx_xvftintrz_w_s(_vhigh5);

    __m256i _vlow32_16 = __lasx_xvsat_w(_vlow32, 15);
    __m256i _vhigh32_16 = __lasx_xvsat_w(_vhigh32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_vhigh32_16, _vlow32_16);
    _v16 = __lasx_xvmaxi_h(_v16, 0);
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE __m256i float2int8leakyrelu(__m256 _v, __m256 _slope)
{
    __m256 _v_leaky = __lasx_xvfmul_s(_v, _slope);

    // simulate round to nearest via +/-0.5
    __m256 _p5 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
    __m256i _signmask = __lasx_xvreplgr2vr_w(1 << 31);

    __m256i _sign = __lasx_xvand_v((__m256i)_v, _signmask);
    __m256 _p5s = (__m256)__lasx_xvor_v((__m256i)_p5, _sign);
    __m256 _v5 = __lasx_xvfadd_s(_v, _p5s);
    __m256i _v32 = __lasx_xvftintrz_w_s(_v5);

    __m256i _sign_leaky = __lasx_xvand_v((__m256i)_v_leaky, _signmask);
    __m256 _p5_leaky = (__m256)__lasx_xvor_v((__m256i)_p5, _sign_leaky);
    __m256 _v5_leaky = __lasx_xvfadd_s(_v_leaky, _p5_leaky);
    __m256i _v32_leaky = __lasx_xvftintrz_w_s(_v5_leaky);

    __m256i _v32_16 = __lasx_xvsat_w(_v32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_v32_16, _v32_16);

    __m256i _v32_16_leaky = __lasx_xvsat_w(_v32_leaky, 15);
    __m256i _v16_leaky = __lasx_xvpickev_h(_v32_16_leaky, _v32_16_leaky);

    _v16 = __lasx_xvmax_h(_v16, _v16_leaky);
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8leakyrelu(__m256 _vlow, __m256 _vhigh, __m256 _slope)
{
    __m256 _vlow_leaky = __lasx_xvfmul_s(_vlow, _slope);
    __m256 _vhigh_leaky = __lasx_xvfmul_s(_vhigh, _slope);

    // simulate round to nearest via +/-0.5
    __m256i _p5 = (__m256i)__lasx_xvreplfr2vr_s(0.5f);
    __m256i _signmask = __lasx_xvreplgr2vr_w(1 << 31);

    __m256i _signlow = __lasx_xvand_v((__m256i)_vlow, _signmask);
    __m256i _signhigh = __lasx_xvand_v((__m256i)_vhigh, _signmask);
    __m256 _p5low = (__m256)__lasx_xvor_v(_p5, _signlow);
    __m256 _p5high = (__m256)__lasx_xvor_v(_p5, _signhigh);
    __m256 _vlow5 = __lasx_xvfadd_s(_vlow, _p5low);
    __m256 _vhigh5 = __lasx_xvfadd_s(_vhigh, _p5high);
    __m256i _vlow32 = __lasx_xvftintrz_w_s(_vlow5);
    __m256i _vhigh32 = __lasx_xvftintrz_w_s(_vhigh5);

    __m256i _signlow_leaky = __lasx_xvand_v((__m256i)_vlow_leaky, _signmask);
    __m256i _signhigh_leaky = __lasx_xvand_v((__m256i)_vhigh_leaky, _signmask);
    __m256 _p5low_leaky = (__m256)__lasx_xvor_v(_p5, _signlow_leaky);
    __m256 _p5high_leaky = (__m256)__lasx_xvor_v(_p5, _signhigh_leaky);
    __m256 _vlow5_leaky = __lasx_xvfadd_s(_vlow_leaky, _p5low_leaky);
    __m256 _vhigh5_leaky = __lasx_xvfadd_s(_vhigh_leaky, _p5high_leaky);
    __m256i _vlow32_leaky = __lasx_xvftintrz_w_s(_vlow5_leaky);
    __m256i _vhigh32_leaky = __lasx_xvftintrz_w_s(_vhigh5_leaky);

    __m256i _vlow32_16 = __lasx_xvsat_w(_vlow32, 15);
    __m256i _vhigh32_16 = __lasx_xvsat_w(_vhigh32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_vhigh32_16, _vlow32_16);

    __m256i _vlow32_16_leaky = __lasx_xvsat_w(_vlow32_leaky, 15);
    __m256i _vhigh32_16_leaky = __lasx_xvsat_w(_vhigh32_leaky, 15);
    __m256i _v16_leaky = __lasx_xvpickev_h(_vhigh32_16_leaky, _vlow32_16_leaky);

    _v16 = __lasx_xvmax_h(_v16, _v16_leaky);
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);

    return _v8[0];
}
// transpose4x4_epi32 - transpose 4x4 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose4x4_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_w(_r3, _r2);

    _r0 = __lsx_vilvr_d(_tmp2, _tmp0);
    _r1 = __lsx_vilvl_d(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_d(_tmp3, _tmp1);
    _r3 = __lsx_vilvl_d(_tmp3, _tmp1);
}

// transpose4x4_ps - transpose 4x4 block of float (LSX)
static NCNN_FORCEINLINE void transpose4x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r1 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp3, (__m128i)_tmp1);
    _r3 = (__m128)__lsx_vilvl_d((__m128i)_tmp3, (__m128i)_tmp1);
}

// transpose4x8_epi32 - transpose 4x8 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose4x8_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_w(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_w(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvl_w(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_w(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvl_w(_r7, _r6);

    _r0 = __lsx_vilvr_d(_tmp2, _tmp0);
    _r1 = __lsx_vilvr_d(_tmp6, _tmp4);
    _r2 = __lsx_vilvl_d(_tmp2, _tmp0);
    _r3 = __lsx_vilvl_d(_tmp6, _tmp4);
    _r4 = __lsx_vilvr_d(_tmp3, _tmp1);
    _r5 = __lsx_vilvr_d(_tmp7, _tmp5);
    _r6 = __lsx_vilvl_d(_tmp3, _tmp1);
    _r7 = __lsx_vilvl_d(_tmp7, _tmp5);
}

// transpose8x8_epi16 - transpose 8x8 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose8x8_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_h(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_h(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvl_h(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_h(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvl_h(_r7, _r6);

    __m128i _tmp8 = __lsx_vilvr_w(_tmp2, _tmp0);
    __m128i _tmp9 = __lsx_vilvl_w(_tmp2, _tmp0);
    __m128i _tmpa = __lsx_vilvr_w(_tmp3, _tmp1);
    __m128i _tmpb = __lsx_vilvl_w(_tmp3, _tmp1);
    __m128i _tmpc = __lsx_vilvr_w(_tmp6, _tmp4);
    __m128i _tmpd = __lsx_vilvl_w(_tmp6, _tmp4);
    __m128i _tmpe = __lsx_vilvr_w(_tmp7, _tmp5);
    __m128i _tmpf = __lsx_vilvl_w(_tmp7, _tmp5);

    _r0 = __lsx_vilvr_d(_tmp8, _tmpc);
    _r1 = __lsx_vilvl_d(_tmp8, _tmpc);
    _r2 = __lsx_vilvr_d(_tmp9, _tmpd);
    _r3 = __lsx_vilvl_d(_tmp9, _tmpd);
    _r4 = __lsx_vilvr_d(_tmpa, _tmpe);
    _r5 = __lsx_vilvl_d(_tmpa, _tmpe);
    _r6 = __lsx_vilvr_d(_tmpb, _tmpf);
    _r7 = __lsx_vilvl_d(_tmpb, _tmpf);
}

// transpose8x4_epi16 - transpose 8x4 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose8x4_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp2, _tmp0);
    _r1 = __lsx_vilvl_w(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_w(_tmp3, _tmp1);
    _r3 = __lsx_vilvl_w(_tmp3, _tmp1);
}

// transpose8x4_epi8 - transpose 8x4 block of int8 (LSX)
static NCNN_FORCEINLINE void transpose8x4_epi8(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_b(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvr_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp1, _tmp0);
    _r1 = __lsx_vilvl_w(_tmp1, _tmp0);
}

// transpose4x4_epi16 - transpose 4x4 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose4x4_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp2, _tmp0);
    _r1 = __lsx_vilvl_w(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_w(_tmp3, _tmp1);
    _r3 = __lsx_vilvl_w(_tmp3, _tmp1);
}

// transpose8x4_ps - transpose 8x4 block of float (LSX)
static NCNN_FORCEINLINE void transpose8x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r1 = (__m128)__lsx_vilvl_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp3, (__m128i)_tmp1);
    _r3 = (__m128)__lsx_vilvl_d((__m128i)_tmp3, (__m128i)_tmp1);
}

// transpose8x8_ps - transpose 8x8 block of float (LSX)
static NCNN_FORCEINLINE void transpose8x8_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3, __m128& _r4, __m128& _r5, __m128& _r6, __m128& _r7)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvl_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp4 = (__m128)__lsx_vilvr_w((__m128i)_r5, (__m128i)_r4);
    __m128 _tmp5 = (__m128)__lsx_vilvl_w((__m128i)_r5, (__m128i)_r4);
    __m128 _tmp6 = (__m128)__lsx_vilvr_w((__m128i)_r7, (__m128i)_r6);
    __m128 _tmp7 = (__m128)__lsx_vilvl_w((__m128i)_r7, (__m128i)_r6);

    __m128 _tmp8 = (__m128)__lsx_vilvr_w((__m128i)_tmp2, (__m128i)_tmp0);
    __m128 _tmp9 = (__m128)__lsx_vilvl_w((__m128i)_tmp2, (__m128i)_tmp0);
    __m128 _tmpa = (__m128)__lsx_vilvr_w((__m128i)_tmp3, (__m128i)_tmp1);
    __m128 _tmpb = (__m128)__lsx_vilvl_w((__m128i)_tmp3, (__m128i)_tmp1);
    __m128 _tmpc = (__m128)__lsx_vilvr_w((__m128i)_tmp6, (__m128i)_tmp4);
    __m128 _tmpd = (__m128)__lsx_vilvl_w((__m128i)_tmp6, (__m128i)_tmp4);
    __m128 _tmpe = (__m128)__lsx_vilvr_w((__m128i)_tmp7, (__m128i)_tmp5);
    __m128 _tmpf = (__m128)__lsx_vilvl_w((__m128i)_tmp7, (__m128i)_tmp5);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp8, (__m128i)_tmpc);
    _r1 = (__m128)__lsx_vilvl_d((__m128i)_tmp8, (__m128i)_tmpc);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp9, (__m128i)_tmpd);
    _r3 = (__m128)__lsx_vilvl_d((__m128i)_tmp9, (__m128i)_tmpd);
    _r4 = (__m128)__lsx_vilvr_d((__m128i)_tmpa, (__m128i)_tmpe);
    _r5 = (__m128)__lsx_vilvl_d((__m128i)_tmpa, (__m128i)_tmpe);
    _r6 = (__m128)__lsx_vilvr_d((__m128i)_tmpb, (__m128i)_tmpf);
    _r7 = (__m128)__lsx_vilvl_d((__m128i)_tmpb, (__m128i)_tmpf);
}

// transpose8x8_epi32 - transpose 8x8 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose8x8_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvl_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvl_w(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_w(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvl_w(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_w(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvl_w(_r7, _r6);

    __m128i _tmp8 = __lsx_vilvr_w(_tmp2, _tmp0);
    __m128i _tmp9 = __lsx_vilvl_w(_tmp2, _tmp0);
    __m128i _tmpa = __lsx_vilvr_w(_tmp3, _tmp1);
    __m128i _tmpb = __lsx_vilvl_w(_tmp3, _tmp1);
    __m128i _tmpc = __lsx_vilvr_w(_tmp6, _tmp4);
    __m128i _tmpd = __lsx_vilvl_w(_tmp6, _tmp4);
    __m128i _tmpe = __lsx_vilvr_w(_tmp7, _tmp5);
    __m128i _tmpf = __lsx_vilvl_w(_tmp7, _tmp5);

    _r0 = __lsx_vilvr_d(_tmp8, _tmpc);
    _r1 = __lsx_vilvl_d(_tmp8, _tmpc);
    _r2 = __lsx_vilvr_d(_tmp9, _tmpd);
    _r3 = __lsx_vilvl_d(_tmp9, _tmpd);
    _r4 = __lsx_vilvr_d(_tmpa, _tmpe);
    _r5 = __lsx_vilvl_d(_tmpa, _tmpe);
    _r6 = __lsx_vilvr_d(_tmpb, _tmpf);
    _r7 = __lsx_vilvl_d(_tmpb, _tmpf);
}

// transpose16x4_epi8 - transpose 16x4 block of int8 (LSX)
static NCNN_FORCEINLINE void transpose16x4_epi8(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_b(_r0, _r1);
    __m128i _tmp1 = __lsx_vilvl_b(_r0, _r1);
    __m128i _tmp2 = __lsx_vilvr_b(_r2, _r3);
    __m128i _tmp3 = __lsx_vilvl_b(_r2, _r3);

    _r0 = __lsx_vilvr_h(_tmp0, _tmp2);
    _r1 = __lsx_vilvl_h(_tmp0, _tmp2);
    _r2 = __lsx_vilvr_h(_tmp1, _tmp3);
    _r3 = __lsx_vilvl_h(_tmp1, _tmp3);
}

// FMA equivalents (LSX has fmadd/fmsub)
static NCNN_FORCEINLINE __m128 _mm_comp_fmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    return (__m128)__lsx_vfmadd_s((__m128i)_c, (__m128i)_a, (__m128i)_b);
}

static NCNN_FORCEINLINE __m128 _mm_comp_fnmadd_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    // return -a * b + c
    return (__m128)__lsx_vfmsub_s((__m128i)_c, (__m128i)_a, (__m128i)_b);
}

static NCNN_FORCEINLINE __m128 _mm_comp_fmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    // return a * b - c
    return (__m128)__lsx_vfmadd_s((__m128i)_a, (__m128i)_b, (__m128i)_c);
}

static NCNN_FORCEINLINE __m128 _mm_comp_fnmsub_ps(const __m128& _a, const __m128& _b, const __m128& _c)
{
    // return -(a * b) - c
    __m128 _neg_c = (__m128)__lsx_vneg_s((__m128i)_c);
    return (__m128)__lsx_vfmadd_s((__m128i)_neg_c, (__m128i)_a, (__m128i)_b);
}

// reduce operations (LSX)
static NCNN_FORCEINLINE float _mm_reduce_add_ps(const __m128& x)
{
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)x, (__m128i)x);
    __m128 sum64 = (__m128)__lsx_vfadd_s((__m128i)hi64, (__m128i)x);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)sum64);
    __m128 sum32 = (__m128)__lsx_vfadd_s((__m128i)sum64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)sum32, 0, 0);
}

static NCNN_FORCEINLINE float _mm_reduce_max_ps(const __m128& x)
{
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)x, (__m128i)x);
    __m128 max64 = (__m128)__lsx_vfmax_s((__m128i)hi64, (__m128i)x);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)max64);
    __m128 max32 = (__m128)__lsx_vfmax_s((__m128i)max64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)max32, 0, 0);
}

static NCNN_FORCEINLINE int _mm_reduce_add_epi32(const __m128i& x)
{
    // TODO find a more efficient way
    int* _v_p = (int*)&x;
    return _v_p[0] + _v_p[1] + _v_p[2] + _v_p[3];
}

static NCNN_FORCEINLINE int _mm_reduce_max_epi32(const __m128i& x)
{
    __m128i hi64 = __lsx_vilvl_d(x, x);
    __m128i max64 = __lsx_vmax_w(hi64, x);
    __m128i hi32 = __lsx_vilvr_w(hi64, max64);
    __m128i max32 = __lsx_vmax_w(max64, hi32);
    return __lsx_vpickve_w_f(max32, 0, 0);
}

static NCNN_FORCEINLINE int _mm_reduce_min_epi32(const __m128i& x)
{
    __m128i hi64 = __lsx_vilvl_d(x, x);
    __m128i min64 = __lsx_vmin_w(hi64, x);
    __m128i hi32 = __lsx_vilvr_w(hi64, min64);
    __m128i min32 = __lsx_vmin_w(min64, hi32);
    return __lsx_vpickve_w_f(min32, 0, 0);
}

static NCNN_FORCEINLINE int64_t _mm_reduce_add_epi64(const __m128i& x)
{
    __m128i hi64 = __lsx_vilvl_d(x, x);
    __m128i sum64 = __lsx_vadd_d(hi64, x);
    return ((int64_t*)&sum64)[0];
}

// Additional x86 intrinsic wrappers (LSX)

// _mm_xor_si128 - bitwise xor
static NCNN_FORCEINLINE __m128i _mm_xor_si128(__m128i a, __m128i b)
{
    return __lsx_vxor_v(a, b);
}

// _mm_xor_ps - float xor
static NCNN_FORCEINLINE __m128 _mm_xor_ps(__m128 a, __m128 b)
{
    return (__m128)__lsx_vxor_v((__m128i)a, (__m128i)b);
}

// _mm_sub_epi32 - integer subtract
static NCNN_FORCEINLINE __m128i _mm_sub_epi32(__m128i a, __m128i b)
{
    return __lsx_vsub_w(a, b);
}

// _mm_sub_ps - float subtract
static NCNN_FORCEINLINE __m128 _mm_sub_ps(__m128 a, __m128 b)
{
    return (__m128)__lsx_vfsub_s(a, b);
}

// _mm_subs_epi16 - signed saturated subtract
static NCNN_FORCEINLINE __m128i _mm_subs_epi16(__m128i a, __m128i b)
{
    return __lsx_vsubs_s_h(a, b);
}

// _mm_subs_epu8 - unsigned saturated subtract
static NCNN_FORCEINLINE __m128i _mm_subs_epu8(__m128i a, __m128i b)
{
    return (v16i8)__lsx_vsubs_u_b((v16u8)a, (v16u8)b);
}

// _mm_cmpgt_epi32 - greater than compare
static NCNN_FORCEINLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)
{
    return (v4i32)__lsx_vclt_s_w(b, a);
}

// _mm_cmpeq_epi8 - equality compare for bytes
static NCNN_FORCEINLINE __m128i _mm_cmpeq_epi8(__m128i a, __m128i b)
{
    return (v16i8)__lsx_vceq_b((v16i8)a, (v16i8)b);
}

// _mm_cmplt_epi32 - less than compare
static NCNN_FORCEINLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b)
{
    return (v4i32)__lsx_vclt_s_w(a, b);
}

// _mm_cmpeq_epi32 - equality compare for 32-bit integers
static NCNN_FORCEINLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
{
    return (v4i32)__lsx_vceq_w(a, b);
}

// _mm_min_epi16 - signed min for 16-bit
static NCNN_FORCEINLINE __m128i _mm_min_epi16(__m128i a, __m128i b)
{
    return (v8i16)__lsx_vmin_s_h((v8i16)a, (v8i16)b);
}

// _mm_max_epi16 - signed max for 16-bit
static NCNN_FORCEINLINE __m128i _mm_max_epi16(__m128i a, __m128i b)
{
    return (v8i16)__lsx_vmax_s_h((v8i16)a, (v8i16)b);
}

// _mm_min_epu8 - unsigned min for bytes
static NCNN_FORCEINLINE __m128i _mm_min_epu8(__m128i a, __m128i b)
{
    return (v16i8)__lsx_vmin_u_b((v16u8)a, (v16u8)b);
}

// _mm_max_epu8 - unsigned max for bytes
static NCNN_FORCEINLINE __m128i _mm_max_epu8(__m128i a, __m128i b)
{
    return (v16i8)__lsx_vmax_u_b((v16u8)a, (v16u8)b);
}

// _mm_packs_epi32 - pack signed 32-bit to signed 16-bit with saturation
static NCNN_FORCEINLINE __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
    return (v8i16)__lsx_vsat_s_w(a, 15);
}

// _mm_packus_epi16 - pack signed 16-bit to unsigned 8-bit with unsigned saturation
static NCNN_FORCEINLINE __m128i _mm_packus_epi16(__m128i a, __m128i b)
{
    return (v16i8)__lsx_vsat_u_h((v8i16)a, 7);
}

// _mm_cvtepi32_ps - convert int32 to float
static NCNN_FORCEINLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
    return (__m128)__lsx_vffint_s_w(a);
}

// _mm_cvttps_epi32 - convert float to int32 with truncate
static NCNN_FORCEINLINE __m128i _mm_cvttps_epi32(__m128 a)
{
    return __lsx_vftintrz_w_s(a);
}

// _mm_movemask_epi8 - create mask from sign bits of bytes
static NCNN_FORCEINLINE int _mm_movemask_epi8(__m128i a)
{
    v16u8 signs = (v16u8)__lsx_vsrli_b((v16i8)a, 7);
    uint8_t tmp[16];
    __lsx_vst_b((v16i8)signs, (void*)tmp, 0);
    int mask = 0;
    for (int i = 0; i < 16; i++)
        mask |= (tmp[i] & 1) << i;
    return mask;
}

// _mm_blend_epi16 - blend with immediate mask (8 16-bit lanes)
static NCNN_FORCEINLINE __m128i _mm_blend_epi16(__m128i a, __m128i b, int imm)
{
    v8i16 mask = __lsx_vreplgr2vr_h(0);
    if (imm & 1) mask = __lsx_vor_v(mask, __lsx_vreplgr2vr_h(0xFFFF));
    if (imm & 2) mask = (v8i16)__lsx_vor_v((v16u8)mask, (v16u8)__lsx_vslli_h(__lsx_vreplgr2vr_h(0xFFFF), 1));
    if (imm & 4) mask = (v8i16)__lsx_vor_v((v16u8)mask, (v16u8)__lsx_vslli_h(__lsx_vreplgr2vr_h(0xFFFF), 2));
    if (imm & 8) mask = (v8i16)__lsx_vor_v((v16u8)mask, (v16u8)__lsx_vslli_h(__lsx_vreplgr2vr_h(0xFFFF), 3));
    __m128i mask_w = (v4i32)__lsx_vor_v((v16u8)(v8i16)mask, (v16u8)__lsx_vslli_w((v4i32)(v8i16)mask, 4));
    return (v4i32)__lsx_vor_v(
               (v16u8)__lsx_vand_v((v16u8)a, (v16u8)__lsx_vnor_v((v16u8)mask_w, (v16u8)mask_w)),
               (v16u8)__lsx_vand_v((v16u8)b, (v16u8)mask_w));
}

// _mm_andnot_si128 - bitwise and not ( (~a) & b )
static NCNN_FORCEINLINE __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
    return __lsx_vand_v(__lsx_vnor_v(a, a), b);
}

// _mm_srai_epi32 - shift right arithmetic 32-bit integers
static NCNN_FORCEINLINE __m128i _mm_srai_epi32(__m128i a, int imm)
{
    return (v4i32)__lsx_vsrai_w(a, imm);
}

// _mm_slli_epi16 - shift left 16-bit integers
static NCNN_FORCEINLINE __m128i _mm_slli_epi16(__m128i a, int imm)
{
    return (v8i16)__lsx_vslli_h((v8i16)a, imm);
}

// _mm_srli_epi16 - shift right logical 16-bit integers
static NCNN_FORCEINLINE __m128i _mm_srli_epi16(__m128i a, int imm)
{
    return (v8i16)__lsx_vsrli_h((v8i16)a, imm);
}

// _mm_cvtsi128_si32 - extract lowest 32-bit integer
static NCNN_FORCEINLINE int _mm_cvtsi128_si32(__m128i a)
{
    return __lsx_vpickve_w_f(a, 0, 0);
}

// _mm_cvtsi32_si128 - set single 32-bit integer
static NCNN_FORCEINLINE __m128i _mm_cvtsi32_si128(int a)
{
    return __lsx_vreplgr2vr_w(a);
}

// _mm_setr_epi32 - set 4 32-bit integers in reverse order
static NCNN_FORCEINLINE __m128i _mm_setr_epi32(int e0, int e1, int e2, int e3)
{
    int tmp[4] = {e0, e1, e2, e3};
    return __lsx_vld(tmp, 0);
}

// _mm_shuffle_epi32 - shuffle 32-bit integers within 128-bit
static NCNN_FORCEINLINE __m128i _mm_shuffle_epi32(__m128i a, int imm)
{
    return (v4i32)__lsx_vshuf4i_w(a, imm);
}

// _mm_mul_epu32 - multiply unsigned 32-bit to 64-bit (low 64-bit result)
static NCNN_FORCEINLINE __m128i _mm_mul_epu32(__m128i a, __m128i b)
{
    __m128i a_lo = __lsx_vilvr_w(a, __lsx_vreplgr2vr_w(0));
    __m128i b_lo = __lsx_vilvr_w(b, __lsx_vreplgr2vr_w(0));
    return __lsx_vmul_d(a_lo, b_lo);
}

// _mm_movehl_ps - move high to low (float)
static NCNN_FORCEINLINE __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
    return (__m128)__lsx_vilvl_w((__m128i)b, (__m128i)a);
}

// _mm_movelh_ps - move low to high (float)
static NCNN_FORCEINLINE __m128 _mm_movelh_ps(__m128 a, __m128 b)
{
    return (__m128)__lsx_vilvr_w((__m128i)b, (__m128i)a);
}

// rcp_nr
static NCNN_FORCEINLINE __m128 _mm_rcp_nr_ps(const __m128& x)
{
    __m128 y = (__m128)__lsx_vfrcp_s((__m128i)x);
    __m128 t = _mm_comp_fnmadd_ps(x, y, (__m128)__lsx_vreplfr2vr_s(2.0f));
    y = (__m128)__lsx_vfmadd_s((__m128i)y, (__m128i)t, (__m128i)__lsx_vreplfr2vr_s(0.0f));
    return y;
}

// _mm_comp_mullo_epi32
static NCNN_FORCEINLINE __m128i _mm_comp_mullo_epi32(const __m128i& a, const __m128i& b)
{
    return __lsx_vmul_w(a, b);
}

// BF16 conversion utilities (LSX)
static NCNN_FORCEINLINE __m128 bfloat2float_sse(const __m128i& v0)
{
    __m128i _zero = __lsx_vreplgr2vr_w(0);
    __m128i _a = __lsx_vilvr_w(v0, _zero);
    return (__m128)_a;
}

static NCNN_FORCEINLINE __m128i float2bfloat_sse(const __m128& v)
{
    __m128i _a = (__m128i)v;
    _a = (v4i32)__lsx_vsrli_w((v4i32)_a, 16);
    __m128i _v = __lsx_vpickev_h(_a, __lsx_vreplgr2vr_w(0));
    return _v;
}

static NCNN_FORCEINLINE __m128i float2bfloat_sse(const __m128& v0, const __m128& v1)
{
    __m128i _a = (__m128i)v0;
    __m128i _b = (__m128i)v1;
    _a = (v4i32)__lsx_vsrli_w((v4i32)_a, 16);
    _b = (v4i32)__lsx_vsrli_w((v4i32)_b, 16);
    __m128i _v = __lsx_vpickev_h(_b, _a);
    return _v;
}

// DPWSSD for INT8 dot product (LSX)
static NCNN_FORCEINLINE __m128i _mm_comp_dpwssd_epi32(const __m128i& src, const __m128i& a, const __m128i& b)
{
    return __lsx_vdpadd_s_w(src, a, b);
}

// DPBUSSD for unsigned byte dot product (LSX)
static NCNN_FORCEINLINE __m128i _mm_comp_dpbusd_epi32(const __m128i& src, const __m128i& a, const __m128i& b)
{
    // Unpack unsigned bytes to unsigned 16-bit, then multiply and accumulate
    __m128i a_lo = __lsx_vilvr_b(a, __lsx_vreplgr2vr_w(0));
    __m128i a_hi = __lsx_vilvl_b(a, __lsx_vreplgr2vr_w(0));
    __m128i b_lo = __lsx_vilvr_b(b, __lsx_vreplgr2vr_w(0));
    __m128i b_hi = __lsx_vilvl_b(b, __lsx_vreplgr2vr_w(0));
    __m128i prod_lo = __lsx_vmul_h(a_lo, b_lo);
    __m128i prod_hi = __lsx_vmul_h(a_hi, b_hi);
    __m128i sum = __lsx_vadd_w(prod_lo, prod_hi);
    return __lsx_vadd_w(sum, src);
}

// DPWSSDS - signed saturated version (LSX)
static NCNN_FORCEINLINE __m128i _mm_comp_dpwssds_epi32(const __m128i& src, const __m128i& a, const __m128i& b)
{
    __m128i prod = __lsx_vdpadd_s_w(src, a, b);
    return __lsx_vsat_s_w(prod, 31);
}

// DPBUSSDS - unsigned saturated version (LSX)
static NCNN_FORCEINLINE __m128i _mm_comp_dpbusds_epi32(const __m128i& src, const __m128i& a, const __m128i& b)
{
    __m128i dp = _mm_comp_dpbusd_epi32(src, a, b);
    return __lsx_vsat_s_w(dp, 31);
}

// int8_short_to_int32_scalar
static NCNN_FORCEINLINE int32_t int8_short_to_int32_scalar(int8_t* v0)
{
    int32_t _v0 = v0[0] + (v0[1] << 8) + (v0[2] << 16) + (v0[3] << 24);
    return _v0;
}

// HorizontalSums for 8 accumulators (LSX)
static NCNN_FORCEINLINE __m128 HorizontalSums(__m128& v0, __m128& v1, __m128& v2, __m128& v3, __m128& v4, __m128& v5, __m128& v6, __m128& v7)
{
    __m128 s01 = (__m128)__lsx_vfadd_s((__m128i)v0, (__m128i)v1);
    __m128 s23 = (__m128)__lsx_vfadd_s((__m128i)v2, (__m128i)v3);
    __m128 s45 = (__m128)__lsx_vfadd_s((__m128i)v4, (__m128i)v5);
    __m128 s67 = (__m128)__lsx_vfadd_s((__m128i)v6, (__m128i)v7);
    __m128 s0123 = (__m128)__lsx_vfadd_s((__m128i)s01, (__m128i)s23);
    __m128 s4556 = (__m128)__lsx_vfadd_s((__m128i)s45, (__m128i)s67);
    return (__m128)__lsx_vfadd_s((__m128i)s0123, (__m128i)s4556);
}

// fast integer division (LSX)
class FastDivider_epu32
{
public:
    NCNN_FORCEINLINE FastDivider_epu32(unsigned int d)
    {
        unsigned int m, sh1, sh2;
        if (d == 1)
        {
            m = 1;
            sh1 = 0;
            sh2 = 0;
        }
        else
        {
            uint32_t sh = portable_ceil_log2(d);
            uint32_t m0 = sh == 32 ? 0 : 1 << sh;
            m = 1 + uint32_t((uint64_t(m0 - d) << 32) / d);
            sh1 = 1;
            sh2 = sh - 1;
        }
        _multiplier = __lsx_vreplgr2vr_w(m);
        _shift1 = __lsx_vreplgr2vr_w(sh1);
        _shift2 = __lsx_vreplgr2vr_w(sh2);
    }

    NCNN_FORCEINLINE __m128i _mm_comp_div_epu32(const __m128i& x) const
    {
        __m128i xm_low = __lsx_vsrli_w(__lsx_vdpadd_s_w(__lsx_vreplgr2vr_w(0), x, _multiplier), 32);
        __m128i xm = __lsx_vor_v(xm_low, __lsx_vand_v(__lsx_vsrli_w(x, 32), __lsx_vreplgr2vr_w(0xFFFF)));
        return __lsx_vsrli_w(__lsx_vadd_w(xm, __lsx_vsrli_w(__lsx_vsub_w(x, xm), _shift1)), (v4u32)_shift2);
    }

protected:
    static int portable_ceil_log2(int d)
    {
        return 32 - __builtin_clz(d - 1);
    }

protected:
    __m128i _multiplier;
    __m128i _shift1;
    __m128i _shift2;
};

#if __loongarch_asx
// LASX 256-bit variants

// combine functions
static NCNN_FORCEINLINE __m256 combine4x2_ps(const __m128& a, const __m128& b)
{
    return (__m256)__lasx_xvinsgr2vr_w((__m256i)__lasx_xvpermi_q_w((__m256i)__lasx_xvpickeven_w((__m256i)a, (__m256i)b), (__m256i)__lasx_xvpickodd_w((__m256i)a, (__m256i)b), 0x20), *(int*)&b, 7);
}

static NCNN_FORCEINLINE __m256i combine4x2_epi32(const __m128i& a, const __m128i& b)
{
    return __lasx_xvinsgr2vr_w(__lasx_xvpermi_q_w(a, b, 0x20), ((int*)&b)[3], 7);
}

// transpose8x8_ps - transpose 8x8 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x8_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7)
{
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvl_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp2 = (__m256)__lasx_xvilvr_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp3 = (__m256)__lasx_xvilvl_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp4 = (__m256)__lasx_xvilvr_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp5 = (__m256)__lasx_xvilvl_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp6 = (__m256)__lasx_xvilvr_w((__m256i)_r7, (__m256i)_r6);
    __m256 _tmp7 = (__m256)__lasx_xvilvl_w((__m256i)_r7, (__m256i)_r6);

    __m256 _tmp8 = (__m256)__lasx_xvilvr_w((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp9 = (__m256)__lasx_xvilvl_w((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmpa = (__m256)__lasx_xvilvr_w((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpb = (__m256)__lasx_xvilvl_w((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpc = (__m256)__lasx_xvilvr_w((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmpd = (__m256)__lasx_xvilvl_w((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmpe = (__m256)__lasx_xvilvr_w((__m256i)_tmp7, (__m256i)_tmp5);
    __m256 _tmpf = (__m256)__lasx_xvilvl_w((__m256i)_tmp7, (__m256i)_tmp5);

    _r0 = (__m256)__lasx_xvilvr_d((__m256i)_tmp8, (__m256i)_tmpc);
    _r1 = (__m256)__lasx_xvilvl_d((__m256i)_tmp8, (__m256i)_tmpc);
    _r2 = (__m256)__lasx_xvilvr_d((__m256i)_tmp9, (__m256i)_tmpd);
    _r3 = (__m256)__lasx_xvilvl_d((__m256i)_tmp9, (__m256i)_tmpd);
    _r4 = (__m256)__lasx_xvilvr_d((__m256i)_tmpa, (__m256i)_tmpe);
    _r5 = (__m256)__lasx_xvilvl_d((__m256i)_tmpa, (__m256i)_tmpe);
    _r6 = (__m256)__lasx_xvilvr_d((__m256i)_tmpb, (__m256i)_tmpf);
    _r7 = (__m256)__lasx_xvilvl_d((__m256i)_tmpb, (__m256i)_tmpf);
}

// transpose8x4_ps - transpose 8x4 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x4_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3)
{
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvl_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp2 = (__m256)__lasx_xvilvr_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp3 = (__m256)__lasx_xvilvl_w((__m256i)_r3, (__m256i)_r2);

    __m256 _tmp4 = (__m256)__lasx_xvilvr_w((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp5 = (__m256)__lasx_xvilvl_w((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp6 = (__m256)__lasx_xvilvr_w((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmp7 = (__m256)__lasx_xvilvl_w((__m256i)_tmp3, (__m256i)_tmp1);

    _r0 = (__m256)__lasx_xvilvr_d((__m256i)_tmp4, (__m256i)_tmp5);
    _r1 = (__m256)__lasx_xvilvl_d((__m256i)_tmp4, (__m256i)_tmp5);
    _r2 = (__m256)__lasx_xvilvr_d((__m256i)_tmp6, (__m256i)_tmp7);
    _r3 = (__m256)__lasx_xvilvl_d((__m256i)_tmp6, (__m256i)_tmp7);
}

// transpose8x2_ps - transpose 8x2 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x2_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvl_w((__m256i)_r1, (__m256i)_r0);

    _r0 = (__m256)__lasx_xvilvr_d((__m256i)_tmp0, (__m256i)_tmp1);
    _r1 = (__m256)__lasx_xvilvl_d((__m256i)_tmp0, (__m256i)_tmp1);
}

// transpose2x8_ps - transpose 2x8 block of float (LASX)
static NCNN_FORCEINLINE void transpose2x8_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = __lasx_xvpermi_q_w((__m256i)_r0, (__m256i)_r1, 0x20);
    __m256 _tmp1 = __lasx_xvpermi_q_w((__m256i)_r0, (__m256i)_r1, 0x31);

    _r0 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp0, 0x88);
    _r1 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp1, 0xDD);
}

// transpose8x8_epi32 - transpose 8x8 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x8_epi32(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_w(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_w(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvl_w(_r3, _r2);
    __m256i _tmp4 = __lasx_xvilvr_w(_r5, _r4);
    __m256i _tmp5 = __lasx_xvilvl_w(_r5, _r4);
    __m256i _tmp6 = __lasx_xvilvr_w(_r7, _r6);
    __m256i _tmp7 = __lasx_xvilvl_w(_r7, _r6);

    __m256i _tmp8 = __lasx_xvilvr_w(_tmp2, _tmp0);
    __m256i _tmp9 = __lasx_xvilvl_w(_tmp2, _tmp0);
    __m256i _tmpa = __lasx_xvilvr_w(_tmp3, _tmp1);
    __m256i _tmpb = __lasx_xvilvl_w(_tmp3, _tmp1);
    __m256i _tmpc = __lasx_xvilvr_w(_tmp6, _tmp4);
    __m256i _tmpd = __lasx_xvilvl_w(_tmp6, _tmp4);
    __m256i _tmpe = __lasx_xvilvr_w(_tmp7, _tmp5);
    __m256i _tmpf = __lasx_xvilvl_w(_tmp7, _tmp5);

    _r0 = __lasx_xvilvr_d(_tmp8, _tmpc);
    _r1 = __lasx_xvilvl_d(_tmp8, _tmpc);
    _r2 = __lasx_xvilvr_d(_tmp9, _tmpd);
    _r3 = __lasx_xvilvl_d(_tmp9, _tmpd);
    _r4 = __lasx_xvilvr_d(_tmpa, _tmpe);
    _r5 = __lasx_xvilvl_d(_tmpa, _tmpe);
    _r6 = __lasx_xvilvr_d(_tmpb, _tmpf);
    _r7 = __lasx_xvilvl_d(_tmpb, _tmpf);
}

// transpose8x4_epi32 - transpose 8x4 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x4_epi32(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_w(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_w(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvl_w(_r3, _r2);

    __m256i _tmp4 = __lasx_xvilvr_w(_tmp2, _tmp0);
    __m256i _tmp5 = __lasx_xvilvl_w(_tmp2, _tmp0);
    __m256i _tmp6 = __lasx_xvilvr_w(_tmp3, _tmp1);
    __m256i _tmp7 = __lasx_xvilvl_w(_tmp3, _tmp1);

    _r0 = __lasx_xvilvr_d(_tmp4, _tmp5);
    _r1 = __lasx_xvilvl_d(_tmp4, _tmp5);
    _r2 = __lasx_xvilvr_d(_tmp6, _tmp7);
    _r3 = __lasx_xvilvl_d(_tmp6, _tmp7);
}

// transpose8x2_epi32 - transpose 8x2 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x2_epi32(__m256i& _r0, __m256i& _r1)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_w(_r1, _r0);

    _r0 = __lasx_xvilvr_d(_tmp0, _tmp1);
    _r1 = __lasx_xvilvl_d(_tmp0, _tmp1);
}

// transpose16x4_epi16 - transpose 16x4 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x4_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_h(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_h(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvl_h(_r3, _r2);

    __m256i _tmp4 = __lasx_xvilvr_w(_tmp0, _tmp2);
    __m256i _tmp5 = __lasx_xvilvl_w(_tmp0, _tmp2);
    __m256i _tmp6 = __lasx_xvilvr_w(_tmp1, _tmp3);
    __m256i _tmp7 = __lasx_xvilvl_w(_tmp1, _tmp3);

    _r0 = __lasx_xvilvr_d(_tmp4, _tmp5);
    _r1 = __lasx_xvilvl_d(_tmp4, _tmp5);
    _r2 = __lasx_xvilvr_d(_tmp6, _tmp7);
    _r3 = __lasx_xvilvl_d(_tmp6, _tmp7);
}

// transpose16x2_epi16 - transpose 16x2 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x2_epi16(__m256i& _r0, __m256i& _r1)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_h(_r1, _r0);

    _r0 = __lasx_xvilvr_d(_tmp0, _tmp1);
    _r1 = __lasx_xvilvl_d(_tmp0, _tmp1);
}

// transpose16x8_epi16 - transpose 16x8 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x8_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvl_h(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_h(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvl_h(_r3, _r2);
    __m256i _tmp4 = __lasx_xvilvr_h(_r5, _r4);
    __m256i _tmp5 = __lasx_xvilvl_h(_r5, _r4);
    __m256i _tmp6 = __lasx_xvilvr_h(_r7, _r6);
    __m256i _tmp7 = __lasx_xvilvl_h(_r7, _r6);

    __m256i _tmpg = __lasx_xvilvr_w(_tmp0, _tmp2);
    __m256i _tmph = __lasx_xvilvl_w(_tmp0, _tmp2);
    __m256i _tmpi = __lasx_xvilvr_w(_tmp1, _tmp3);
    __m256i _tmpj = __lasx_xvilvl_w(_tmp1, _tmp3);
    __m256i _tmpk = __lasx_xvilvr_w(_tmp4, _tmp6);
    __m256i _tmpl = __lasx_xvilvl_w(_tmp4, _tmp6);
    __m256i _tmpm = __lasx_xvilvr_w(_tmp5, _tmp7);
    __m256i _tmpn = __lasx_xvilvl_w(_tmp5, _tmp7);

    _tmp0 = __lasx_xvilvr_d(_tmpg, _tmpk);
    _tmp1 = __lasx_xvilvl_d(_tmpg, _tmpk);
    _tmp2 = __lasx_xvilvr_d(_tmph, _tmpl);
    _tmp3 = __lasx_xvilvl_d(_tmph, _tmpl);
    _tmp4 = __lasx_xvilvr_d(_tmpi, _tmpm);
    _tmp5 = __lasx_xvilvl_d(_tmpi, _tmpm);
    _tmp6 = __lasx_xvilvr_d(_tmpj, _tmpn);
    _tmp7 = __lasx_xvilvl_d(_tmpj, _tmpn);

    _r0 = __lasx_xvpermi_q_d(_tmp0, _tmp1, 0x20);
    _r1 = __lasx_xvpermi_q_d(_tmp2, _tmp3, 0x20);
    _r2 = __lasx_xvpermi_q_d(_tmp4, _tmp5, 0x20);
    _r3 = __lasx_xvpermi_q_d(_tmp6, _tmp7, 0x20);
    _r4 = __lasx_xvpermi_q_d(_tmp0, _tmp1, 0x31);
    _r5 = __lasx_xvpermi_q_d(_tmp2, _tmp3, 0x31);
    _r6 = __lasx_xvpermi_q_d(_tmp4, _tmp5, 0x31);
    _r7 = __lasx_xvpermi_q_d(_tmp6, _tmp7, 0x31);
}

// FMA equivalents (LASX)
static NCNN_FORCEINLINE __m256 _mm256_comp_fmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return (__m256)__lasx_xvfmadd_s((__m256i)_c, (__m256i)_a, (__m256i)_b);
}

static NCNN_FORCEINLINE __m256 _mm256_comp_fnmadd_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    // return -a * b + c
    return (__m256)__lasx_xvfmsub_s((__m256i)_c, (__m256i)_a, (__m256i)_b);
}

static NCNN_FORCEINLINE __m256 _mm256_comp_fmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    return (__m256)__lasx_xvfmadd_s((__m256i)_a, (__m256i)_b, (__m256i)_c);
}

static NCNN_FORCEINLINE __m256 _mm256_comp_fnmsub_ps(const __m256& _a, const __m256& _b, const __m256& _c)
{
    __m256 _neg_c = (__m256)__lasx_xvneg_s((__m256i)_c);
    return (__m256)__lasx_xvfmadd_s((__m256i)_neg_c, (__m256i)_a, (__m256i)_b);
}

// rcp_nr (LASX)
static NCNN_FORCEINLINE __m256 _mm256_rcp_nr_ps(const __m256& x)
{
    __m256 y = (__m256)__lasx_xvfrcp_s((__m256i)x);
    __m256 t = _mm256_comp_fnmadd_ps(x, y, (__m256)__lasx_xvreplfr2vr_s(2.0f));
    y = (__m256)__lasx_xvfmadd_s((__m256i)y, (__m256i)t, (__m256i)__lasx_xvreplfr2vr_s(0.0f));
    return y;
}

// _mm256_fmadd_1_ps and _mm256_fmrsub_1_ps
static NCNN_FORCEINLINE __m256 _mm256_fmadd_1_ps(const __m256& a, const __m256& b, float c)
{
    return _mm256_comp_fmadd_ps(b, (__m256)__lasx_xvreplfr2vr_s(c), a);
}

static NCNN_FORCEINLINE __m256 _mm256_fmrsub_1_ps(const __m256& a, const __m256& b, float c)
{
    return _mm256_comp_fnmadd_ps(b, (__m256)__lasx_xvreplfr2vr_s(c), a);
}

// _mm256_comp_fmadd_ps4 and _mm256_comp_fmadd_ps8
static NCNN_FORCEINLINE void _mm256_comp_fmadd_ps4(__m256& _sum,
        const __m256& _w0, const __m256& _w1, const __m256& _w2, const __m256& _w3,
        const __m256& _v0, const __m256& _v1, const __m256& _v2, const __m256& _v3)
{
    __m256 _mul0 = (__m256)__lasx_xvfmul_s((__m256i)_w0, (__m256i)_v0);
    __m256 _mul1 = (__m256)__lasx_xvfmul_s((__m256i)_w1, (__m256i)_v1);
    __m256 _sum01 = (__m256)__lasx_xvfadd_s((__m256i)_mul0, (__m256i)_mul1);
    __m256 _mul2 = (__m256)__lasx_xvfmul_s((__m256i)_w2, (__m256i)_v2);
    __m256 _mul3 = (__m256)__lasx_xvfmul_s((__m256i)_w3, (__m256i)_v3);
    __m256 _sum23 = (__m256)__lasx_xvfadd_s((__m256i)_mul2, (__m256i)_mul3);
    __m256 _sum0123 = (__m256)__lasx_xvfadd_s((__m256i)_sum01, (__m256i)_sum23);
    _sum = (__m256)__lasx_xvfadd_s((__m256i)_sum, (__m256i)_sum0123);
}

static NCNN_FORCEINLINE void _mm256_comp_fmadd_ps8(__m256& _sum,
        const __m256& _w0, const __m256& _w1, const __m256& _w2, const __m256& _w3, const __m256& _w4, const __m256& _w5, const __m256& _w6, const __m256& _w7,
        const __m256& _v0, const __m256& _v1, const __m256& _v2, const __m256& _v3, const __m256& _v4, const __m256& _v5, const __m256& _v6, const __m256& _v7)
{
    _mm256_comp_fmadd_ps4(_sum, _w0, _w1, _w2, _w3, _v0, _v1, _v2, _v3);
    _mm256_comp_fmadd_ps4(_sum, _w4, _w5, _w6, _w7, _v4, _v5, _v6, _v7);
}

// reduce operations (LASX)
static NCNN_FORCEINLINE float _mm256_reduce_add_ps(const __m256& x)
{
    __m128 lo = __lasx_xvpickve_w_f((__m256i)x, 0, 0);
    __m128 hi = __lasx_xvpickve_w_f((__m256i)x, 4, 0);
    __m128 sum = (__m128)__lsx_vfadd_s((__m128i)lo, (__m128i)hi);
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)sum, (__m128i)sum);
    __m128 sum64 = (__m128)__lsx_vfadd_s((__m128i)hi64, (__m128i)sum);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)sum64);
    __m128 sum32 = (__m128)__lsx_vfadd_s((__m128i)sum64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)sum32, 0, 0);
}

static NCNN_FORCEINLINE float _mm256_reduce_max_ps(const __m256& x)
{
    __m128 lo = __lasx_xvpickve_w_f((__m256i)x, 0, 0);
    __m128 hi = __lasx_xvpickve_w_f((__m256i)x, 4, 0);
    __m128 maxv = (__m128)__lsx_vfmax_s((__m128i)lo, (__m128i)hi);
    __m128 hi64 = (__m128)__lsx_vilvl_d((__m128i)maxv, (__m128i)maxv);
    __m128 max64 = (__m128)__lsx_vfmax_s((__m128i)hi64, (__m128i)maxv);
    __m128 hi32 = (__m128)__lsx_vilvr_w((__m128i)hi64, (__m128i)max64);
    __m128 max32 = (__m128)__lsx_vfmax_s((__m128i)max64, (__m128i)hi32);
    return __lsx_vpickve_w_f((__m128i)max32, 0, 0);
}

static NCNN_FORCEINLINE int _mm256_reduce_add_epi32(const __m256i& x)
{
    __m256i hi64 = __lasx_xvilvl_d(x, x);
    __m256i sum64 = __lasx_xvadd_d(hi64, x);
    __m256i hi32 = __lasx_xvilvr_w(hi64, sum64);
    __m256i sum32 = __lasx_xvadd_w(sum64, hi32);
    return __lasx_xvpickve_w_f(sum32, 0, 0);
}

static NCNN_FORCEINLINE int _mm256_reduce_max_epi32(const __m256i& x)
{
    __m256i hi64 = __lasx_xvilvl_d(x, x);
    __m256i max64 = __lasx_xvmax_w(hi64, x);
    __m256i hi32 = __lasx_xvilvr_w(hi64, max64);
    __m256i max32 = __lasx_xvmax_w(max64, hi32);
    return __lasx_xvpickve_w_f(max32, 0, 0);
}

static NCNN_FORCEINLINE int _mm256_reduce_min_epi32(const __m256i& x)
{
    __m256i hi64 = __lasx_xvilvl_d(x, x);
    __m256i min64 = __lasx_xvmin_w(hi64, x);
    __m256i hi32 = __lasx_xvilvr_w(hi64, min64);
    __m256i min32 = __lasx_xvmin_w(min64, hi32);
    return __lasx_xvpickve_w_f(min32, 0, 0);
}

static NCNN_FORCEINLINE int64_t _mm256_reduce_add_epi64(const __m256i& x)
{
    __m256i hi64 = __lasx_xvilvl_d(x, x);
    __m256i sum64 = __lasx_xvadd_d(hi64, x);
    return ((int64_t*)&sum64)[0];
}

// Additional x86 intrinsic wrappers (LASX)

// _mm256_xor_si256 - bitwise xor
static NCNN_FORCEINLINE __m256i _mm256_xor_si256(__m256i a, __m256i b)
{
    return __lasx_xvxor_v(a, b);
}

// _mm256_xor_ps - float xor
static NCNN_FORCEINLINE __m256 _mm256_xor_ps(__m256 a, __m256 b)
{
    return (__m256)__lasx_xvxor_v((__m256i)a, (__m256i)b);
}

// _mm256_sub_epi32 - integer subtract
static NCNN_FORCEINLINE __m256i _mm256_sub_epi32(__m256i a, __m256i b)
{
    return __lasx_xvsub_w(a, b);
}

// _mm256_sub_ps - float subtract
static NCNN_FORCEINLINE __m256 _mm256_sub_ps(__m256 a, __m256 b)
{
    return (__m256)__lasx_xvfsub_s(a, b);
}

// _mm256_subs_epi16 - signed saturated subtract
static NCNN_FORCEINLINE __m256i _mm256_subs_epi16(__m256i a, __m256i b)
{
    return (v16i8)__lasx_xvsubs_s_h((v16i16)a, (v16i16)b);
}

// _mm256_subs_epu8 - unsigned saturated subtract
static NCNN_FORCEINLINE __m256i _mm256_subs_epu8(__m256i a, __m256i b)
{
    return (v16i8)__lasx_xvsubs_u_b((v32u8)a, (v32u8)b);
}

// _mm256_cmpgt_epi32 - greater than compare
static NCNN_FORCEINLINE __m256i _mm256_cmpgt_epi32(__m256i a, __m256i b)
{
    return (v8i32)__lasx_xvclt_s_w(b, a);
}

// _mm256_cmpeq_epi8 - equality compare for bytes
static NCNN_FORCEINLINE __m256i _mm256_cmpeq_epi8(__m256i a, __m256i b)
{
    return (v32i8)__lasx_xvceq_b((v32i8)a, (v32i8)b);
}

// _mm256_cmplt_epi32 - less than compare
static NCNN_FORCEINLINE __m256i _mm256_cmplt_epi32(__m256i a, __m256i b)
{
    return (v8i32)__lasx_xvclt_s_w(a, b);
}

// _mm256_cmpeq_epi32 - equality compare for 32-bit integers
static NCNN_FORCEINLINE __m256i _mm256_cmpeq_epi32(__m256i a, __m256i b)
{
    return (v8i32)__lasx_xvceq_w(a, b);
}

// _mm256_min_epi16 - signed min for 16-bit
static NCNN_FORCEINLINE __m256i _mm256_min_epi16(__m256i a, __m256i b)
{
    return (v16i16)__lasx_xvmin_s_h((v16i16)a, (v16i16)b);
}

// _mm256_max_epi16 - signed max for 16-bit
static NCNN_FORCEINLINE __m256i _mm256_max_epi16(__m256i a, __m256i b)
{
    return (v16i16)__lasx_xvmax_s_h((v16i16)a, (v16i16)b);
}

// _mm256_min_epu8 - unsigned min for bytes
static NCNN_FORCEINLINE __m256i _mm256_min_epu8(__m256i a, __m256i b)
{
    return (v32i8)__lasx_xvmin_u_b((v32u8)a, (v32u8)b);
}

// _mm256_max_epu8 - unsigned max for bytes
static NCNN_FORCEINLINE __m256i _mm256_max_epu8(__m256i a, __m256i b)
{
    return (v32i8)__lasx_xvmax_u_b((v32u8)a, (v32u8)b);
}

// _mm256_packs_epi32 - pack signed 32-bit to signed 16-bit with saturation
static NCNN_FORCEINLINE __m256i _mm256_packs_epi32(__m256i a, __m256i b)
{
    return (v16i16)__lasx_xvsat_s_w(a, 15);
}

// _mm256_packus_epi16 - pack signed 16-bit to unsigned 8-bit
static NCNN_FORCEINLINE __m256i _mm256_packus_epi16(__m256i a, __m256i b)
{
    return (v32i8)__lasx_xvsat_u_h((v16i16)a, 7);
}

// _mm256_cvtepi32_ps - convert int32 to float
static NCNN_FORCEINLINE __m256 _mm256_cvtepi32_ps(__m256i a)
{
    return (__m256)__lasx_xvffint_s_w(a);
}

// _mm256_cvttps_epi32 - convert float to int32 with truncate
static NCNN_FORCEINLINE __m256i _mm256_cvttps_epi32(__m256 a)
{
    return __lasx_xvftintrz_w_s(a);
}

// _mm256_movemask_epi8 - create mask from sign bits of bytes
static NCNN_FORCEINLINE int _mm256_movemask_epi8(__m256i a)
{
    v32u8 signs = (v32u8)__lasx_xvsrli_b((v32i8)a, 7);
    uint8_t tmp[32];
    __lasx_xvst_b((v32i8)signs, (void*)tmp, 0);
    int mask = 0;
    for (int i = 0; i < 32; i++)
        mask |= (tmp[i] & 1) << i;
    return mask;
}

// _mm256_blend_epi16 - blend with immediate mask (16 16-bit lanes)
static NCNN_FORCEINLINE __m256i _mm256_blend_epi16(__m256i a, __m256i b, int imm)
{
    __m256i mask = __lasx_xvreplgr2vr_h(0);
    for (int i = 0; i < 8; i++)
    {
        if (imm & (1 << i))
            mask = __lasx_xvor_v(mask, __lasx_xvslli_h(__lasx_xvreplgr2vr_h(0xFFFF), i));
    }
    __m256i mask_w = __lasx_xvor_v(mask, __lasx_xvslli_w(mask, 8));
    return __lasx_xvor_v(
               __lasx_xvand_v(a, __lasx_xvnor_v(mask_w, mask_w)),
               __lasx_xvand_v(b, mask_w));
}

// _mm256_andnot_si256 - bitwise and not
static NCNN_FORCEINLINE __m256i _mm256_andnot_si256(__m256i a, __m256i b)
{
    return __lasx_xvand_v(__lasx_xvnor_v(a, a), b);
}

// _mm256_srai_epi32 - shift right arithmetic 32-bit integers
static NCNN_FORCEINLINE __m256i _mm256_srai_epi32(__m256i a, int imm)
{
    return (v8i32)__lasx_xvsrai_w(a, imm);
}

// _mm256_slli_epi16 - shift left 16-bit integers
static NCNN_FORCEINLINE __m256i _mm256_slli_epi16(__m256i a, int imm)
{
    return (v16i16)__lasx_xvslli_h((v16i16)a, imm);
}

// _mm256_srli_epi16 - shift right logical 16-bit integers
static NCNN_FORCEINLINE __m256i _mm256_srli_epi16(__m256i a, int imm)
{
    return (v16i16)__lasx_xvsrli_h((v16i16)a, imm);
}

// _mm256_srli_epi32 - shift right logical 32-bit integers
static NCNN_FORCEINLINE __m256i _mm256_srli_epi32(__m256i a, int imm)
{
    return (v8i32)__lasx_xvsrli_w(a, imm);
}

// _mm256_slli_epi32 - shift left 32-bit integers
static NCNN_FORCEINLINE __m256i _mm256_slli_epi32(__m256i a, int imm)
{
    return (v8i32)__lasx_xvslli_w(a, imm);
}

// _mm256_cvtsi256_si32 - extract lowest 32-bit integer
static NCNN_FORCEINLINE int _mm256_cvtsi256_si32(__m256i a)
{
    return __lasx_xvpickve_w_f(a, 0, 0);
}

// _mm256_cvtsi32_si256 - set single 32-bit integer
static NCNN_FORCEINLINE __m256i _mm256_cvtsi32_si256(int a)
{
    return __lasx_xvreplgr2vr_w(a);
}

// _mm256_setr_epi32 - set 8 32-bit integers in reverse order
static NCNN_FORCEINLINE __m256i _mm256_setr_epi32(int e0, int e1, int e2, int e3, int e4, int e5, int e6, int e7)
{
    int tmp[8] = {e0, e1, e2, e3, e4, e5, e6, e7};
    return __lasx_xvld(tmp, 0);
}

// _mm256_shuffle_epi32 - shuffle 32-bit integers within 256-bit
static NCNN_FORCEINLINE __m256i _mm256_shuffle_epi32(__m256i a, int imm)
{
    return (v8i32)__lasx_xvshuf4i_w(a, imm);
}

// _mm256_mul_epu32 - multiply unsigned 32-bit to 64-bit (low 64-bit result)
static NCNN_FORCEINLINE __m256i _mm256_mul_epu32(__m256i a, __m256i b)
{
    __m256i a_lo = __lasx_xvilvr_w(a, __lasx_xvreplgr2vr_w(0));
    __m256i b_lo = __lasx_xvilvr_w(b, __lasx_xvreplgr2vr_w(0));
    return __lasx_xvmul_d(a_lo, b_lo);
}

// _mm256_movehl_ps - move high to low (float)
static NCNN_FORCEINLINE __m256 _mm256_movehl_ps(__m256 a, __m256 b)
{
    return (__m256)__lasx_xvilvl_w((__m256i)b, (__m256i)a);
}

// _mm256_movelh_ps - move low to high (float)
static NCNN_FORCEINLINE __m256 _mm256_movelh_ps(__m256 a, __m256 b)
{
    return (__m256)__lasx_xvilvr_w((__m256i)b, (__m256i)a);
}

// HorizontalSums (LASX)
static NCNN_FORCEINLINE __m256 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3, __m256& v4, __m256& v5, __m256& v6, __m256& v7)
{
    __m256 s01 = (__m256)__lasx_xvfadd_s((__m256i)v0, (__m256i)v1);
    __m256 s23 = (__m256)__lasx_xvfadd_s((__m256i)v2, (__m256i)v3);
    __m256 s45 = (__m256)__lasx_xvfadd_s((__m256i)v4, (__m256i)v5);
    __m256 s67 = (__m256)__lasx_xvfadd_s((__m256i)v6, (__m256i)v7);
    __m256 s0123 = (__m256)__lasx_xvfadd_s((__m256i)s01, (__m256i)s23);
    __m256 s4556 = (__m256)__lasx_xvfadd_s((__m256i)s45, (__m256i)s67);

    __m256 vb0 = __lasx_xvshuf4i_w((__m256i)s0123, 0xD8);
    __m256 vb1 = __lasx_xvpermi_q_w((__m256i)s0123, (__m256i)s4556, 0x31);

    return (__m256)__lasx_xvfadd_s((__m256i)vb0, (__m256i)vb1);
}

static NCNN_FORCEINLINE __m128 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3)
{
    __m256 s01 = (__m256)__lasx_xvfadd_s((__m256i)v0, (__m256i)v1);
    __m256 s23 = (__m256)__lasx_xvfadd_s((__m256i)v2, (__m256i)v3);
    __m256 s0123 = (__m256)__lasx_xvfadd_s((__m256i)s01, (__m256i)s23);

    return (__m128)__lasx_xvpickve_w_f((__m256i)s0123, 0, 0);
}

// BF16 conversion (LASX)
static NCNN_FORCEINLINE __m256 bfloat2float_avx(const __m128i& v0)
{
    __m128i _zero = __lsx_vreplgr2vr_w(0);
    __m128i _a = __lsx_vilvr_w(v0, _zero);
    __m128i _b = __lsx_vilvl_w(v0, _zero);
    return (__m256)__lasx_xvinsgr2vr_w(__lasx_xvpermi_q_w((__m256i)__lsx_vreplgr2vr_w(0), (__m256i)_a, 0x20), ((int*)&_b)[3], 7);
}

static NCNN_FORCEINLINE __m128i float2bfloat_avx(const __m256& v0)
{
    __m256i _ab = (__m256i)v0;
    _ab = (v8i32)__lasx_xvsrli_w((v8i32)_ab, 16);
    __m128i _a = __lasx_xvpickve_w_f(_ab, 0, 0);
    __m128i _b = __lasx_xvpickve_w_f(_ab, 4, 0);
    __m128i _v = __lsx_vpickev_h(_b, _a);
    return _v;
}

static NCNN_FORCEINLINE __m256i float2bfloat_avx(const __m256& v0, const __m256& v1)
{
    __m256i _a = (__m256i)v0;
    __m256i _b = (__m256i)v1;
    _a = (v8i32)__lasx_xvsrli_w((v8i32)_a, 16);
    _b = (v8i32)__lasx_xvsrli_w((v8i32)_b, 16);
    __m256i _v = __lasx_xvpickev_h(_b, _a);
    _v = __lasx_xvpermi_q_d(_v, _v, 0xD8);
    return _v;
}

// DPWSSD (LASX)
static NCNN_FORCEINLINE __m256i _mm256_comp_dpwssd_epi32(const __m256i& src, const __m256i& a, const __m256i& b)
{
    return __lasx_xvdpadd_s_w(src, a, b);
}

// DPBUSSD for unsigned byte dot product (LASX)
static NCNN_FORCEINLINE __m256i _mm256_comp_dpbusd_epi32(const __m256i& src, const __m256i& a, const __m256i& b)
{
    __m256i a_lo = __lasx_xvilvr_b(a, __lasx_xvreplgr2vr_w(0));
    __m256i a_hi = __lasx_xvilvl_b(a, __lasx_xvreplgr2vr_w(0));
    __m256i b_lo = __lasx_xvilvr_b(b, __lasx_xvreplgr2vr_w(0));
    __m256i b_hi = __lasx_xvilvl_b(b, __lasx_xvreplgr2vr_w(0));
    __m256i prod_lo = __lasx_xvmul_h(a_lo, b_lo);
    __m256i prod_hi = __lasx_xvmul_h(a_hi, b_hi);
    __m256i sum = __lasx_xvadd_w(prod_lo, prod_hi);
    return __lasx_xvadd_w(sum, src);
}

// DPWSSDS - signed saturated version (LASX)
static NCNN_FORCEINLINE __m256i _mm256_comp_dpwssds_epi32(const __m256i& src, const __m256i& a, const __m256i& b)
{
    __m256i prod = __lasx_xvdpadd_s_w(src, a, b);
    return __lasx_xvsat_s_w(prod, 31);
}

// DPBUSSDS - unsigned saturated version (LASX)
static NCNN_FORCEINLINE __m256i _mm256_comp_dpbusds_epi32(const __m256i& src, const __m256i& a, const __m256i& b)
{
    __m256i dp = _mm256_comp_dpbusd_epi32(src, a, b);
    return __lasx_xvsat_s_w(dp, 31);
}

// cvtepi32 helpers (LASX)
static NCNN_FORCEINLINE __m128i _mm_comp_cvtepi32_epi16(const __m128i& a)
{
    __m128i _si = __lsx_vreplgr2vr_h(0x0100);
    return __lsx_vshuf_b(_si, a, __lsx_vreplgr2vr_w(0x0E0602080A0C00));
}

static NCNN_FORCEINLINE __m128i _mm256_comp_cvtepi32_epi16(const __m256i& a)
{
    __m128i _si = __lsx_vreplgr2vr_h(0x0100);
    __m256i _t = __lasx_xvshuf_b(__lasx_xvreplgr2vr_w(0x01000100), a, __lasx_xvreplgr2vr_w(0x0E0602080A0C00));
    _t = __lasx_xvpermi_q_d(_t, _t, 0xD8);
    return __lasx_xvpickve_w_f(_t, 0, 0);
}

static NCNN_FORCEINLINE __m128i _mm_comp_cvtepi32_epi8(const __m128i& a)
{
    __m128i _si = __lsx_vreplgr2vr_b(0x03);
    return __lsx_vshuf_b(_si, a, __lsx_vreplgr2vr_w(0x0C080400));
}

static NCNN_FORCEINLINE __m128i _mm256_comp_cvtepi32_epi8(const __m256i& a)
{
    __m128i _si = __lsx_vreplgr2vr_b(0x03);
    __m256i _t = __lasx_xvshuf_b(__lasx_xvreplgr2vr_w(0x03030303), a, __lasx_xvreplgr2vr_w(0x0C080400));
    _t = __lasx_xvpermi_q_d(_t, _t, 0xD8);
    return __lsx_vshuf_b(_si, __lasx_xvpickve_w_f(_t, 0, 0), __lsx_vreplgr2vr_w(0x0F0B07030F0B0703));
}

// fast integer division (LASX)
class FastDivider_epu32_256
{
public:
    NCNN_FORCEINLINE FastDivider_epu32_256(unsigned int d)
    {
        unsigned int m, sh1, sh2;
        if (d == 1)
        {
            m = 1;
            sh1 = 0;
            sh2 = 0;
        }
        else
        {
            uint32_t sh = portable_ceil_log2(d);
            uint32_t m0 = sh == 32 ? 0 : 1 << sh;
            m = 1 + uint32_t((uint64_t(m0 - d) << 32) / d);
            sh1 = 1;
            sh2 = sh - 1;
        }
        _multiplier = __lasx_xvreplgr2vr_w(m);
        _shift1 = __lasx_xvreplgr2vr_w(sh1);
        _shift2 = __lasx_xvreplgr2vr_w(sh2);
    }

    NCNN_FORCEINLINE __m256i _mm256_comp_div_epu32(const __m256i& x) const
    {
        __m256i xm_low = __lasx_xvsrli_w(__lasx_xvdpadd_s_w(__lasx_xvreplgr2vr_w(0), x, _multiplier), 32);
        __m256i xm = __lasx_xvor_v(xm_low, __lasx_xvand_v(__lasx_xvsrli_w(x, 32), __lasx_xvreplgr2vr_w(0xFFFF)));
        return __lasx_xvsrli_w(__lasx_xvadd_w(xm, __lasx_xvsrli_w(__lasx_xvsub_w(x, xm), _shift1)), (__m256i)_shift2);
    }

protected:
    static int portable_ceil_log2(int d)
    {
        return 32 - __builtin_clz(d - 1);
    }

protected:
    __m256i _multiplier;
    __m256i _shift1;
    __m256i _shift2;
};
#endif // __loongarch_asx

#endif // LOONGARCH_USABILITY_H
