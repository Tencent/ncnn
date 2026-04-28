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

#if __loongarch_sx
#define _LSX_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif // __loongarch_sx

// GCC 15+ removed vilvr/vilvl old naming; vilvr → vilvl, old vilvl → vilvh
#if __loongarch_sx
#ifndef __lsx_vilvr_b
#define __lsx_vilvr_b __lsx_vilvl_b
#endif
#ifndef __lsx_vilvr_h
#define __lsx_vilvr_h __lsx_vilvl_h
#endif
#ifndef __lsx_vilvr_w
#define __lsx_vilvr_w __lsx_vilvl_w
#endif
#ifndef __lsx_vilvr_d
#define __lsx_vilvr_d __lsx_vilvl_d
#endif
#if __loongarch_asx
#ifndef __lasx_xvilvr_b
#define __lasx_xvilvr_b __lasx_xvilvl_b
#endif
#ifndef __lasx_xvilvr_h
#define __lasx_xvilvr_h __lasx_xvilvl_h
#endif
#ifndef __lasx_xvilvr_w
#define __lasx_xvilvr_w __lasx_xvilvl_w
#endif
#ifndef __lasx_xvilvr_d
#define __lasx_xvilvr_d __lasx_xvilvl_d
#endif
#endif // __loongarch_asx
#endif // __loongarch_sx

// Compat helpers for removed or renamed intrinsics
#if __loongarch_sx
static NCNN_FORCEINLINE __m128i __lsx_extract_lo128(__m128i v)
{
    return v;
}
static NCNN_FORCEINLINE __m128 __lsx_extract_lo128f(__m128 v)
{
    return v;
}
#if __loongarch_asx
static NCNN_FORCEINLINE __m256 __ncnn_lsx_to_lasx(__m128 v)
{
    __m256 dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
static NCNN_FORCEINLINE __m256d __ncnn_lsx_to_lasx(__m128d v)
{
    __m256d dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
static NCNN_FORCEINLINE __m256i __ncnn_lsx_to_lasx(__m128i v)
{
    __m256i dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
static NCNN_FORCEINLINE __m128 __ncnn_lasx_to_lsx(__m256 v)
{
    __m128 dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
static NCNN_FORCEINLINE __m128d __ncnn_lasx_to_lsx(__m256d v)
{
    __m128d dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
static NCNN_FORCEINLINE __m128i __ncnn_lasx_to_lsx(__m256i v)
{
    __m128i dest;
    __asm__(""
            : "=f"(dest)
            : "0"(v));
    return dest;
}
#define __lsx_to_lasx(v)  __ncnn_lsx_to_lasx((v))
#define __lasx_to_lsx(v)  __ncnn_lasx_to_lsx((v))
#define __lasx_xvsext_h_b __lasx_vext2xv_h_b
#define __lasx_xvsext_w_h __lasx_vext2xv_w_h
#define __lasx_xvsext_d_w __lasx_vext2xv_d_w

static NCNN_FORCEINLINE __m128i __lasx_extract_lo128(__m256i v)
{
    return __lasx_to_lsx(v);
}
static NCNN_FORCEINLINE __m128i __lasx_extract_hi128(__m256i v)
{
    return __lasx_to_lsx(__lasx_xvpermi_q(v, v, _LSX_SHUFFLE(0, 1, 0, 1)));
}
#endif // __loongarch_asx
#endif // __loongarch_sx

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
    __m128 hi64 = (__m128)__lsx_vbsrl_v((__m128i)_v, 8);
    __m128 sum64 = __lsx_vfadd_s(hi64, _v);
    __m128 hi32 = (__m128)__lsx_vbsrl_v((__m128i)sum64, 4);
    __m128 sum32 = __lsx_vfadd_s(sum64, hi32);
    float result;
    *(int*)&result = __lsx_vpickve2gr_w((__m128i)sum32, 0);
    return result;
}

static NCNN_FORCEINLINE int __lsx_reduce_add_w(__m128i _v)
{
    __m128i hi64 = __lsx_vbsrl_v(_v, 8);
    __m128i sum64 = __lsx_vadd_w(hi64, _v);
    __m128i hi32 = __lsx_vbsrl_v(sum64, 4);
    __m128i sum32 = __lsx_vadd_w(sum64, hi32);
    return __lsx_vpickve2gr_w(sum32, 0);
}

static NCNN_FORCEINLINE float __lsx_reduce_fmax_s(__m128 _v)
{
    __m128 hi64 = (__m128)__lsx_vbsrl_v((__m128i)_v, 8);
    __m128 max64 = __lsx_vfmax_s(hi64, _v);
    __m128 hi32 = (__m128)__lsx_vbsrl_v((__m128i)max64, 4);
    __m128 max32 = __lsx_vfmax_s(max64, hi32);
    float result;
    *(int*)&result = __lsx_vpickve2gr_w((__m128i)max32, 0);
    return result;
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
    __m256i _vi = (__m256i)_v;
    __m128 lo = (__m128) * (__m128i*)&_vi;
    __m256i _hi256 = __lasx_xvpermi_q(_vi, _vi, _LSX_SHUFFLE(0, 1, 0, 1));
    __m128 hi = (__m128) * (__m128i*)&_hi256;
    __m128 sum = __lsx_vfadd_s(lo, hi);
    return __lsx_reduce_fadd_s(sum);
}

static NCNN_FORCEINLINE int __lasx_reduce_add_w(__m256i _v)
{
    __m128i lo = *(__m128i*)&_v;
    __m256i _hi256 = __lasx_xvpermi_q(_v, _v, _LSX_SHUFFLE(0, 1, 0, 1));
    __m128i hi = *(__m128i*)&_hi256;
    __m128i sum = __lsx_vadd_w(lo, hi);
    return __lsx_reduce_add_w(sum);
}

static NCNN_FORCEINLINE float __lasx_reduce_fmax_s(__m256 _v)
{
    __m256i _vi = (__m256i)_v;
    __m128 lo = (__m128) * (__m128i*)&_vi;
    __m256i _hi256 = __lasx_xvpermi_q(_vi, _vi, _LSX_SHUFFLE(0, 1, 0, 1));
    __m128 hi = (__m128) * (__m128i*)&_hi256;
    __m128 maxv = __lsx_vfmax_s(lo, hi);
    return __lsx_reduce_fmax_s(maxv);
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
static NCNN_FORCEINLINE __m128i round_lsx(__m128 _v)
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
    __m128i _v32 = round_lsx(_v);

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
    __m128i _v32 = round_lsx(_v);

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
static NCNN_FORCEINLINE __m256i round_lasx(__m256 _v)
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
    __m256i _v32 = round_lasx(_v);

    __m256i _v32_16 = __lasx_xvsat_w(_v32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_v32_16, _v32_16);
    _v16 = __lasx_xvmax_h(_v16, __lasx_xvreplgr2vr_h(-127));
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);
    _v8 = __lsx_to_lasx(__lsx_vilvl_w(__lasx_extract_hi128(_v8), __lasx_extract_lo128(_v8)));

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
    __m256i _v32 = round_lasx(_v);

    __m256i _v32_16 = __lasx_xvsat_w(_v32, 15);
    __m256i _v16 = __lasx_xvpickev_h(_v32_16, _v32_16);
    _v16 = __lasx_xvmaxi_h(_v16, 0);
    __m256i _v16_8 = __lasx_xvsat_h(_v16, 7);
    __m256i _v8 = __lasx_xvpickev_b(_v16_8, _v16_8);
    _v8 = __lsx_to_lasx(__lsx_vilvl_w(__lasx_extract_hi128(_v8), __lasx_extract_lo128(_v8)));

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
    _v8 = __lsx_to_lasx(__lsx_vilvl_w(__lasx_extract_hi128(_v8), __lasx_extract_lo128(_v8)));

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
#endif // __loongarch_asx

#if __loongarch_sx
// transpose4x4_epi32 - transpose 4x4 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose4x4_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_w(_r3, _r2);

    _r0 = __lsx_vilvr_d(_tmp2, _tmp0);
    _r1 = __lsx_vilvh_d(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_d(_tmp3, _tmp1);
    _r3 = __lsx_vilvh_d(_tmp3, _tmp1);
}

// transpose4x4_ps - transpose 4x4 block of float (LSX)
static NCNN_FORCEINLINE void transpose4x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r1 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp3, (__m128i)_tmp1);
    _r3 = (__m128)__lsx_vilvh_d((__m128i)_tmp3, (__m128i)_tmp1);
}

// transpose4x8_epi32 - transpose 4x8 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose4x8_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_w(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_w(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvh_w(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_w(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvh_w(_r7, _r6);

    _r0 = __lsx_vilvr_d(_tmp2, _tmp0);
    _r1 = __lsx_vilvr_d(_tmp6, _tmp4);
    _r2 = __lsx_vilvh_d(_tmp2, _tmp0);
    _r3 = __lsx_vilvh_d(_tmp6, _tmp4);
    _r4 = __lsx_vilvr_d(_tmp3, _tmp1);
    _r5 = __lsx_vilvr_d(_tmp7, _tmp5);
    _r6 = __lsx_vilvh_d(_tmp3, _tmp1);
    _r7 = __lsx_vilvh_d(_tmp7, _tmp5);
}

// transpose8x8_epi16 - transpose 8x8 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose8x8_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_h(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_h(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvh_h(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_h(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvh_h(_r7, _r6);

    __m128i _tmp8 = __lsx_vilvr_w(_tmp2, _tmp0);
    __m128i _tmp9 = __lsx_vilvh_w(_tmp2, _tmp0);
    __m128i _tmpa = __lsx_vilvr_w(_tmp3, _tmp1);
    __m128i _tmpb = __lsx_vilvh_w(_tmp3, _tmp1);
    __m128i _tmpc = __lsx_vilvr_w(_tmp6, _tmp4);
    __m128i _tmpd = __lsx_vilvh_w(_tmp6, _tmp4);
    __m128i _tmpe = __lsx_vilvr_w(_tmp7, _tmp5);
    __m128i _tmpf = __lsx_vilvh_w(_tmp7, _tmp5);

    _r0 = __lsx_vilvr_d(_tmp8, _tmpc);
    _r1 = __lsx_vilvh_d(_tmp8, _tmpc);
    _r2 = __lsx_vilvr_d(_tmp9, _tmpd);
    _r3 = __lsx_vilvh_d(_tmp9, _tmpd);
    _r4 = __lsx_vilvr_d(_tmpa, _tmpe);
    _r5 = __lsx_vilvh_d(_tmpa, _tmpe);
    _r6 = __lsx_vilvr_d(_tmpb, _tmpf);
    _r7 = __lsx_vilvh_d(_tmpb, _tmpf);
}

// transpose8x4_epi16 - transpose 8x4 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose8x4_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp2, _tmp0);
    _r1 = __lsx_vilvh_w(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_w(_tmp3, _tmp1);
    _r3 = __lsx_vilvh_w(_tmp3, _tmp1);
}

// transpose8x4_epi8 - transpose 8x4 block of int8 (LSX)
static NCNN_FORCEINLINE void transpose8x4_epi8(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_b(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvr_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp1, _tmp0);
    _r1 = __lsx_vilvh_w(_tmp1, _tmp0);
}

// transpose4x4_epi16 - transpose 4x4 block of int16 (LSX)
static NCNN_FORCEINLINE void transpose4x4_epi16(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_h(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_h(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_h(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_h(_r3, _r2);

    _r0 = __lsx_vilvr_w(_tmp2, _tmp0);
    _r1 = __lsx_vilvh_w(_tmp2, _tmp0);
    _r2 = __lsx_vilvr_w(_tmp3, _tmp1);
    _r3 = __lsx_vilvh_w(_tmp3, _tmp1);
}

// transpose8x4_ps - transpose 8x4 block of float (LSX)
static NCNN_FORCEINLINE void transpose8x4_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r1 = (__m128)__lsx_vilvh_d((__m128i)_tmp2, (__m128i)_tmp0);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp3, (__m128i)_tmp1);
    _r3 = (__m128)__lsx_vilvh_d((__m128i)_tmp3, (__m128i)_tmp1);
}

// transpose8x8_ps - transpose 8x8 block of float (LSX)
static NCNN_FORCEINLINE void transpose8x8_ps(__m128& _r0, __m128& _r1, __m128& _r2, __m128& _r3, __m128& _r4, __m128& _r5, __m128& _r6, __m128& _r7)
{
    __m128 _tmp0 = (__m128)__lsx_vilvr_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
    __m128 _tmp2 = (__m128)__lsx_vilvr_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_r3, (__m128i)_r2);
    __m128 _tmp4 = (__m128)__lsx_vilvr_w((__m128i)_r5, (__m128i)_r4);
    __m128 _tmp5 = (__m128)__lsx_vilvh_w((__m128i)_r5, (__m128i)_r4);
    __m128 _tmp6 = (__m128)__lsx_vilvr_w((__m128i)_r7, (__m128i)_r6);
    __m128 _tmp7 = (__m128)__lsx_vilvh_w((__m128i)_r7, (__m128i)_r6);

    __m128 _tmp8 = (__m128)__lsx_vilvr_w((__m128i)_tmp2, (__m128i)_tmp0);
    __m128 _tmp9 = (__m128)__lsx_vilvh_w((__m128i)_tmp2, (__m128i)_tmp0);
    __m128 _tmpa = (__m128)__lsx_vilvr_w((__m128i)_tmp3, (__m128i)_tmp1);
    __m128 _tmpb = (__m128)__lsx_vilvh_w((__m128i)_tmp3, (__m128i)_tmp1);
    __m128 _tmpc = (__m128)__lsx_vilvr_w((__m128i)_tmp6, (__m128i)_tmp4);
    __m128 _tmpd = (__m128)__lsx_vilvh_w((__m128i)_tmp6, (__m128i)_tmp4);
    __m128 _tmpe = (__m128)__lsx_vilvr_w((__m128i)_tmp7, (__m128i)_tmp5);
    __m128 _tmpf = (__m128)__lsx_vilvh_w((__m128i)_tmp7, (__m128i)_tmp5);

    _r0 = (__m128)__lsx_vilvr_d((__m128i)_tmp8, (__m128i)_tmpc);
    _r1 = (__m128)__lsx_vilvh_d((__m128i)_tmp8, (__m128i)_tmpc);
    _r2 = (__m128)__lsx_vilvr_d((__m128i)_tmp9, (__m128i)_tmpd);
    _r3 = (__m128)__lsx_vilvh_d((__m128i)_tmp9, (__m128i)_tmpd);
    _r4 = (__m128)__lsx_vilvr_d((__m128i)_tmpa, (__m128i)_tmpe);
    _r5 = (__m128)__lsx_vilvh_d((__m128i)_tmpa, (__m128i)_tmpe);
    _r6 = (__m128)__lsx_vilvr_d((__m128i)_tmpb, (__m128i)_tmpf);
    _r7 = (__m128)__lsx_vilvh_d((__m128i)_tmpb, (__m128i)_tmpf);
}

// transpose8x8_epi32 - transpose 8x8 block of int32 (LSX)
static NCNN_FORCEINLINE void transpose8x8_epi32(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3, __m128i& _r4, __m128i& _r5, __m128i& _r6, __m128i& _r7)
{
    __m128i _tmp0 = __lsx_vilvr_w(_r1, _r0);
    __m128i _tmp1 = __lsx_vilvh_w(_r1, _r0);
    __m128i _tmp2 = __lsx_vilvr_w(_r3, _r2);
    __m128i _tmp3 = __lsx_vilvh_w(_r3, _r2);
    __m128i _tmp4 = __lsx_vilvr_w(_r5, _r4);
    __m128i _tmp5 = __lsx_vilvh_w(_r5, _r4);
    __m128i _tmp6 = __lsx_vilvr_w(_r7, _r6);
    __m128i _tmp7 = __lsx_vilvh_w(_r7, _r6);

    __m128i _tmp8 = __lsx_vilvr_w(_tmp2, _tmp0);
    __m128i _tmp9 = __lsx_vilvh_w(_tmp2, _tmp0);
    __m128i _tmpa = __lsx_vilvr_w(_tmp3, _tmp1);
    __m128i _tmpb = __lsx_vilvh_w(_tmp3, _tmp1);
    __m128i _tmpc = __lsx_vilvr_w(_tmp6, _tmp4);
    __m128i _tmpd = __lsx_vilvh_w(_tmp6, _tmp4);
    __m128i _tmpe = __lsx_vilvr_w(_tmp7, _tmp5);
    __m128i _tmpf = __lsx_vilvh_w(_tmp7, _tmp5);

    _r0 = __lsx_vilvr_d(_tmp8, _tmpc);
    _r1 = __lsx_vilvh_d(_tmp8, _tmpc);
    _r2 = __lsx_vilvr_d(_tmp9, _tmpd);
    _r3 = __lsx_vilvh_d(_tmp9, _tmpd);
    _r4 = __lsx_vilvr_d(_tmpa, _tmpe);
    _r5 = __lsx_vilvh_d(_tmpa, _tmpe);
    _r6 = __lsx_vilvr_d(_tmpb, _tmpf);
    _r7 = __lsx_vilvh_d(_tmpb, _tmpf);
}

// transpose16x4_epi8 - transpose 16x4 block of int8 (LSX)
static NCNN_FORCEINLINE void transpose16x4_epi8(__m128i& _r0, __m128i& _r1, __m128i& _r2, __m128i& _r3)
{
    __m128i _tmp0 = __lsx_vilvr_b(_r0, _r1);
    __m128i _tmp1 = __lsx_vilvh_b(_r0, _r1);
    __m128i _tmp2 = __lsx_vilvr_b(_r2, _r3);
    __m128i _tmp3 = __lsx_vilvh_b(_r2, _r3);

    _r0 = __lsx_vilvr_h(_tmp0, _tmp2);
    _r1 = __lsx_vilvh_h(_tmp0, _tmp2);
    _r2 = __lsx_vilvr_h(_tmp1, _tmp3);
    _r3 = __lsx_vilvh_h(_tmp1, _tmp3);
}

// BF16 conversion utilities (LSX)
static NCNN_FORCEINLINE __m128 bfloat2float_lsx(const __m128i& v0)
{
    // BF16 in low 64 bits, zero-extend to 32-bit then shift left 16
    __m128i _zero = __lsx_vreplgr2vr_w(0);
    __m128i _a = __lsx_vilvl_h(v0, _zero);
    return (__m128)_a;
}

static NCNN_FORCEINLINE __m128 bfloat2float_lsx(const __m128i* ptr)
{
    __m128i v0 = __lsx_vld(ptr, 0);
    return bfloat2float_lsx(v0);
}

// Load 4 bf16 values from pointer using safe 64-bit load (no 16-byte overread)
static NCNN_FORCEINLINE __m128 bfloat2float_lsx(const unsigned short* ptr)
{
    int64_t v;
    memcpy(&v, ptr, 8);
    __m128i _zero = __lsx_vreplgr2vr_w(0);
    __m128i _raw = __lsx_vreplgr2vr_d(v);
    __m128i _a = __lsx_vilvl_h(_raw, _zero);
    return (__m128)_a;
}

static NCNN_FORCEINLINE __m128i float2bfloat_lsx(const __m128& v)
{
    __m128i _a = (__m128i)v;
    _a = __lsx_vsrli_w(_a, 16);
    __m128i _v = __lsx_vpickev_h(__lsx_vreplgr2vr_w(0), _a);
    return _v;
}

static NCNN_FORCEINLINE __m128i float2bfloat_lsx(const __m128& v0, const __m128& v1)
{
    __m128i _a = (__m128i)v0;
    __m128i _b = (__m128i)v1;
    _a = __lsx_vsrli_w(_a, 16);
    _b = __lsx_vsrli_w(_b, 16);
    __m128i _v = __lsx_vpickev_h(_b, _a);
    return _v;
}

// HorizontalSums for 4 accumulators (LSX)
// Compute horizontal sum of each __m128 vector, pack 4 results into __m128
static NCNN_FORCEINLINE __m128 HorizontalSums(__m128& v0, __m128& v1, __m128& v2, __m128& v3)
{
    transpose4x4_ps(v0, v1, v2, v3);
    v0 = __lsx_vfadd_s(v0, v1);
    v0 = __lsx_vfadd_s(v0, v2);
    v0 = __lsx_vfadd_s(v0, v3);
    return v0;
}

// HorizontalSums for 8 accumulators (LSX)
// Compute horizontal sum of each __m128 vector, pack first 4 results into __m128
// Note: only the first 4 sums fit in __m128. The last 4 sums are computed but only
// the first group is returned. Use two calls for 8 accumulators if needed.
static NCNN_FORCEINLINE void HorizontalSums(__m128& v0, __m128& v1, __m128& v2, __m128& v3, __m128& v4, __m128& v5, __m128& v6, __m128& v7, __m128& sum03, __m128& sum47)
{
    transpose4x4_ps(v0, v1, v2, v3);
    sum03 = __lsx_vfadd_s(v0, v1);
    sum03 = __lsx_vfadd_s(sum03, v2);
    sum03 = __lsx_vfadd_s(sum03, v3);

    transpose4x4_ps(v4, v5, v6, v7);
    sum47 = __lsx_vfadd_s(v4, v5);
    sum47 = __lsx_vfadd_s(sum47, v6);
    sum47 = __lsx_vfadd_s(sum47, v7);
}
#endif // __loongarch_sx

#if __loongarch_asx
// LASX 256-bit variants

// combine functions
static NCNN_FORCEINLINE __m256 combine4x2_ps(const __m128& a, const __m128& b)
{
    // combine two 128-bit into one 256-bit: low=a, high=b
    __m256i _r = __lasx_xvpermi_q((__m256i)(__m256)__lasx_xvreplfr2vr_s(0.f), (__m256i)(__m256)__lasx_xvreplfr2vr_s(0.f), _LSX_SHUFFLE(0, 2, 0, 0));
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&a)[0], 0);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&a)[1], 1);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&a)[2], 2);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&a)[3], 3);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&b)[0], 4);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&b)[1], 5);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&b)[2], 6);
    _r = __lasx_xvinsgr2vr_w(_r, ((const int*)&b)[3], 7);
    return (__m256)_r;
}

static NCNN_FORCEINLINE __m256i combine4x2_epi32(const __m128i& a, const __m128i& b)
{
    __m256i _r = __lasx_xvreplgr2vr_w(0);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(a, 0), 0);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(a, 1), 1);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(a, 2), 2);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(a, 3), 3);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(b, 0), 4);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(b, 1), 5);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(b, 2), 6);
    _r = __lasx_xvinsgr2vr_w(_r, __lsx_vpickve2gr_w(b, 3), 7);
    return _r;
}

// transpose8x12_ps - transpose 8x12 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x12_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7,
        __m256& _r8, __m256& _r9, __m256& _ra, __m256& _rb)
{
    // Step 1: 32-bit word interleave
    __m256 _tmp0 = (__m256)__lasx_xvilvl_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp2 = (__m256)__lasx_xvilvl_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp4 = (__m256)__lasx_xvilvl_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp5 = (__m256)__lasx_xvilvh_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp6 = (__m256)__lasx_xvilvl_w((__m256i)_r7, (__m256i)_r6);
    __m256 _tmp7 = (__m256)__lasx_xvilvh_w((__m256i)_r7, (__m256i)_r6);
    __m256 _tmp8 = (__m256)__lasx_xvilvl_w((__m256i)_r9, (__m256i)_r8);
    __m256 _tmp9 = (__m256)__lasx_xvilvh_w((__m256i)_r9, (__m256i)_r8);
    __m256 _tmpa = (__m256)__lasx_xvilvl_w((__m256i)_rb, (__m256i)_ra);
    __m256 _tmpb = (__m256)__lasx_xvilvh_w((__m256i)_rb, (__m256i)_ra);

    // Step 2: 64-bit doubleword interleave
    __m256 _tmpc = (__m256)__lasx_xvilvl_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmpd = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmpe = (__m256)__lasx_xvilvl_d((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpf = (__m256)__lasx_xvilvh_d((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpg = (__m256)__lasx_xvilvl_d((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmph = (__m256)__lasx_xvilvh_d((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmpi = (__m256)__lasx_xvilvl_d((__m256i)_tmp7, (__m256i)_tmp5);
    __m256 _tmpj = (__m256)__lasx_xvilvh_d((__m256i)_tmp7, (__m256i)_tmp5);
    __m256 _tmpk = (__m256)__lasx_xvilvl_d((__m256i)_tmpa, (__m256i)_tmp8);
    __m256 _tmpl = (__m256)__lasx_xvilvh_d((__m256i)_tmpa, (__m256i)_tmp8);
    __m256 _tmpm = (__m256)__lasx_xvilvl_d((__m256i)_tmpb, (__m256i)_tmp9);
    __m256 _tmpn = (__m256)__lasx_xvilvh_d((__m256i)_tmpb, (__m256i)_tmp9);

    // Step 3: cross-lane 128-bit permute
    _r0 = (__m256)__lasx_xvpermi_q((__m256i)_tmpg, (__m256i)_tmpc, _LSX_SHUFFLE(0, 2, 0, 0));
    _r1 = (__m256)__lasx_xvpermi_q((__m256i)_tmpd, (__m256i)_tmpk, _LSX_SHUFFLE(0, 2, 0, 0));
    _r2 = (__m256)__lasx_xvpermi_q((__m256i)_tmpl, (__m256i)_tmph, _LSX_SHUFFLE(0, 2, 0, 0));
    _r3 = (__m256)__lasx_xvpermi_q((__m256i)_tmpi, (__m256i)_tmpe, _LSX_SHUFFLE(0, 2, 0, 0));
    _r4 = (__m256)__lasx_xvpermi_q((__m256i)_tmpf, (__m256i)_tmpm, _LSX_SHUFFLE(0, 2, 0, 0));
    _r5 = (__m256)__lasx_xvpermi_q((__m256i)_tmpn, (__m256i)_tmpj, _LSX_SHUFFLE(0, 2, 0, 0));
    _r6 = (__m256)__lasx_xvpermi_q((__m256i)_tmpg, (__m256i)_tmpc, _LSX_SHUFFLE(0, 3, 0, 1));
    _r7 = (__m256)__lasx_xvpermi_q((__m256i)_tmpd, (__m256i)_tmpk, _LSX_SHUFFLE(0, 3, 0, 1));
    _r8 = (__m256)__lasx_xvpermi_q((__m256i)_tmpl, (__m256i)_tmph, _LSX_SHUFFLE(0, 3, 0, 1));
    _r9 = (__m256)__lasx_xvpermi_q((__m256i)_tmpi, (__m256i)_tmpe, _LSX_SHUFFLE(0, 3, 0, 1));
    _ra = (__m256)__lasx_xvpermi_q((__m256i)_tmpf, (__m256i)_tmpm, _LSX_SHUFFLE(0, 3, 0, 1));
    _rb = (__m256)__lasx_xvpermi_q((__m256i)_tmpn, (__m256i)_tmpj, _LSX_SHUFFLE(0, 3, 0, 1));
}

// transpose8x8_ps - transpose 8x8 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x8_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7)
{
    // step 1: 32-bit word interleave
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp2 = (__m256)__lasx_xvilvr_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp4 = (__m256)__lasx_xvilvr_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp5 = (__m256)__lasx_xvilvh_w((__m256i)_r5, (__m256i)_r4);
    __m256 _tmp6 = (__m256)__lasx_xvilvr_w((__m256i)_r7, (__m256i)_r6);
    __m256 _tmp7 = (__m256)__lasx_xvilvh_w((__m256i)_r7, (__m256i)_r6);

    // step 2: 64-bit doubleword interleave
    __m256 _tmp8 = (__m256)__lasx_xvilvr_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp9 = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmpa = (__m256)__lasx_xvilvr_d((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpb = (__m256)__lasx_xvilvh_d((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmpc = (__m256)__lasx_xvilvr_d((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmpd = (__m256)__lasx_xvilvh_d((__m256i)_tmp6, (__m256i)_tmp4);
    __m256 _tmpe = (__m256)__lasx_xvilvr_d((__m256i)_tmp7, (__m256i)_tmp5);
    __m256 _tmpf = (__m256)__lasx_xvilvh_d((__m256i)_tmp7, (__m256i)_tmp5);

    // step 3: cross-lane 128-bit permute
    _r0 = (__m256)__lasx_xvpermi_q((__m256i)_tmpc, (__m256i)_tmp8, _LSX_SHUFFLE(0, 2, 0, 0));
    _r1 = (__m256)__lasx_xvpermi_q((__m256i)_tmpd, (__m256i)_tmp9, _LSX_SHUFFLE(0, 2, 0, 0));
    _r2 = (__m256)__lasx_xvpermi_q((__m256i)_tmpe, (__m256i)_tmpa, _LSX_SHUFFLE(0, 2, 0, 0));
    _r3 = (__m256)__lasx_xvpermi_q((__m256i)_tmpf, (__m256i)_tmpb, _LSX_SHUFFLE(0, 2, 0, 0));
    _r4 = (__m256)__lasx_xvpermi_q((__m256i)_tmpc, (__m256i)_tmp8, _LSX_SHUFFLE(0, 3, 0, 1));
    _r5 = (__m256)__lasx_xvpermi_q((__m256i)_tmpd, (__m256i)_tmp9, _LSX_SHUFFLE(0, 3, 0, 1));
    _r6 = (__m256)__lasx_xvpermi_q((__m256i)_tmpe, (__m256i)_tmpa, _LSX_SHUFFLE(0, 3, 0, 1));
    _r7 = (__m256)__lasx_xvpermi_q((__m256i)_tmpf, (__m256i)_tmpb, _LSX_SHUFFLE(0, 3, 0, 1));
}

// transpose8x4_ps - transpose 8x4 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x4_ps(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3)
{
    // Step 1: lane-wise word interleave
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp2 = (__m256)__lasx_xvilvr_w((__m256i)_r3, (__m256i)_r2);
    __m256 _tmp3 = (__m256)__lasx_xvilvh_w((__m256i)_r3, (__m256i)_r2);

    // Step 2: lane-wise doubleword interleave
    __m256 _tmp4 = (__m256)__lasx_xvilvr_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp5 = (__m256)__lasx_xvilvh_d((__m256i)_tmp2, (__m256i)_tmp0);
    __m256 _tmp6 = (__m256)__lasx_xvilvr_d((__m256i)_tmp3, (__m256i)_tmp1);
    __m256 _tmp7 = (__m256)__lasx_xvilvh_d((__m256i)_tmp3, (__m256i)_tmp1);

    // Step 3: cross-lane 128-bit permute
    _r0 = (__m256)__lasx_xvpermi_q((__m256i)_tmp5, (__m256i)_tmp4, _LSX_SHUFFLE(0, 2, 0, 0));
    _r1 = (__m256)__lasx_xvpermi_q((__m256i)_tmp7, (__m256i)_tmp6, _LSX_SHUFFLE(0, 2, 0, 0));
    _r2 = (__m256)__lasx_xvpermi_q((__m256i)_tmp5, (__m256i)_tmp4, _LSX_SHUFFLE(0, 3, 0, 1));
    _r3 = (__m256)__lasx_xvpermi_q((__m256i)_tmp7, (__m256i)_tmp6, _LSX_SHUFFLE(0, 3, 0, 1));
}

// transpose8x2_ps - transpose 8x2 block of float (LASX)
static NCNN_FORCEINLINE void transpose8x2_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = (__m256)__lasx_xvilvr_w((__m256i)_r1, (__m256i)_r0);
    __m256 _tmp1 = (__m256)__lasx_xvilvh_w((__m256i)_r1, (__m256i)_r0);

    _r0 = (__m256)__lasx_xvpermi_q((__m256i)_tmp1, (__m256i)_tmp0, _LSX_SHUFFLE(0, 2, 0, 0));
    _r1 = (__m256)__lasx_xvpermi_q((__m256i)_tmp1, (__m256i)_tmp0, _LSX_SHUFFLE(0, 3, 0, 1));
}

// transpose2x8_ps - transpose 2x8 block of float (LASX)
static NCNN_FORCEINLINE void transpose2x8_ps(__m256& _r0, __m256& _r1)
{
    __m256 _tmp0 = (__m256)__lasx_xvpermi_q((__m256i)_r1, (__m256i)_r0, _LSX_SHUFFLE(0, 2, 0, 0));
    __m256 _tmp1 = (__m256)__lasx_xvpermi_q((__m256i)_r1, (__m256i)_r0, _LSX_SHUFFLE(0, 3, 0, 1));

    _r0 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp0, _LSX_SHUFFLE(2, 0, 2, 0));
    _r1 = (__m256)__lasx_xvshuf4i_w((__m256i)_tmp1, _LSX_SHUFFLE(3, 1, 3, 1));
}

// transpose8x8_epi32 - transpose 8x8 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x8_epi32(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_w(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_w(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvh_w(_r3, _r2);
    __m256i _tmp4 = __lasx_xvilvr_w(_r5, _r4);
    __m256i _tmp5 = __lasx_xvilvh_w(_r5, _r4);
    __m256i _tmp6 = __lasx_xvilvr_w(_r7, _r6);
    __m256i _tmp7 = __lasx_xvilvh_w(_r7, _r6);

    __m256i _tmp8 = __lasx_xvilvr_w(_tmp2, _tmp0);
    __m256i _tmp9 = __lasx_xvilvh_w(_tmp2, _tmp0);
    __m256i _tmpa = __lasx_xvilvr_w(_tmp3, _tmp1);
    __m256i _tmpb = __lasx_xvilvh_w(_tmp3, _tmp1);
    __m256i _tmpc = __lasx_xvilvr_w(_tmp6, _tmp4);
    __m256i _tmpd = __lasx_xvilvh_w(_tmp6, _tmp4);
    __m256i _tmpe = __lasx_xvilvr_w(_tmp7, _tmp5);
    __m256i _tmpf = __lasx_xvilvh_w(_tmp7, _tmp5);

    _r0 = __lasx_xvilvr_d(_tmp8, _tmpc);
    _r1 = __lasx_xvilvh_d(_tmp8, _tmpc);
    _r2 = __lasx_xvilvr_d(_tmp9, _tmpd);
    _r3 = __lasx_xvilvh_d(_tmp9, _tmpd);
    _r4 = __lasx_xvilvr_d(_tmpa, _tmpe);
    _r5 = __lasx_xvilvh_d(_tmpa, _tmpe);
    _r6 = __lasx_xvilvr_d(_tmpb, _tmpf);
    _r7 = __lasx_xvilvh_d(_tmpb, _tmpf);
}

// transpose8x4_epi32 - transpose 8x4 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x4_epi32(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_w(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_w(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvh_w(_r3, _r2);

    __m256i _tmp4 = __lasx_xvilvr_w(_tmp2, _tmp0);
    __m256i _tmp5 = __lasx_xvilvh_w(_tmp2, _tmp0);
    __m256i _tmp6 = __lasx_xvilvr_w(_tmp3, _tmp1);
    __m256i _tmp7 = __lasx_xvilvh_w(_tmp3, _tmp1);

    _r0 = __lasx_xvilvr_d(_tmp4, _tmp5);
    _r1 = __lasx_xvilvh_d(_tmp4, _tmp5);
    _r2 = __lasx_xvilvr_d(_tmp6, _tmp7);
    _r3 = __lasx_xvilvh_d(_tmp6, _tmp7);
}

// transpose8x2_epi32 - transpose 8x2 block of int32 (LASX)
static NCNN_FORCEINLINE void transpose8x2_epi32(__m256i& _r0, __m256i& _r1)
{
    __m256i _tmp0 = __lasx_xvilvr_w(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_w(_r1, _r0);

    _r0 = __lasx_xvilvr_d(_tmp0, _tmp1);
    _r1 = __lasx_xvilvh_d(_tmp0, _tmp1);
}

// transpose16x4_epi16 - transpose 16x4 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x4_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_h(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_h(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvh_h(_r3, _r2);

    __m256i _tmp4 = __lasx_xvilvr_w(_tmp0, _tmp2);
    __m256i _tmp5 = __lasx_xvilvh_w(_tmp0, _tmp2);
    __m256i _tmp6 = __lasx_xvilvr_w(_tmp1, _tmp3);
    __m256i _tmp7 = __lasx_xvilvh_w(_tmp1, _tmp3);

    _r0 = __lasx_xvilvr_d(_tmp4, _tmp5);
    _r1 = __lasx_xvilvh_d(_tmp4, _tmp5);
    _r2 = __lasx_xvilvr_d(_tmp6, _tmp7);
    _r3 = __lasx_xvilvh_d(_tmp6, _tmp7);
}

// transpose16x2_epi16 - transpose 16x2 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x2_epi16(__m256i& _r0, __m256i& _r1)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_h(_r1, _r0);

    _r0 = __lasx_xvilvr_d(_tmp0, _tmp1);
    _r1 = __lasx_xvilvh_d(_tmp0, _tmp1);
}

// transpose16x8_epi16 - transpose 16x8 block of int16 (LASX)
static NCNN_FORCEINLINE void transpose16x8_epi16(__m256i& _r0, __m256i& _r1, __m256i& _r2, __m256i& _r3, __m256i& _r4, __m256i& _r5, __m256i& _r6, __m256i& _r7)
{
    __m256i _tmp0 = __lasx_xvilvr_h(_r1, _r0);
    __m256i _tmp1 = __lasx_xvilvh_h(_r1, _r0);
    __m256i _tmp2 = __lasx_xvilvr_h(_r3, _r2);
    __m256i _tmp3 = __lasx_xvilvh_h(_r3, _r2);
    __m256i _tmp4 = __lasx_xvilvr_h(_r5, _r4);
    __m256i _tmp5 = __lasx_xvilvh_h(_r5, _r4);
    __m256i _tmp6 = __lasx_xvilvr_h(_r7, _r6);
    __m256i _tmp7 = __lasx_xvilvh_h(_r7, _r6);

    __m256i _tmpg = __lasx_xvilvr_w(_tmp0, _tmp2);
    __m256i _tmph = __lasx_xvilvh_w(_tmp0, _tmp2);
    __m256i _tmpi = __lasx_xvilvr_w(_tmp1, _tmp3);
    __m256i _tmpj = __lasx_xvilvh_w(_tmp1, _tmp3);
    __m256i _tmpk = __lasx_xvilvr_w(_tmp4, _tmp6);
    __m256i _tmpl = __lasx_xvilvh_w(_tmp4, _tmp6);
    __m256i _tmpm = __lasx_xvilvr_w(_tmp5, _tmp7);
    __m256i _tmpn = __lasx_xvilvh_w(_tmp5, _tmp7);

    _tmp0 = __lasx_xvilvr_d(_tmpg, _tmpk);
    _tmp1 = __lasx_xvilvh_d(_tmpg, _tmpk);
    _tmp2 = __lasx_xvilvr_d(_tmph, _tmpl);
    _tmp3 = __lasx_xvilvh_d(_tmph, _tmpl);
    _tmp4 = __lasx_xvilvr_d(_tmpi, _tmpm);
    _tmp5 = __lasx_xvilvh_d(_tmpi, _tmpm);
    _tmp6 = __lasx_xvilvr_d(_tmpj, _tmpn);
    _tmp7 = __lasx_xvilvh_d(_tmpj, _tmpn);

    _r0 = __lasx_xvpermi_q(_tmp1, _tmp0, _LSX_SHUFFLE(0, 2, 0, 0));
    _r1 = __lasx_xvpermi_q(_tmp3, _tmp2, _LSX_SHUFFLE(0, 2, 0, 0));
    _r2 = __lasx_xvpermi_q(_tmp5, _tmp4, _LSX_SHUFFLE(0, 2, 0, 0));
    _r3 = __lasx_xvpermi_q(_tmp7, _tmp6, _LSX_SHUFFLE(0, 2, 0, 0));
    _r4 = __lasx_xvpermi_q(_tmp1, _tmp0, _LSX_SHUFFLE(0, 3, 0, 1));
    _r5 = __lasx_xvpermi_q(_tmp3, _tmp2, _LSX_SHUFFLE(0, 3, 0, 1));
    _r6 = __lasx_xvpermi_q(_tmp5, _tmp4, _LSX_SHUFFLE(0, 3, 0, 1));
    _r7 = __lasx_xvpermi_q(_tmp7, _tmp6, _LSX_SHUFFLE(0, 3, 0, 1));
}

// HorizontalSums (LASX)
// Compute horizontal sum of each __m256 vector, pack 8 results into __m256
static NCNN_FORCEINLINE __m256 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3, __m256& v4, __m256& v5, __m256& v6, __m256& v7)
{
    // Reduce 256 to 128 by adding hi and lo halves
    __m128 s0 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v0), (__m128)__lasx_extract_hi128((__m256i)v0));
    __m128 s1 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v1), (__m128)__lasx_extract_hi128((__m256i)v1));
    __m128 s2 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v2), (__m128)__lasx_extract_hi128((__m256i)v2));
    __m128 s3 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v3), (__m128)__lasx_extract_hi128((__m256i)v3));
    __m128 s4 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v4), (__m128)__lasx_extract_hi128((__m256i)v4));
    __m128 s5 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v5), (__m128)__lasx_extract_hi128((__m256i)v5));
    __m128 s6 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v6), (__m128)__lasx_extract_hi128((__m256i)v6));
    __m128 s7 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v7), (__m128)__lasx_extract_hi128((__m256i)v7));

    // Transpose and reduce first group of 4
    transpose4x4_ps(s0, s1, s2, s3);
    s0 = __lsx_vfadd_s(s0, s1);
    s0 = __lsx_vfadd_s(s0, s2);
    s0 = __lsx_vfadd_s(s0, s3);

    // Transpose and reduce second group of 4
    transpose4x4_ps(s4, s5, s6, s7);
    s4 = __lsx_vfadd_s(s4, s5);
    s4 = __lsx_vfadd_s(s4, s6);
    s4 = __lsx_vfadd_s(s4, s7);

    return (__m256)combine4x2_epi32((__m128i)s0, (__m128i)s4);
}

// Compute horizontal sum of each __m256 vector, pack 4 results into __m128
static NCNN_FORCEINLINE __m128 HorizontalSums(__m256& v0, __m256& v1, __m256& v2, __m256& v3)
{
    // Reduce 256 to 128 by adding hi and lo halves
    __m128 s0 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v0), (__m128)__lasx_extract_hi128((__m256i)v0));
    __m128 s1 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v1), (__m128)__lasx_extract_hi128((__m256i)v1));
    __m128 s2 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v2), (__m128)__lasx_extract_hi128((__m256i)v2));
    __m128 s3 = __lsx_vfadd_s((__m128)__lasx_extract_lo128((__m256i)v3), (__m128)__lasx_extract_hi128((__m256i)v3));

    // Transpose and reduce
    transpose4x4_ps(s0, s1, s2, s3);
    s0 = __lsx_vfadd_s(s0, s1);
    s0 = __lsx_vfadd_s(s0, s2);
    s0 = __lsx_vfadd_s(s0, s3);
    return s0;
}

// BF16 conversion (LASX)
static NCNN_FORCEINLINE __m256 bfloat2float_lasx(const __m128i& v0)
{
    // BF16 x8 in v0 (128-bit), expand to FP32 x8 (256-bit)
    __m128i _zero = __lsx_vreplgr2vr_w(0);
    __m128i _lo = __lsx_vilvl_h(v0, _zero);
    __m128i _hi = __lsx_vilvh_h(v0, _zero);
    return (__m256)combine4x2_epi32(_lo, _hi);
}

static NCNN_FORCEINLINE __m256 bfloat2float_lasx(const __m128i* ptr)
{
    __m128i v0 = __lsx_vld(ptr, 0);
    return bfloat2float_lasx(v0);
}

static NCNN_FORCEINLINE __m128i float2bfloat_lasx(const __m256& v0)
{
    __m256i _ab = (__m256i)v0;
    _ab = __lasx_xvsrli_w(_ab, 16);
    __m128i _a = __lasx_extract_lo128(_ab);
    __m128i _b = __lasx_extract_hi128(_ab);
    __m128i _v = __lsx_vpickev_h(_b, _a);
    return _v;
}

static NCNN_FORCEINLINE __m256i float2bfloat_lasx(const __m256& v0, const __m256& v1)
{
    // Convert each 256-bit float vector to 128-bit bf16 separately
    __m128i _v0_bf16 = float2bfloat_lasx(v0);
    __m128i _v1_bf16 = float2bfloat_lasx(v1);
    // Combine: lo128 = v0's 8 bf16, hi128 = v1's 8 bf16
    __m256i _r = (__m256i)__lasx_xvreplgr2vr_d(0);
    _r = __lasx_xvinsgr2vr_d(_r, __lsx_vpickve2gr_d(_v0_bf16, 0), 0);
    _r = __lasx_xvinsgr2vr_d(_r, __lsx_vpickve2gr_d(_v0_bf16, 1), 1);
    _r = __lasx_xvinsgr2vr_d(_r, __lsx_vpickve2gr_d(_v1_bf16, 0), 2);
    _r = __lasx_xvinsgr2vr_d(_r, __lsx_vpickve2gr_d(_v1_bf16, 1), 3);
    return _r;
}
#endif // __loongarch_asx

#endif // LOONGARCH_USABILITY_H
