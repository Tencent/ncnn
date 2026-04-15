// Copyright 2020 Leo <leo@nullptr.com.cn>
// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MIPS_USABILITY_H
#define MIPS_USABILITY_H

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include <stdint.h>

namespace ncnn {

typedef union
{
    int32_t i;
    float f;
} FloatInt;

} // namespace ncnn

#if __mips_msa
/* declare some mips constants with union */
#define _MIPS_FLOAT_CONST(Name, Val) \
    static const ncnn::FloatInt Name = {.f = Val}

/* float type data load instructions */
static NCNN_FORCEINLINE v4f32 __msa_fill_w_f32(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (v4f32)__msa_fill_w(fi_tmpval.i);
}

static NCNN_FORCEINLINE float __msa_reduce_fmax_w(v4f32 _v)
{
    // _v = {f0, f1, f2, f3}
    // swap pairs: {f2, f3, f0, f1}
    v4f32 _s = (v4f32)__msa_shf_w((v4i32)_v, 0x4E);
    _v = __msa_fmax_w(_v, _s); // {max(f0,f2), max(f1,f3), ...}
    // swap within pair: {max(f1,f3), max(f0,f2), ...}
    _s = (v4f32)__msa_shf_w((v4i32)_v, 0xB1);
    _v = __msa_fmax_w(_v, _s); // {max(f0,f1,f2,f3), ...}
    float result;
    __builtin_memcpy(&result, &_v, sizeof(float));
    return result;
}

static NCNN_FORCEINLINE float __msa_reduce_fadd_w(v4f32 _v)
{
    // _v = {f0, f1, f2, f3}
    // swap pairs: {f2, f3, f0, f1}
    v4f32 _s = (v4f32)__msa_shf_w((v4i32)_v, 0x4E);
    _v = __msa_fadd_w(_v, _s); // {f0+f2, f1+f3, ...}
    // swap within pair: {f1+f3, f0+f2, ...}
    _s = (v4f32)__msa_shf_w((v4i32)_v, 0xB1);
    _v = __msa_fadd_w(_v, _s); // {f0+f1+f2+f3, ...}
    float result;
    __builtin_memcpy(&result, &_v, sizeof(float));
    return result;
}

static NCNN_FORCEINLINE int __msa_reduce_add_w(v4i32 _v)
{
    v2i64 hi64 = (v2i64)__msa_ilvl_d((v2i64)_v, (v2i64)_v);
    v4i32 sum64 = __msa_addv_w((v4i32)hi64, _v);
    v4i32 hi32 = (v4i32)__msa_ilvr_w((v4i32)hi64, sum64);
    v4i32 sum32 = __msa_addv_w(sum64, hi32);
    return __msa_copy_s_w(sum32, 0);
}

static NCNN_FORCEINLINE int __msa_cfcmsa_msacsr()
{
    int v;
    asm volatile("cfcmsa %0, $1 \n"
                 : "=r"(v)
                 :
                 :);
    return v;
}

static NCNN_FORCEINLINE void __msa_ctcmsa_msacsr(int v)
{
    asm volatile("ctcmsa $1, %0 \n"
                 :
                 : "r"(v)
                 :);
}
#endif // __mips_msa

static NCNN_FORCEINLINE signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __mips_msa
static NCNN_FORCEINLINE v16i8 float2int8(v4f32 _v)
{
    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _sign = __msa_and_v((v16u8)_v, _signmask);
    v4f32 _p5s = (v4f32)__msa_or_v((v16u8)_p5, _sign);
    v4f32 _v5 = __msa_fadd_w(_v, _p5s);
    v4i32 _v32 = __msa_ftrunc_s_w(_v5);

    v8i16 _v32_16 = (v8i16)__msa_sat_s_w(_v32, 15);
    v8i16 _v16 = __msa_pckev_h(_v32_16, _v32_16);
    _v16 = __msa_max_s_h(_v16, __msa_fill_h(-127));
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v16i8 _v8 = __msa_pckev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8(v4f32 _vlow, v4f32 _vhigh)
{
    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _signlow = __msa_and_v((v16u8)_vlow, _signmask);
    v16u8 _signhigh = __msa_and_v((v16u8)_vhigh, _signmask);
    v4f32 _p5low = (v4f32)__msa_or_v((v16u8)_p5, _signlow);
    v4f32 _p5high = (v4f32)__msa_or_v((v16u8)_p5, _signhigh);
    v4f32 _vlow5 = __msa_fadd_w(_vlow, _p5low);
    v4f32 _vhigh5 = __msa_fadd_w(_vhigh, _p5high);
    v4i32 _vlow32 = __msa_ftrunc_s_w(_vlow5);
    v4i32 _vhigh32 = __msa_ftrunc_s_w(_vhigh5);

    v8i16 _vlow32_16 = (v8i16)__msa_sat_s_w(_vlow32, 15);
    v8i16 _vhigh32_16 = (v8i16)__msa_sat_s_w(_vhigh32, 15);
    v8i16 _v16 = __msa_pckev_h(_vhigh32_16, _vlow32_16);
    _v16 = __msa_max_s_h(_v16, __msa_fill_h(-127));
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v2i64 _v8 = (v2i64)__msa_pckev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE v16i8 float2int8relu(v4f32 _v)
{
    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _sign = __msa_and_v((v16u8)_v, _signmask);
    v4f32 _p5s = (v4f32)__msa_or_v((v16u8)_p5, _sign);
    v4f32 _v5 = __msa_fadd_w(_v, _p5s);
    v4i32 _v32 = __msa_ftrunc_s_w(_v5);

    v8i16 _v32_16 = (v8i16)__msa_sat_s_w(_v32, 15);
    v8i16 _v16 = __msa_pckev_h(_v32_16, _v32_16);
    _v16 = __msa_maxi_s_h(_v16, 0);
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v16i8 _v8 = __msa_pckev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8relu(v4f32 _vlow, v4f32 _vhigh)
{
    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _signlow = __msa_and_v((v16u8)_vlow, _signmask);
    v16u8 _signhigh = __msa_and_v((v16u8)_vhigh, _signmask);
    v4f32 _p5low = (v4f32)__msa_or_v((v16u8)_p5, _signlow);
    v4f32 _p5high = (v4f32)__msa_or_v((v16u8)_p5, _signhigh);
    v4f32 _vlow5 = __msa_fadd_w(_vlow, _p5low);
    v4f32 _vhigh5 = __msa_fadd_w(_vhigh, _p5high);
    v4i32 _vlow32 = __msa_ftrunc_s_w(_vlow5);
    v4i32 _vhigh32 = __msa_ftrunc_s_w(_vhigh5);

    v8i16 _vlow32_16 = (v8i16)__msa_sat_s_w(_vlow32, 15);
    v8i16 _vhigh32_16 = (v8i16)__msa_sat_s_w(_vhigh32, 15);
    v8i16 _v16 = __msa_pckev_h(_vhigh32_16, _vlow32_16);
    _v16 = __msa_maxi_s_h(_v16, 0);
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v2i64 _v8 = (v2i64)__msa_pckev_b(_v16_8, _v16_8);

    return _v8[0];
}

static NCNN_FORCEINLINE v16i8 float2int8leakyrelu(v4f32 _v, v4f32 _slope)
{
    v4f32 _v_leaky = __msa_fmul_w(_v, _slope);

    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _sign = __msa_and_v((v16u8)_v, _signmask);
    v4f32 _p5s = (v4f32)__msa_or_v((v16u8)_p5, _sign);
    v4f32 _v5 = __msa_fadd_w(_v, _p5s);
    v4i32 _v32 = __msa_ftrunc_s_w(_v5);

    v16u8 _sign_leaky = __msa_and_v((v16u8)_v_leaky, _signmask);
    v4f32 _p5_leaky = (v4f32)__msa_or_v((v16u8)_p5, _sign_leaky);
    v4f32 _v5_leaky = __msa_fadd_w(_v_leaky, _p5_leaky);
    v4i32 _v32_leaky = __msa_ftrunc_s_w(_v5_leaky);

    v8i16 _v32_16 = (v8i16)__msa_sat_s_w(_v32, 15);
    v8i16 _v16 = __msa_pckev_h(_v32_16, _v32_16);

    v8i16 _v32_16_leaky = (v8i16)__msa_sat_s_w(_v32_leaky, 15);
    v8i16 _v16_leaky = __msa_pckev_h(_v32_16_leaky, _v32_16_leaky);

    _v16 = __msa_max_s_h(_v16, _v16_leaky);
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v16i8 _v8 = __msa_pckev_b(_v16_8, _v16_8);

    return _v8;
}

static NCNN_FORCEINLINE int64_t float2int8leakyrelu(v4f32 _vlow, v4f32 _vhigh, v4f32 _slope)
{
    v4f32 _vlow_leaky = __msa_fmul_w(_vlow, _slope);
    v4f32 _vhigh_leaky = __msa_fmul_w(_vhigh, _slope);

    // simulate round to nearest via +/-0.5
    v4f32 _p5 = (v4f32)__msa_fill_w_f32(0.5f);
    v16u8 _signmask = (v16u8)__msa_fill_w(1 << 31);

    v16u8 _signlow = __msa_and_v((v16u8)_vlow, _signmask);
    v16u8 _signhigh = __msa_and_v((v16u8)_vhigh, _signmask);
    v4f32 _p5low = (v4f32)__msa_or_v((v16u8)_p5, _signlow);
    v4f32 _p5high = (v4f32)__msa_or_v((v16u8)_p5, _signhigh);
    v4f32 _vlow5 = __msa_fadd_w(_vlow, _p5low);
    v4f32 _vhigh5 = __msa_fadd_w(_vhigh, _p5high);
    v4i32 _vlow32 = __msa_ftrunc_s_w(_vlow5);
    v4i32 _vhigh32 = __msa_ftrunc_s_w(_vhigh5);

    v16u8 _signlow_leaky = __msa_and_v((v16u8)_vlow_leaky, _signmask);
    v16u8 _signhigh_leaky = __msa_and_v((v16u8)_vhigh_leaky, _signmask);
    v4f32 _p5low_leaky = (v4f32)__msa_or_v((v16u8)_p5, _signlow_leaky);
    v4f32 _p5high_leaky = (v4f32)__msa_or_v((v16u8)_p5, _signhigh_leaky);
    v4f32 _vlow5_leaky = __msa_fadd_w(_vlow_leaky, _p5low_leaky);
    v4f32 _vhigh5_leaky = __msa_fadd_w(_vhigh_leaky, _p5high_leaky);
    v4i32 _vlow32_leaky = __msa_ftrunc_s_w(_vlow5_leaky);
    v4i32 _vhigh32_leaky = __msa_ftrunc_s_w(_vhigh5_leaky);

    v8i16 _vlow32_16 = (v8i16)__msa_sat_s_w(_vlow32, 15);
    v8i16 _vhigh32_16 = (v8i16)__msa_sat_s_w(_vhigh32, 15);
    v8i16 _v16 = __msa_pckev_h(_vhigh32_16, _vlow32_16);

    v8i16 _vlow32_16_leaky = (v8i16)__msa_sat_s_w(_vlow32_leaky, 15);
    v8i16 _vhigh32_16_leaky = (v8i16)__msa_sat_s_w(_vhigh32_leaky, 15);
    v8i16 _v16_leaky = __msa_pckev_h(_vhigh32_16_leaky, _vlow32_16_leaky);

    _v16 = __msa_max_s_h(_v16, _v16_leaky);
    v16i8 _v16_8 = (v16i8)__msa_sat_s_h(_v16, 7);
    v2i64 _v8 = (v2i64)__msa_pckev_b(_v16_8, _v16_8);

    return _v8[0];
}

// transpose4x4_epi16 - transpose 4x4 block of int16
static NCNN_FORCEINLINE void transpose4x4_epi16(v8i16& _r0, v8i16& _r1, v8i16& _r2, v8i16& _r3)
{
    v8i16 _tmp0 = (v8i16)__msa_ilvr_h((v8i16)_r1, (v8i16)_r0);
    v8i16 _tmp1 = (v8i16)__msa_ilvl_h((v8i16)_r1, (v8i16)_r0);
    v8i16 _tmp2 = (v8i16)__msa_ilvr_h((v8i16)_r3, (v8i16)_r2);
    v8i16 _tmp3 = (v8i16)__msa_ilvl_h((v8i16)_r3, (v8i16)_r2);

    _r0 = (v8i16)__msa_ilvr_w((v4i32)_tmp2, (v4i32)_tmp0);
    _r1 = (v8i16)__msa_ilvl_w((v4i32)_tmp2, (v4i32)_tmp0);
    _r2 = (v8i16)__msa_ilvr_w((v4i32)_tmp3, (v4i32)_tmp1);
    _r3 = (v8i16)__msa_ilvl_w((v4i32)_tmp3, (v4i32)_tmp1);
}

// transpose4x4_epi32 - transpose 4x4 block of int32
static NCNN_FORCEINLINE void transpose4x4_epi32(v4i32& _r0, v4i32& _r1, v4i32& _r2, v4i32& _r3)
{
    v4i32 _tmp0 = (v4i32)__msa_ilvr_w(_r1, _r0);
    v4i32 _tmp1 = (v4i32)__msa_ilvl_w(_r1, _r0);
    v4i32 _tmp2 = (v4i32)__msa_ilvr_w(_r3, _r2);
    v4i32 _tmp3 = (v4i32)__msa_ilvl_w(_r3, _r2);

    _r0 = (v4i32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r1 = (v4i32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r2 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp1);
    _r3 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp1);
}

// transpose4x4_ps - transpose 4x4 block of float
static NCNN_FORCEINLINE void transpose4x4_ps(v4f32& _r0, v4f32& _r1, v4f32& _r2, v4f32& _r3)
{
    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_r3, (v4i32)_r2);

    _r0 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r1 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r2 = (v4f32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp1);
    _r3 = (v4f32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp1);
}

// transpose4x8_epi32 - transpose 4x8 block of int32
static NCNN_FORCEINLINE void transpose4x8_epi32(v4i32& _r0, v4i32& _r1, v4i32& _r2, v4i32& _r3, v4i32& _r4, v4i32& _r5, v4i32& _r6, v4i32& _r7)
{
    v4i32 _tmp0 = (v4i32)__msa_ilvr_w(_r1, _r0);
    v4i32 _tmp1 = (v4i32)__msa_ilvl_w(_r1, _r0);
    v4i32 _tmp2 = (v4i32)__msa_ilvr_w(_r3, _r2);
    v4i32 _tmp3 = (v4i32)__msa_ilvl_w(_r3, _r2);
    v4i32 _tmp4 = (v4i32)__msa_ilvr_w(_r5, _r4);
    v4i32 _tmp5 = (v4i32)__msa_ilvl_w(_r5, _r4);
    v4i32 _tmp6 = (v4i32)__msa_ilvr_w(_r7, _r6);
    v4i32 _tmp7 = (v4i32)__msa_ilvl_w(_r7, _r6);

    _r0 = (v4i32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r1 = (v4i32)__msa_ilvr_d((v2i64)_tmp6, (v2i64)_tmp4);
    _r2 = (v4i32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r3 = (v4i32)__msa_ilvl_d((v2i64)_tmp6, (v2i64)_tmp4);
    _r4 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp1);
    _r5 = (v4i32)__msa_ilvr_d((v2i64)_tmp7, (v2i64)_tmp5);
    _r6 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp1);
    _r7 = (v4i32)__msa_ilvl_d((v2i64)_tmp7, (v2i64)_tmp5);
}

// transpose8x8_epi16 - transpose 8x8 block of int16
static NCNN_FORCEINLINE void transpose8x8_epi16(v8i16& _r0, v8i16& _r1, v8i16& _r2, v8i16& _r3, v8i16& _r4, v8i16& _r5, v8i16& _r6, v8i16& _r7)
{
    v8i16 _tmp0 = (v8i16)__msa_ilvr_h(_r1, _r0);
    v8i16 _tmp1 = (v8i16)__msa_ilvl_h(_r1, _r0);
    v8i16 _tmp2 = (v8i16)__msa_ilvr_h(_r3, _r2);
    v8i16 _tmp3 = (v8i16)__msa_ilvl_h(_r3, _r2);
    v8i16 _tmp4 = (v8i16)__msa_ilvr_h(_r5, _r4);
    v8i16 _tmp5 = (v8i16)__msa_ilvl_h(_r5, _r4);
    v8i16 _tmp6 = (v8i16)__msa_ilvr_h(_r7, _r6);
    v8i16 _tmp7 = (v8i16)__msa_ilvl_h(_r7, _r6);

    v8i16 _tmp8 = (v8i16)__msa_ilvr_w((v4i32)_tmp2, (v4i32)_tmp0);
    v8i16 _tmp9 = (v8i16)__msa_ilvl_w((v4i32)_tmp2, (v4i32)_tmp0);
    v8i16 _tmpa = (v8i16)__msa_ilvr_w((v4i32)_tmp3, (v4i32)_tmp1);
    v8i16 _tmpb = (v8i16)__msa_ilvl_w((v4i32)_tmp3, (v4i32)_tmp1);
    v8i16 _tmpc = (v8i16)__msa_ilvr_w((v4i32)_tmp6, (v4i32)_tmp4);
    v8i16 _tmpd = (v8i16)__msa_ilvl_w((v4i32)_tmp6, (v4i32)_tmp4);
    v8i16 _tmpe = (v8i16)__msa_ilvr_w((v4i32)_tmp7, (v4i32)_tmp5);
    v8i16 _tmpf = (v8i16)__msa_ilvl_w((v4i32)_tmp7, (v4i32)_tmp5);

    _r0 = (v8i16)__msa_ilvr_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r1 = (v8i16)__msa_ilvl_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r2 = (v8i16)__msa_ilvr_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r3 = (v8i16)__msa_ilvl_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r4 = (v8i16)__msa_ilvr_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r5 = (v8i16)__msa_ilvl_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r6 = (v8i16)__msa_ilvr_d((v2i64)_tmpb, (v2i64)_tmpf);
    _r7 = (v8i16)__msa_ilvl_d((v2i64)_tmpb, (v2i64)_tmpf);
}

// transpose8x4_epi16 - transpose 8x4 block of int16
static NCNN_FORCEINLINE void transpose8x4_epi16(v8i16& _r0, v8i16& _r1, v8i16& _r2, v8i16& _r3)
{
    v8i16 _tmp0 = (v8i16)__msa_ilvr_h(_r1, _r0);
    v8i16 _tmp1 = (v8i16)__msa_ilvl_h(_r1, _r0);
    v8i16 _tmp2 = (v8i16)__msa_ilvr_h(_r3, _r2);
    v8i16 _tmp3 = (v8i16)__msa_ilvl_h(_r3, _r2);

    _r0 = (v8i16)__msa_ilvr_w((v4i32)_tmp2, (v4i32)_tmp0);
    _r1 = (v8i16)__msa_ilvl_w((v4i32)_tmp2, (v4i32)_tmp0);
    _r2 = (v8i16)__msa_ilvr_w((v4i32)_tmp3, (v4i32)_tmp1);
    _r3 = (v8i16)__msa_ilvl_w((v4i32)_tmp3, (v4i32)_tmp1);
}

// transpose16x4_epi8 - transpose 16x4 block of int8
static NCNN_FORCEINLINE void transpose16x4_epi8(v16i8& _r0, v16i8& _r1, v16i8& _r2, v16i8& _r3)
{
    v16i8 _tmp0 = (v16i8)__msa_ilvr_b(_r1, _r0);
    v16i8 _tmp1 = (v16i8)__msa_ilvl_b(_r1, _r0);
    v16i8 _tmp2 = (v16i8)__msa_ilvr_b(_r3, _r2);
    v16i8 _tmp3 = (v16i8)__msa_ilvl_b(_r3, _r2);

    _r0 = (v16i8)__msa_ilvr_h((v8i16)_tmp2, (v8i16)_tmp0);
    _r1 = (v16i8)__msa_ilvl_h((v8i16)_tmp2, (v8i16)_tmp0);
    _r2 = (v16i8)__msa_ilvr_h((v8i16)_tmp3, (v8i16)_tmp1);
    _r3 = (v16i8)__msa_ilvl_h((v8i16)_tmp3, (v8i16)_tmp1);
}

// transpose8x4_epi8 - transpose 8x4 block of int8
static NCNN_FORCEINLINE void transpose8x4_epi8(v16i8& _r0, v16i8& _r1, v16i8& _r2, v16i8& _r3)
{
    v16i8 _tmp0 = (v16i8)__msa_ilvr_b(_r1, _r0);
    v16i8 _tmp1 = (v16i8)__msa_ilvr_h((v8i16)_r3, (v8i16)_r2);

    _r0 = (v16i8)__msa_ilvr_w((v4i32)_tmp1, (v4i32)_tmp0);
    _r1 = (v16i8)__msa_ilvl_w((v4i32)_tmp1, (v4i32)_tmp0);
}

// transpose8x4_ps - transpose 8x4 block of float
static NCNN_FORCEINLINE void transpose8x4_ps(v4f32& _r0, v4f32& _r1, v4f32& _r2, v4f32& _r3)
{
    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_r3, (v4i32)_r2);

    _r0 = (v4f32)__msa_ilvr_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r1 = (v4f32)__msa_ilvl_d((v2i64)_tmp2, (v2i64)_tmp0);
    _r2 = (v4f32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp1);
    _r3 = (v4f32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp1);
}

// transpose8x8_ps - transpose 8x8 block of float
static NCNN_FORCEINLINE void transpose8x8_ps(v4f32& _r0, v4f32& _r1, v4f32& _r2, v4f32& _r3, v4f32& _r4, v4f32& _r5, v4f32& _r6, v4f32& _r7)
{
    v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
    v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
    v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
    v4f32 _tmp4 = (v4f32)__msa_ilvr_w((v4i32)_r5, (v4i32)_r4);
    v4f32 _tmp5 = (v4f32)__msa_ilvl_w((v4i32)_r5, (v4i32)_r4);
    v4f32 _tmp6 = (v4f32)__msa_ilvr_w((v4i32)_r7, (v4i32)_r6);
    v4f32 _tmp7 = (v4f32)__msa_ilvl_w((v4i32)_r7, (v4i32)_r6);

    v4f32 _tmp8 = (v4f32)__msa_ilvr_w((v4i32)_tmp2, (v4i32)_tmp0);
    v4f32 _tmp9 = (v4f32)__msa_ilvl_w((v4i32)_tmp2, (v4i32)_tmp0);
    v4f32 _tmpa = (v4f32)__msa_ilvr_w((v4i32)_tmp3, (v4i32)_tmp1);
    v4f32 _tmpb = (v4f32)__msa_ilvl_w((v4i32)_tmp3, (v4i32)_tmp1);
    v4f32 _tmpc = (v4f32)__msa_ilvr_w((v4i32)_tmp6, (v4i32)_tmp4);
    v4f32 _tmpd = (v4f32)__msa_ilvl_w((v4i32)_tmp6, (v4i32)_tmp4);
    v4f32 _tmpe = (v4f32)__msa_ilvr_w((v4i32)_tmp7, (v4i32)_tmp5);
    v4f32 _tmpf = (v4f32)__msa_ilvl_w((v4i32)_tmp7, (v4i32)_tmp5);

    _r0 = (v4f32)__msa_ilvr_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r1 = (v4f32)__msa_ilvl_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r2 = (v4f32)__msa_ilvr_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r3 = (v4f32)__msa_ilvl_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r4 = (v4f32)__msa_ilvr_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r5 = (v4f32)__msa_ilvl_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r6 = (v4f32)__msa_ilvr_d((v2i64)_tmpb, (v2i64)_tmpf);
    _r7 = (v4f32)__msa_ilvl_d((v2i64)_tmpb, (v2i64)_tmpf);
}

// transpose8x8_epi32 - transpose 8x8 block of int32
static NCNN_FORCEINLINE void transpose8x8_epi32(v4i32& _r0, v4i32& _r1, v4i32& _r2, v4i32& _r3, v4i32& _r4, v4i32& _r5, v4i32& _r6, v4i32& _r7)
{
    v4i32 _tmp0 = (v4i32)__msa_ilvr_w(_r1, _r0);
    v4i32 _tmp1 = (v4i32)__msa_ilvl_w(_r1, _r0);
    v4i32 _tmp2 = (v4i32)__msa_ilvr_w(_r3, _r2);
    v4i32 _tmp3 = (v4i32)__msa_ilvl_w(_r3, _r2);
    v4i32 _tmp4 = (v4i32)__msa_ilvr_w(_r5, _r4);
    v4i32 _tmp5 = (v4i32)__msa_ilvl_w(_r5, _r4);
    v4i32 _tmp6 = (v4i32)__msa_ilvr_w(_r7, _r6);
    v4i32 _tmp7 = (v4i32)__msa_ilvl_w(_r7, _r6);

    v4i32 _tmp8 = (v4i32)__msa_ilvr_w(_tmp2, _tmp0);
    v4i32 _tmp9 = (v4i32)__msa_ilvl_w(_tmp2, _tmp0);
    v4i32 _tmpa = (v4i32)__msa_ilvr_w(_tmp3, _tmp1);
    v4i32 _tmpb = (v4i32)__msa_ilvl_w(_tmp3, _tmp1);
    v4i32 _tmpc = (v4i32)__msa_ilvr_w(_tmp6, _tmp4);
    v4i32 _tmpd = (v4i32)__msa_ilvl_w(_tmp6, _tmp4);
    v4i32 _tmpe = (v4i32)__msa_ilvr_w(_tmp7, _tmp5);
    v4i32 _tmpf = (v4i32)__msa_ilvl_w(_tmp7, _tmp5);

    _r0 = (v4i32)__msa_ilvr_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r1 = (v4i32)__msa_ilvl_d((v2i64)_tmp8, (v2i64)_tmpc);
    _r2 = (v4i32)__msa_ilvr_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r3 = (v4i32)__msa_ilvl_d((v2i64)_tmp9, (v2i64)_tmpd);
    _r4 = (v4i32)__msa_ilvr_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r5 = (v4i32)__msa_ilvl_d((v2i64)_tmpa, (v2i64)_tmpe);
    _r6 = (v4i32)__msa_ilvr_d((v2i64)_tmpb, (v2i64)_tmpf);
    _r7 = (v4i32)__msa_ilvl_d((v2i64)_tmpb, (v2i64)_tmpf);
}

// BF16 conversion utilities
static NCNN_FORCEINLINE v4f32 bfloat2float_msa(const v4i32& v0)
{
    // BF16 values in low 64 bits (4 x 16-bit)
    // Interleave with zeros to zero-extend each bf16 to high 16 bits of 32-bit fp32
    v8i16 _zero = (v8i16)__msa_fill_w(0);
    return (v4f32)__msa_ilvr_h((v8i16)v0, _zero);
}

// Load 4 bf16 values from potentially unaligned pointer and convert to fp32
// Use __msa_ld_b instead of __msa_ld_w to avoid 16-byte alignment masking
static NCNN_FORCEINLINE v4f32 bfloat2float_msa(const unsigned short* ptr)
{
    // Use 64-bit load (not 128-bit __msa_ld_b) to avoid overread past allocation
    int64_t v;
    memcpy(&v, ptr, 8);
    v8i16 _zero = (v8i16)__msa_fill_w(0);
    v8i16 _raw = (v8i16)__msa_fill_d(v);
    return (v4f32)__msa_ilvr_h(_raw, _zero);
}

static NCNN_FORCEINLINE v4i32 float2bfloat_msa(const v4f32& v0)
{
    v4i32 _a = (v4i32)v0;
    _a = __msa_srli_w(_a, 16);
    v8i16 _v = __msa_pckev_h((v8i16)__msa_fill_w(0), (v8i16)_a);
    return (v4i32)_v;
}

// Store 4 bf16 values to potentially unaligned pointer
static NCNN_FORCEINLINE void float2bfloat_msa_store(const v4f32& v0, unsigned short* ptr)
{
    v4i32 _bf16 = float2bfloat_msa(v0);
    int64_t val = __msa_copy_s_d((v2i64)_bf16, 0);
    __builtin_memcpy(ptr, &val, sizeof(int64_t));
}

static NCNN_FORCEINLINE v4i32 float2bfloat_msa(const v4f32& v0, const v4f32& v1)
{
    v4i32 _a = (v4i32)v0;
    v4i32 _b = (v4i32)v1;
    _a = __msa_srli_w(_a, 16);
    _b = __msa_srli_w(_b, 16);
    v8i16 _v = __msa_pckev_h((v8i16)_b, (v8i16)_a);
    return (v4i32)_v;
}

#endif // __mips_msa

#endif // MIPS_USABILITY_H
