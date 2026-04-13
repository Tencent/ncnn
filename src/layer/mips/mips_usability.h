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

// FMA equivalents (MSA has fmadd/fmsub)
static NCNN_FORCEINLINE v4f32 _mm_comp_fmadd_ps(const v4f32& _a, const v4f32& _b, const v4f32& _c)
{
    return __msa_fmadd_w(_c, _a, _b);
}

static NCNN_FORCEINLINE v4f32 _mm_comp_fnmadd_ps(const v4f32& _a, const v4f32& _b, const v4f32& _c)
{
    // return -a * b + c = c - a * b
    return __msa_fmsub_w(_c, _a, _b);
}

static NCNN_FORCEINLINE v4f32 _mm_comp_fmsub_ps(const v4f32& _a, const v4f32& _b, const v4f32& _c)
{
    // return a * b - c
    return __msa_fsub_w(__msa_fmul_w(_a, _b), _c);
}

static NCNN_FORCEINLINE v4f32 _mm_comp_fnmsub_ps(const v4f32& _a, const v4f32& _b, const v4f32& _c)
{
    // return -(a * b) - c
    v4f32 neg_ab = (v4f32)__msa_xor_v((v16u8)__msa_fmul_w(_a, _b), (v16u8)__msa_fill_w(0x80000000));
    return __msa_fsub_w(neg_ab, _c);
}

// _mm_comp_mullo_epi32 - SSE4.1 equivalent using MSA mulv
static NCNN_FORCEINLINE v4i32 _mm_comp_mullo_epi32(const v4i32& a, const v4i32& b)
{
    return __msa_mulv_w(a, b);
}

// _mm_rcp_nr_ps - reciprocal with Newton-Raphson refinement
static NCNN_FORCEINLINE v4f32 _mm_rcp_nr_ps(const v4f32& x)
{
    v4f32 y = __msa_frcp_w(x);
    // Newton-Raphson: y = y * (2 - x * y)
    v4f32 two = __msa_fill_w_f32(2.0f);
    v4f32 t = __msa_fmsub_w(two, x, y); // 2 - x * y
    y = __msa_fmul_w(y, t);
    return y;
}

// reduce operations
static NCNN_FORCEINLINE float _mm_reduce_add_ps(const v4f32& x)
{
    // shf immediate: (1,0,3,2) = 0x4E, (0,3,2,1) = 0x39
    v4f32 s = __msa_fadd_w(x, (v4f32)__msa_shf_w((v4i32)x, 0x4E));
    s = __msa_fadd_w(s, (v4f32)__msa_shf_w((v4i32)s, 0x39));
    float ret;
    __msa_st_w((v4i32)s, &ret, 0);
    return ret;
}

static NCNN_FORCEINLINE float _mm_reduce_max_ps(const v4f32& x)
{
    v4f32 s = __msa_fmax_w(x, (v4f32)__msa_shf_w((v4i32)x, 0x4E));
    s = __msa_fmax_w(s, (v4f32)__msa_shf_w((v4i32)s, 0x39));
    float ret;
    __msa_st_w((v4i32)s, &ret, 0);
    return ret;
}

static NCNN_FORCEINLINE int _mm_reduce_add_epi32(const v4i32& x)
{
    v2i64 hi64 = (v2i64)__msa_ilvl_d((v2i64)x, (v2i64)x);
    v4i32 sum64 = __msa_addv_w((v4i32)hi64, x);
    v4i32 hi32 = (v4i32)__msa_ilvr_w((v4i32)hi64, sum64);
    v4i32 sum32 = __msa_addv_w(sum64, hi32);
    return __msa_copy_s_w(sum32, 0);
}

static NCNN_FORCEINLINE int _mm_reduce_max_epi32(const v4i32& x)
{
    v2i64 hi64 = (v2i64)__msa_ilvl_d((v2i64)x, (v2i64)x);
    v4i32 max64 = __msa_max_s_w((v4i32)hi64, x);
    v4i32 hi32 = (v4i32)__msa_ilvr_w((v4i32)hi64, max64);
    v4i32 max32 = __msa_max_s_w(max64, hi32);
    return __msa_copy_s_w(max32, 0);
}

static NCNN_FORCEINLINE int _mm_reduce_min_epi32(const v4i32& x)
{
    v2i64 hi64 = (v2i64)__msa_ilvl_d((v2i64)x, (v2i64)x);
    v4i32 min64 = __msa_min_s_w((v4i32)hi64, x);
    v4i32 hi32 = (v4i32)__msa_ilvr_w((v4i32)hi64, min64);
    v4i32 min32 = __msa_min_s_w(min64, hi32);
    return __msa_copy_s_w(min32, 0);
}

static NCNN_FORCEINLINE int64_t _mm_reduce_add_epi64(const v2i64& x)
{
    v2i64 hi64 = (v2i64)__msa_ilvl_d(x, x);
    v2i64 sum64 = __msa_addv_d(hi64, x);
    return __msa_copy_s_d(sum64, 0);
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
    v8i16 _zero = (v8i16)__msa_fill_w(0);
    v16i8 _raw = __msa_ld_b(ptr, 0);
    return (v4f32)__msa_ilvr_h((v8i16)_raw, _zero);
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

// DPWSSD for INT8 dot product
static NCNN_FORCEINLINE v4i32 _mm_comp_dpwssd_epi32(const v4i32& src, const v4i32& a, const v4i32& b)
{
    return __msa_dpadd_s_w(src, (v8i16)a, (v8i16)b);
}

// DPBUSSD for unsigned byte dot product
static NCNN_FORCEINLINE v4i32 _mm_comp_dpbusd_epi32(const v4i32& src, const v16u8& a, const v16u8& b)
{
    // Compute dot product of unsigned bytes in groups of 4 -> int32
    // Zero-extend unsigned bytes to signed 16-bit
    v16i8 _zero = (v16i8)__msa_fill_w(0);
    v8i16 a_lo = (v8i16)__msa_ilvr_b(_zero, (v16i8)a);
    v8i16 a_hi = (v8i16)__msa_ilvl_b(_zero, (v16i8)a);
    v8i16 b_lo = (v8i16)__msa_ilvr_b(_zero, (v16i8)b);
    v8i16 b_hi = (v8i16)__msa_ilvl_b(_zero, (v16i8)b);
    // Multiply and add pairs of 16-bit -> 32-bit
    v4i32 prod_lo = __msa_dpadd_s_w(__msa_fill_w(0), a_lo, b_lo);
    v4i32 prod_hi = __msa_dpadd_s_w(__msa_fill_w(0), a_hi, b_hi);
    v4i32 sum = __msa_addv_w(prod_lo, prod_hi);
    return __msa_addv_w(sum, src);
}

// DPWSSDS - signed saturated version
static NCNN_FORCEINLINE v4i32 _mm_comp_dpwssds_epi32(const v4i32& src, const v4i32& a, const v4i32& b)
{
    v4i32 prod = __msa_dpadd_s_w(src, (v8i16)a, (v8i16)b);
    return __msa_sat_s_w(prod, 31);
}

// DPBUSSDS - unsigned saturated version
static NCNN_FORCEINLINE v4i32 _mm_comp_dpbusds_epi32(const v4i32& src, const v16u8& a, const v16u8& b)
{
    v4i32 dp = _mm_comp_dpbusd_epi32(src, a, b);
    return __msa_sat_s_w(dp, 31);
}

// int8_short_to_int32_scalar
static NCNN_FORCEINLINE int32_t int8_short_to_int32_scalar(int8_t* v0)
{
    int32_t _v0 = v0[0] + (v0[1] << 8) + (v0[2] << 16) + (v0[3] << 24);
    return _v0;
}

// HorizontalSums for 8 accumulators
static NCNN_FORCEINLINE v4f32 HorizontalSums(v4f32& v0, v4f32& v1, v4f32& v2, v4f32& v3, v4f32& v4, v4f32& v5, v4f32& v6, v4f32& v7)
{
    v4f32 s01 = __msa_fadd_w(v0, v1);
    v4f32 s23 = __msa_fadd_w(v2, v3);
    v4f32 s45 = __msa_fadd_w(v4, v5);
    v4f32 s67 = __msa_fadd_w(v6, v7);
    v4f32 s0123 = __msa_fadd_w(s01, s23);
    v4f32 s4556 = __msa_fadd_w(s45, s67);
    return __msa_fadd_w(s0123, s4556);
}

// fast integer division
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
        _multiplier = __msa_fill_w(m);
        _shift1 = __msa_fill_w(sh1);
        _shift2 = __msa_fill_w(sh2);
    }

    NCNN_FORCEINLINE v4i32 _mm_comp_div_epu32(const v4i32& x) const
    {
        // Scalar fallback for integer division by constant
        int tmp[4];
        __msa_st_w(x, tmp, 0);
        int mul[4];
        __msa_st_w(_multiplier, mul, 0);
        int sh1[4];
        __msa_st_w(_shift1, sh1, 0);
        int sh2[4];
        __msa_st_w(_shift2, sh2, 0);
        int result[4];
        for (int i = 0; i < 4; i++)
        {
            uint32_t xu = (uint32_t)tmp[i];
            uint64_t xm = ((uint64_t)xu * (uint64_t)(uint32_t)mul[i]) >> 32;
            result[i] = (int)((xm + ((xu - xm) >> (uint32_t)sh1[i])) >> (uint32_t)sh2[i]);
        }
        return __msa_ld_w(result, 0);
    }

protected:
    static int portable_ceil_log2(int d)
    {
        return 32 - __builtin_clz(d - 1);
    }

protected:
    v4i32 _multiplier;
    v4i32 _shift1;
    v4i32 _shift2;
};

// Typedefs for x86 source compatibility
typedef v4f32 __m128;
typedef v4i32 __m128i;
typedef v2f64 __m128d;

// _MM_SHUFFLE macro for x86 source compatibility
#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

// bfloat2float_sse compatibility alias
#define bfloat2float_sse bfloat2float_msa
#define float2bfloat_sse float2bfloat_msa

// x86-compatibility wrappers for MIPS MSA
// These allow porting x86 SSE2 code to MIPS MSA with minimal changes

// Float intrinsics
static NCNN_FORCEINLINE v4f32 _mm_setzero_ps()
{
    return (v4f32)__msa_fill_w(0);
}

static NCNN_FORCEINLINE v4f32 _mm_set1_ps(float val)
{
    return __msa_fill_w_f32(val);
}

static NCNN_FORCEINLINE v4f32 _mm_load_ps(const float* addr)
{
    return (v4f32)__msa_ld_w((const void*)addr, 0);
}

static NCNN_FORCEINLINE v4f32 _mm_loadu_ps(const float* addr)
{
    return (v4f32)__msa_ld_w((const void*)addr, 0);
}

static NCNN_FORCEINLINE void _mm_store_ps(float* addr, v4f32 val)
{
    __msa_st_w((v4i32)val, (void*)addr, 0);
}

static NCNN_FORCEINLINE void _mm_storeu_ps(float* addr, v4f32 val)
{
    __msa_st_w((v4i32)val, (void*)addr, 0);
}

static NCNN_FORCEINLINE v4f32 _mm_add_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_fadd_w(a, b);
}

static NCNN_FORCEINLINE v4f32 _mm_mul_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_fmul_w(a, b);
}

static NCNN_FORCEINLINE v4f32 _mm_shuffle_ps(v4f32 a, v4f32 b, int imm)
{
    // _MM_SHUFFLE immediate is compatible with MSA shf.w immediate
    return (v4f32)__msa_shf_w((v4i32)a, imm);
}

static NCNN_FORCEINLINE v4f32 _mm_unpacklo_ps(v4f32 a, v4f32 b)
{
    // x86: {a[0],b[0],a[1],b[1]}
    // MSA ilvr_w(ws,wt) = {wt[0],ws[0],wt[1],ws[1]}
    return (v4f32)__msa_ilvr_w((v4i32)b, (v4i32)a);
}

static NCNN_FORCEINLINE v4f32 _mm_unpackhi_ps(v4f32 a, v4f32 b)
{
    // x86: {a[2],b[2],a[3],b[3]}
    // MSA ilvl_w(ws,wt) = {wt[2],ws[2],wt[3],ws[3]}
    return (v4f32)__msa_ilvl_w((v4i32)b, (v4i32)a);
}

static NCNN_FORCEINLINE v4f32 _mm_castpd_ps(v2f64 a)
{
    return (v4f32)a;
}

static NCNN_FORCEINLINE v2f64 _mm_castps_pd(v4f32 a)
{
    return (v2f64)a;
}

static NCNN_FORCEINLINE v4f32 _mm_load1_ps(const float* addr)
{
    return __msa_fill_w_f32(*addr);
}

static NCNN_FORCEINLINE v4f32 _mm_setr_ps(float e3, float e2, float e1, float e0)
{
    float tmp[4] = {e3, e2, e1, e0};
    return (v4f32)__msa_ld_w(tmp, 0);
}

// Integer intrinsics
static NCNN_FORCEINLINE v4i32 _mm_setzero_si128()
{
    return __msa_fill_w(0);
}

static NCNN_FORCEINLINE v4i32 _mm_set1_epi32(int val)
{
    return __msa_fill_w(val);
}

static NCNN_FORCEINLINE v4i32 _mm_load_si128(const v4i32* addr)
{
    return __msa_ld_w((const void*)addr, 0);
}

static NCNN_FORCEINLINE v4i32 _mm_loadu_si128(const v4i32* addr)
{
    return __msa_ld_w((const void*)addr, 0);
}

static NCNN_FORCEINLINE void _mm_store_si128(v4i32* addr, v4i32 val)
{
    __msa_st_w(val, (void*)addr, 0);
}

static NCNN_FORCEINLINE void _mm_storeu_si128(v4i32* addr, v4i32 val)
{
    __msa_st_w(val, (void*)addr, 0);
}

static NCNN_FORCEINLINE v2i64 _mm_loadl_epi64(const void* addr)
{
    return (v2i64)__msa_ld_d((const void*)addr, 0);
}

static NCNN_FORCEINLINE void _mm_storel_epi64(void* addr, v2i64 val)
{
    __msa_st_d(val, (void*)addr, 0);
}

static NCNN_FORCEINLINE v4i32 _mm_castps_si128(v4f32 a)
{
    return (v4i32)a;
}

static NCNN_FORCEINLINE v2f64 _mm_castsi128_pd(v4i32 a)
{
    return (v2f64)a;
}

static NCNN_FORCEINLINE v2i64 _mm_castpd_si128(v2f64 a)
{
    return (v2i64)a;
}

static NCNN_FORCEINLINE v4i32 _mm_unpacklo_epi16(v8i16 a, v8i16 b)
{
    return (v4i32)__msa_ilvr_h(b, a);
}

static NCNN_FORCEINLINE v4i32 _mm_unpackhi_epi16(v8i16 a, v8i16 b)
{
    return (v4i32)__msa_ilvl_h(b, a);
}

static NCNN_FORCEINLINE v4i32 _mm_unpacklo_epi8(v16i8 a, v16i8 b)
{
    return (v4i32)__msa_ilvr_b(b, a);
}

static NCNN_FORCEINLINE v4i32 _mm_unpackhi_epi8(v16i8 a, v16i8 b)
{
    return (v4i32)__msa_ilvl_b(b, a);
}

static NCNN_FORCEINLINE v4i32 _mm_unpacklo_epi32(v4i32 a, v4i32 b)
{
    return __msa_ilvr_w(b, a);
}

static NCNN_FORCEINLINE v4i32 _mm_unpackhi_epi32(v4i32 a, v4i32 b)
{
    return __msa_ilvl_w(b, a);
}

static NCNN_FORCEINLINE v2i64 _mm_unpacklo_pd(v2f64 a, v2f64 b)
{
    return (v2i64)__msa_ilvr_d((v2i64)b, (v2i64)a);
}

static NCNN_FORCEINLINE v2i64 _mm_unpackhi_pd(v2f64 a, v2f64 b)
{
    return (v2i64)__msa_ilvl_d((v2i64)b, (v2i64)a);
}

static NCNN_FORCEINLINE v4f32 _mm_castsi128_ps(v4i32 a)
{
    return (v4f32)a;
}

static NCNN_FORCEINLINE void _mm_storeh_pd(double* addr, v2f64 val)
{
    __msa_st_d((v2i64)val, (void*)addr, 8);
}

static NCNN_FORCEINLINE v4i32 _mm_slli_epi32(v4i32 a, int imm)
{
    return (v4i32)__msa_slli_w(a, imm);
}

static NCNN_FORCEINLINE v4i32 _mm_srli_epi32(v4i32 a, int imm)
{
    return (v4i32)__msa_srli_w(a, imm);
}

static NCNN_FORCEINLINE v4i32 _mm_and_si128(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_and_v((v16u8)a, (v16u8)b);
}

static NCNN_FORCEINLINE v4i32 _mm_or_si128(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_or_v((v16u8)a, (v16u8)b);
}

// _mm_srli_si128 - whole-register byte shift right by immediate
static NCNN_FORCEINLINE v4i32 _mm_srli_si128(v4i32 a, int imm)
{
    if (imm == 8)
    {
        // High 64 bits to low 64 bits, zero the high
        v2i64 _zero = (v2i64)__msa_fill_w(0);
        return (v4i32)__msa_ilvl_d(_zero, (v2i64)a);
    }
    // General byte shift using sldi
    return (v4i32)__msa_sldi_b((v16i8)__msa_fill_w(0), (v16i8)a, imm);
}

// _mm_xor_si128 - bitwise xor
static NCNN_FORCEINLINE v4i32 _mm_xor_si128(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_xor_v((v16u8)a, (v16u8)b);
}

// _mm_xor_ps - float xor (for sign bit tricks)
static NCNN_FORCEINLINE v4f32 _mm_xor_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_xor_v((v16u8)a, (v16u8)b);
}

// _mm_sub_epi32 - integer subtract
static NCNN_FORCEINLINE v4i32 _mm_sub_epi32(v4i32 a, v4i32 b)
{
    return __msa_subv_w(a, b);
}

// _mm_sub_ps - float subtract
static NCNN_FORCEINLINE v4f32 _mm_sub_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_fsub_w(a, b);
}

// _mm_subs_epi16 - signed saturated subtract
static NCNN_FORCEINLINE v8i16 _mm_subs_epi16(v8i16 a, v8i16 b)
{
    return (v8i16)__msa_subs_s_h(a, b);
}

// _mm_subs_epu8 - unsigned saturated subtract
static NCNN_FORCEINLINE v16u8 _mm_subs_epu8(v16u8 a, v16u8 b)
{
    return __msa_subs_u_b(a, b);
}

// _mm_cmpgt_epi32 - greater than compare
static NCNN_FORCEINLINE v4i32 _mm_cmpgt_epi32(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_clt_s_w(b, a);
}

// _mm_cmpeq_epi8 - equality compare for bytes
static NCNN_FORCEINLINE v16i8 _mm_cmpeq_epi8(v16i8 a, v16i8 b)
{
    return (v16i8)__msa_ceq_b(a, b);
}

// _mm_min_epi16 - signed min for 16-bit
static NCNN_FORCEINLINE v8i16 _mm_min_epi16(v8i16 a, v8i16 b)
{
    return __msa_min_s_h(a, b);
}

// _mm_max_epi16 - signed max for 16-bit
static NCNN_FORCEINLINE v8i16 _mm_max_epi16(v8i16 a, v8i16 b)
{
    return __msa_max_s_h(a, b);
}

// _mm_min_epu8 - unsigned min for bytes
static NCNN_FORCEINLINE v16u8 _mm_min_epu8(v16u8 a, v16u8 b)
{
    return __msa_min_u_b(a, b);
}

// _mm_max_epu8 - unsigned max for bytes
static NCNN_FORCEINLINE v16u8 _mm_max_epu8(v16u8 a, v16u8 b)
{
    return __msa_max_u_b(a, b);
}

// _mm_packs_epi32 - pack signed 32-bit to signed 16-bit with saturation
static NCNN_FORCEINLINE v8i16 _mm_packs_epi32(v4i32 a, v4i32 b)
{
    v4i32 _a = __msa_sat_s_w(a, 15);
    v4i32 _b = __msa_sat_s_w(b, 15);
    return (v8i16)__msa_pckev_h((v8i16)_b, (v8i16)_a);
}

// _mm_packus_epi16 - pack signed 16-bit to unsigned 8-bit with unsigned saturation
static NCNN_FORCEINLINE v16u8 _mm_packus_epi16(v8i16 a, v8i16 b)
{
    v8i16 _a = (v8i16)__msa_sat_u_h((v8u16)__msa_max_s_h(a, __msa_fill_h(0)), 7);
    v8i16 _b = (v8i16)__msa_sat_u_h((v8u16)__msa_max_s_h(b, __msa_fill_h(0)), 7);
    return (v16u8)__msa_pckev_b((v16i8)_b, (v16i8)_a);
}

// _mm_cvtepi32_ps - convert int32 to float
static NCNN_FORCEINLINE v4f32 _mm_cvtepi32_ps(v4i32 a)
{
    return (v4f32)__msa_ffint_s_w(a);
}

// _mm_cvttps_epi32 - convert float to int32 with truncate
static NCNN_FORCEINLINE v4i32 _mm_cvttps_epi32(v4f32 a)
{
    return __msa_ftrunc_s_w(a);
}

// _mm_movemask_epi8 - create mask from sign bits of bytes
static NCNN_FORCEINLINE int _mm_movemask_epi8(v16i8 a)
{
    // Extract sign bits from each byte
    v16u8 signs = (v16u8)__msa_srli_b(a, 7);
    // MSA doesn't have a direct movemask, need to extract and combine
    uint8_t tmp[16];
    __msa_st_b((v16i8)signs, (void*)tmp, 0);
    int mask = 0;
    for (int i = 0; i < 16; i++)
        mask |= (tmp[i] & 1) << i;
    return mask;
}

// _mm_andnot_si128 - bitwise and not ( (~a) & b )
static NCNN_FORCEINLINE v4i32 _mm_andnot_si128(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_and_v((v16u8)__msa_nor_v((v16u8)a, (v16u8)a), (v16u8)b);
}

// _mm_slli_epi16 - shift left 16-bit integers
static NCNN_FORCEINLINE v8i16 _mm_slli_epi16(v8i16 a, int imm)
{
    return (v8i16)__msa_slli_h(a, imm);
}

// _mm_srli_epi16 - shift right logical 16-bit integers
static NCNN_FORCEINLINE v8i16 _mm_srli_epi16(v8i16 a, int imm)
{
    return (v8i16)__msa_srli_h(a, imm);
}

// _mm_srai_epi32 - shift right arithmetic 32-bit integers
static NCNN_FORCEINLINE v4i32 _mm_srai_epi32(v4i32 a, int imm)
{
    return (v4i32)__msa_srai_w(a, imm);
}

// _mm_cmplt_epi32 - less than compare
static NCNN_FORCEINLINE v4i32 _mm_cmplt_epi32(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_clt_s_w(a, b);
}

// _mm_cmpeq_epi32 - equality compare for 32-bit integers
static NCNN_FORCEINLINE v4i32 _mm_cmpeq_epi32(v4i32 a, v4i32 b)
{
    return (v4i32)__msa_ceq_w(a, b);
}

// _mm_cvtsi128_si32 - extract lowest 32-bit integer
static NCNN_FORCEINLINE int _mm_cvtsi128_si32(v4i32 a)
{
    return __msa_copy_s_w(a, 0);
}

// _mm_cvtsi32_si128 - set single 32-bit integer
static NCNN_FORCEINLINE v4i32 _mm_cvtsi32_si128(int a)
{
    return __msa_fill_w(a);
}

// _mm_setr_epi32 - set 4 32-bit integers in reverse order
static NCNN_FORCEINLINE v4i32 _mm_setr_epi32(int e0, int e1, int e2, int e3)
{
    int tmp[4] = {e0, e1, e2, e3};
    return __msa_ld_w(tmp, 0);
}

// _mm_shuffle_epi32 - shuffle 32-bit integers within 128-bit
static NCNN_FORCEINLINE v4i32 _mm_shuffle_epi32(v4i32 a, int imm)
{
    return (v4i32)__msa_shf_w(a, imm);
}

// _mm_mul_epu32 - multiply unsigned 32-bit to 64-bit (low 32-bit elements only)
static NCNN_FORCEINLINE v2i64 _mm_mul_epu32(v4i32 a, v4i32 b)
{
    v2i64 a_lo = (v2i64)__msa_ilvr_w(__msa_fill_w(0), a);
    v2i64 b_lo = (v2i64)__msa_ilvr_w(__msa_fill_w(0), b);
    return __msa_mulv_d(a_lo, b_lo);
}

// _mm_movehl_ps - {b[2],b[3],a[2],a[3]}
static NCNN_FORCEINLINE v4f32 _mm_movehl_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_ilvl_d((v2i64)a, (v2i64)b);
}

// _mm_movelh_ps - {a[0],a[1],b[0],b[1]}
static NCNN_FORCEINLINE v4f32 _mm_movelh_ps(v4f32 a, v4f32 b)
{
    return (v4f32)__msa_ilvr_d((v2i64)b, (v2i64)a);
}

// _mm_blend_epi16 - blend with immediate mask (8 16-bit lanes)
static NCNN_FORCEINLINE v4i32 _mm_blend_epi16(v4i32 a, v4i32 b, int imm)
{
    // Scalar approach: build mask from imm bits
    short mask_arr[8];
    for (int i = 0; i < 8; i++)
        mask_arr[i] = (imm & (1 << i)) ? (short)0xFFFF : (short)0;
    v8i16 mask = (v8i16)__msa_ld_h(mask_arr, 0);
    return (v4i32)__msa_or_v(
        __msa_and_v((v16u8)b, (v16u8)mask),
        __msa_and_v((v16u8)a, (v16u8)__msa_nor_v((v16u8)mask, (v16u8)mask)));
}

// _MM_TRANSPOSE4_PS macro
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    {                                             \
        v4f32 _tmp0, _tmp1, _tmp2, _tmp3;         \
        transpose4x4_ps(row0, row1, row2, row3);  \
    }

#endif // __mips_msa

#endif // MIPS_USABILITY_H
