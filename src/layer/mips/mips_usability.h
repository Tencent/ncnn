// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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
    // TODO find a more efficient way
    return _v[0] + _v[1] + _v[2] + _v[3];
}

static NCNN_FORCEINLINE int __msa_reduce_add_w(v4i32 _v)
{
    // TODO find a more efficient way
    return _v[0] + _v[1] + _v[2] + _v[3];
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
#endif // __mips_msa

#endif // MIPS_USABILITY_H
