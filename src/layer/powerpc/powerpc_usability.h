// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef POWERPC_USABILITY_H
#define POWERPC_USABILITY_H

#if __ALTIVEC__
#include <altivec.h>
#endif // __ALTIVEC__

#include <stdint.h>
#include <string.h>

#include "platform.h"

namespace ncnn {

typedef union
{
    int32_t i;
    uint32_t u;
    float f;
} FloatInt;

} // namespace ncnn

#if __ALTIVEC__

typedef __vector float v4f32;
typedef __vector signed int v4i32;
typedef __vector unsigned int v4u32;
typedef __vector signed short v8i16;
typedef __vector unsigned short v8u16;
typedef __vector signed char v16i8;
typedef __vector unsigned char v16u8;
typedef __vector __bool int v4b32;
typedef __vector __bool short v8b16;

/* declare some ppc constants with union */
#define _POWERPC_FLOAT_CONST(Name, Val) \
    static const ncnn::FloatInt Name = {.f = Val}

// plain multiply
#if __VSX__
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_mul(v4f32 _a, v4f32 _b)
{
    return vec_mul(_a, _b);
}
#else
// vec_mul maps to vsx xvmulsp and is unavailable on plain vmx, vmaddfp with zero operand is the canonical vmx multiply
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_mul(v4f32 _a, v4f32 _b)
{
    return vec_madd(_a, _b, (v4f32)vec_splats(0.f));
}
#endif

// powerpc altivec has no vector division and square root instruction
// the reciprocal and reciprocal square root estimates are refined by newton iterations
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_recip(v4f32 _x)
{
    // newton-raphson: y = y * (2 - x * y)
    v4f32 _y = vec_re(_x);
    v4f32 _t = vec_nmsub(_x, _y, vec_splats(2.f));
    return __ncnn_altivec_mul(_y, _t);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_recip2(v4f32 _x)
{
    // two refinements, about 1 ulp
    v4f32 _y = vec_re(_x);
    v4f32 _t = vec_nmsub(_x, _y, vec_splats(2.f));
    _y = __ncnn_altivec_mul(_y, _t);
    _t = vec_nmsub(_x, _y, vec_splats(2.f));
    return __ncnn_altivec_mul(_y, _t);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_div(v4f32 _a, v4f32 _b)
{
#if __VSX__
    return vec_div(_a, _b);
#else
    return __ncnn_altivec_mul(_a, __ncnn_altivec_recip(_b));
#endif
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_div2(v4f32 _a, v4f32 _b)
{
#if __VSX__
    return vec_div(_a, _b);
#else
    return __ncnn_altivec_mul(_a, __ncnn_altivec_recip2(_b));
#endif
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_rsqrt(v4f32 _x)
{
    // newton-raphson: y = y * (1.5 - 0.5 * x * y * y)
    v4f32 _y = vec_rsqrte(_x);
    v4f32 _h = __ncnn_altivec_mul(_x, vec_splats(0.5f));
    v4f32 _t = vec_nmsub(_h, __ncnn_altivec_mul(_y, _y), vec_splats(1.5f));
    return __ncnn_altivec_mul(_y, _t);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_rsqrt2(v4f32 _x)
{
    v4f32 _y = vec_rsqrte(_x);
    v4f32 _h = __ncnn_altivec_mul(_x, vec_splats(0.5f));
    v4f32 _t = vec_nmsub(_h, __ncnn_altivec_mul(_y, _y), vec_splats(1.5f));
    _y = __ncnn_altivec_mul(_y, _t);
    _t = vec_nmsub(_h, __ncnn_altivec_mul(_y, _y), vec_splats(1.5f));
    return __ncnn_altivec_mul(_y, _t);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_sqrt(v4f32 _x)
{
#if __VSX__
    return vec_sqrt(_x);
#else
    return __ncnn_altivec_mul(_x, __ncnn_altivec_rsqrt(_x));
#endif
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_sqrt2(v4f32 _x)
{
    return __ncnn_altivec_mul(_x, __ncnn_altivec_rsqrt2(_x));
}

// a + b * c
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_fmadd(v4f32 _a, v4f32 _b, v4f32 _c)
{
    return vec_madd(_b, _c, _a);
}

// a - b * c
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_fmsub(v4f32 _a, v4f32 _b, v4f32 _c)
{
    return vec_nmsub(_b, _c, _a);
}

// -(a * b - c)
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_fnmsub(v4f32 _a, v4f32 _b, v4f32 _c)
{
    return vec_nmsub(_a, _b, _c);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_abs(v4f32 _x)
{
    return vec_abs(_x);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_neg(v4f32 _x)
{
    return vec_sub((v4f32)vec_splats(0.f), _x);
}

// x > y ? 0xffffffff : 0
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_cmpgt(v4f32 _x, v4f32 _y)
{
    return (v4i32)vec_cmpgt(_x, _y);
}

// x >= y ? 0xffffffff : 0
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_cmpge(v4f32 _x, v4f32 _y)
{
    return (v4i32)vec_cmpge(_x, _y);
}

// x < y ? 0xffffffff : 0
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_cmplt(v4f32 _x, v4f32 _y)
{
    return (v4i32)vec_cmpgt(_y, _x);
}

// x <= y ? 0xffffffff : 0
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_cmple(v4f32 _x, v4f32 _y)
{
    return (v4i32)vec_cmpge(_y, _x);
}

// x == y ? 0xffffffff : 0
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_cmpeq(v4f32 _x, v4f32 _y)
{
    return (v4i32)vec_cmpeq(_x, _y);
}

// select by bitmask of all ones and zeros
// _mask != 0 ? _a : _b
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_sel(v4i32 _mask, v4f32 _a, v4f32 _b)
{
    return vec_sel(_b, _a, (v4b32)_mask);
}

static NCNN_FORCEINLINE v4i32 __ncnn_altivec_and(v4i32 _a, v4i32 _b)
{
    return (v4i32)vec_and((v4u32)_a, (v4u32)_b);
}

static NCNN_FORCEINLINE v4i32 __ncnn_altivec_or(v4i32 _a, v4i32 _b)
{
    return (v4i32)vec_or((v4u32)_a, (v4u32)_b);
}

static NCNN_FORCEINLINE v4i32 __ncnn_altivec_xor(v4i32 _a, v4i32 _b)
{
    return (v4i32)vec_xor((v4u32)_a, (v4u32)_b);
}

static NCNN_FORCEINLINE v4i32 __ncnn_altivec_not(v4i32 _a)
{
    return (v4i32)vec_nor((v4u32)_a, (v4u32)_a);
}

// clear sign bit
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_clear_sign(v4f32 _a)
{
    return vec_and(_a, (v4f32)vec_splats(0x7fffffffu));
}

// flip sign bit
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_flip_sign(v4f32 _a)
{
    return vec_xor(_a, (v4f32)vec_splats(0x80000000u));
}

// copy sign of _sign to _mag
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_copysign(v4f32 _mag, v4f32 _sign)
{
    return vec_cpsgn(_sign, _mag);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_fill_f32(float _v)
{
    return vec_splats(_v);
}

// round to nearest even
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_round(v4f32 _x)
{
    return vec_round(_x);
}

// truncate toward zero
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_trunc(v4f32 _x)
{
    return vec_trunc(_x);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_floor(v4f32 _x)
{
    return vec_floor(_x);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_ceil(v4f32 _x)
{
    return vec_ceil(_x);
}

// float -> int
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_ftrunc_s(v4f32 _x)
{
    return vec_cts(vec_trunc(_x), 0);
}

// float -> int, round to nearest even
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_ftint_s(v4f32 _x)
{
    return vec_cts(vec_round(_x), 0);
}

// int -> float
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_ffint_s(v4i32 _x)
{
    return vec_ctf(_x, 0);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_shl(v4f32 _x, int _n)
{
    v4u32 _s = vec_splats((unsigned int)_n);
    return (v4f32)vec_sl((v4u32)_x, _s);
}

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_shr(v4f32 _x, int _n)
{
    v4u32 _s = vec_splats((unsigned int)_n);
    return (v4f32)vec_sr((v4u32)_x, _s);
}

// reduce to first lane
static NCNN_FORCEINLINE float __ncnn_altivec_reduce_add(v4f32 _x)
{
    v4f32 _s = vec_sld(_x, _x, 8);
    _x = vec_add(_x, _s);
    _s = vec_sld(_x, _x, 4);
    _x = vec_add(_x, _s);
    return vec_extract(_x, 0);
}

static NCNN_FORCEINLINE float __ncnn_altivec_reduce_max(v4f32 _x)
{
    v4f32 _s = vec_sld(_x, _x, 8);
    _x = vec_max(_x, _s);
    _s = vec_sld(_x, _x, 4);
    _x = vec_max(_x, _s);
    return vec_extract(_x, 0);
}

static NCNN_FORCEINLINE float __ncnn_altivec_reduce_min(v4f32 _x)
{
    v4f32 _s = vec_sld(_x, _x, 8);
    _x = vec_min(_x, _s);
    _s = vec_sld(_x, _x, 4);
    _x = vec_min(_x, _s);
    return vec_extract(_x, 0);
}

static NCNN_FORCEINLINE int __ncnn_altivec_reduce_add_i(v4i32 _x)
{
    v4i32 _s = (v4i32)vec_sld((v4u32)_x, (v4u32)_x, 8);
    _x = (v4i32)vec_add((v4u32)_x, (v4u32)_s);
    _s = (v4i32)vec_sld((v4u32)_x, (v4u32)_x, 4);
    _x = (v4i32)vec_add((v4u32)_x, (v4u32)_s);
    return vec_extract(_x, 0);
}

// horizontal pairwise add
// {a0,a1,a2,a3} -> {a0+a1, a2+a3, a0+a1, a2+a3}
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_hadd(v4f32 _x)
{
    v4f32 _s = vec_sld(_x, _x, 4);
    return vec_add(_x, _s);
}

// symmetric permutes
// out[i] = in[perm[i]]
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_permute(v4f32 _x, const v16u8& _perm)
{
    return vec_perm(_x, _x, _perm);
}

// {a0,a1,a2,a3} -> {a0,a2,a0,a2}  (broadcast lane 0 and 2)
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_even_lanes(v4f32 _x)
{
    const v16u8 _perm = {0, 1, 2, 3, 8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11};
    return vec_perm(_x, _x, _perm);
}

// {a0,a1,a2,a3} -> {a1,a3,a1,a3}  (broadcast lane 1 and 3)
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_odd_lanes(v4f32 _x)
{
    const v16u8 _perm = {4, 5, 6, 7, 12, 13, 14, 15, 4, 5, 6, 7, 12, 13, 14, 15};
    return vec_perm(_x, _x, _perm);
}

// {a0,a1,a2,a3} -> {a0,a1,a0,a1}
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_low_lanes(v4f32 _x)
{
    const v16u8 _perm = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    return vec_perm(_x, _x, _perm);
}

// {a0,a1,a2,a3} -> {a2,a3,a2,a3}
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_high_lanes(v4f32 _x)
{
    const v16u8 _perm = {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15};
    return vec_perm(_x, _x, _perm);
}

// {a0,a1,a2,a3} -> {a0,a3,a2,a1}
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_reverse234(v4f32 _x)
{
    const v16u8 _perm = {0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7};
    return vec_perm(_x, _x, _perm);
}

// transpose 4x4 block of float
static NCNN_FORCEINLINE void transpose4x4_ps(v4f32& _r0, v4f32& _r1, v4f32& _r2, v4f32& _r3)
{
    v4f32 _tmp0 = vec_mergeh(_r0, _r1);
    v4f32 _tmp1 = vec_mergel(_r0, _r1);
    v4f32 _tmp2 = vec_mergeh(_r2, _r3);
    v4f32 _tmp3 = vec_mergel(_r2, _r3);

    // concatenate the low / high 64bit halves of two vectors
    // vec_mergeh() interleaves elements, so a byte permute is required here
    const v16u8 _perm_lo = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    const v16u8 _perm_hi = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

    _r0 = vec_perm(_tmp0, _tmp2, _perm_lo);
    _r1 = vec_perm(_tmp0, _tmp2, _perm_hi);
    _r2 = vec_perm(_tmp1, _tmp3, _perm_lo);
    _r3 = vec_perm(_tmp1, _tmp3, _perm_hi);
}

// transpose 4x4 block of int32
static NCNN_FORCEINLINE void transpose4x4_epi32(v4i32& _r0, v4i32& _r1, v4i32& _r2, v4i32& _r3)
{
    v4i32 _tmp0 = (v4i32)vec_mergeh((v4u32)_r0, (v4u32)_r1);
    v4i32 _tmp1 = (v4i32)vec_mergel((v4u32)_r0, (v4u32)_r1);
    v4i32 _tmp2 = (v4i32)vec_mergeh((v4u32)_r2, (v4u32)_r3);
    v4i32 _tmp3 = (v4i32)vec_mergel((v4u32)_r2, (v4u32)_r3);

    const v16u8 _perm_lo = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    const v16u8 _perm_hi = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

    _r0 = (v4i32)vec_perm((v4u32)_tmp0, (v4u32)_tmp2, _perm_lo);
    _r1 = (v4i32)vec_perm((v4u32)_tmp0, (v4u32)_tmp2, _perm_hi);
    _r2 = (v4i32)vec_perm((v4u32)_tmp1, (v4u32)_tmp3, _perm_lo);
    _r3 = (v4i32)vec_perm((v4u32)_tmp1, (v4u32)_tmp3, _perm_hi);
}

// unaligned load/store
// altivec vec_ld/vec_st require 16-byte alignment, and unaligned vec_st traps on some cpus (e.g. cell)
// so use memcpy which gcc expands into a branchless ld/std + aligned lvx/stvx pair
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_load(const float* _ptr)
{
#if __VSX__
    // vsx supports natural alignment loads
    return vec_vsx_ld(0, _ptr);
#else
    v4f32 _v;
    memcpy(&_v, _ptr, sizeof(v4f32));
    return _v;
#endif
}

static NCNN_FORCEINLINE void __ncnn_altivec_store(float* _ptr, const v4f32& _v)
{
#if __VSX__
    vec_vsx_st(_v, 0, _ptr);
#else
    memcpy(_ptr, &_v, sizeof(v4f32));
#endif
}

// ---- int8 quantized inference helpers ----

// dot product accumulate, unsigned activations x signed weights -> 4 x int32 (vmsummbm)
// this is the int8 workhorse, 16 mac per instruction
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_msum_u8_s8(v16u8 _a_u8, v16i8 _b_s8, v4i32 _acc)
{
    return vec_msum(_b_s8, _a_u8, _acc);
}

// signed int16 x signed int16 -> 4 x int32 (vmsumshm)
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_msum_s16(v8i16 _a, v8i16 _b, v4i32 _acc)
{
    return vec_msum(_a, _b, _acc);
}

// round half away from zero, then truncate toward zero to int32
static NCNN_FORCEINLINE v4i32 __ncnn_altivec_round_away_i32(v4f32 _v)
{
    v4f32 _p5 = __ncnn_altivec_copysign(vec_splats(0.5f), _v);
    return vec_cts(vec_trunc(vec_add(_v, _p5)), 0);
}

// saturate 4 int32 to int8, result in the low 4 bytes of the returned vector
static NCNN_FORCEINLINE v16i8 __ncnn_altivec_i32_to_i8(v4i32 _v, int _lo, int _hi)
{
    _v = vec_min(_v, vec_splats(_hi));
    _v = vec_max(_v, vec_splats(_lo));
    v8i16 _h = vec_pack(_v, _v);
    return vec_pack(_h, _h);
}

// saturate 8 int32 (two vectors) to 8 int8 written to _out in ascending element order
// note: written through a byte buffer on purpose, packing the 8 bytes into an int64 return value
// would come out in the wrong byte order on big endian
static NCNN_FORCEINLINE void __ncnn_altivec_i32_to_i8x8(signed char* _out, v4i32 _lo32, v4i32 _hi32, int _lo, int _hi)
{
    _lo32 = vec_min(_lo32, vec_splats(_hi));
    _lo32 = vec_max(_lo32, vec_splats(_lo));
    _hi32 = vec_min(_hi32, vec_splats(_hi));
    _hi32 = vec_max(_hi32, vec_splats(_lo));

    v8i16 _h = vec_pack(_lo32, _hi32);
    v16i8 _b = vec_pack(_h, _h);

    memcpy(_out, &_b, 8);
}

// ---- bfloat16 storage helpers ----
// bfloat16 is the high 16 bit of float32, no hardware support required

static NCNN_FORCEINLINE v4f32 __ncnn_altivec_bf16_to_f32(const unsigned short* _ptr)
{
    // load 4 bf16 (8 bytes) into the low lanes and widen
    v8u16 _v = (v8u16)vec_splats((unsigned int)0);
    memcpy(&_v, _ptr, 8);
    return (v4f32)vec_mergeh(_v, (v8u16)vec_splats((unsigned int)0));
}

static NCNN_FORCEINLINE void __ncnn_altivec_f32_to_bf16(unsigned short* _ptr, const v4f32& _v)
{
    v4u32 _hi = vec_sr((v4u32)_v, (v4u32)vec_splats(16u));
    v8u16 _p = (v8u16)vec_pack((v4i32)_hi, vec_splats(0));
    memcpy(_ptr, &_p, 8);
}

// ---- int8 <-> float32 ----

// int8 -> float32 (4 lanes)
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_int8_to_float(const signed char* _ptr)
{
    float _v[4] = {(float)_ptr[0], (float)_ptr[1], (float)_ptr[2], (float)_ptr[3]};
    return __ncnn_altivec_load(_v);
}

static NCNN_FORCEINLINE signed char float2int8(float v)
{
    int int32 = (int)__builtin_roundf(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static NCNN_FORCEINLINE v16i8 float2int8(v4f32 _v)
{
    // round to nearest even then saturate to int8 range
    v4i32 _i32 = __ncnn_altivec_ftint_s(_v);
    _i32 = vec_min(_i32, vec_splats(127));
    _i32 = vec_max(_i32, vec_splats(-127));

    v8i16 _i16 = vec_pack(_i32, _i32);
    return vec_pack(_i16, _i16);
}

static NCNN_FORCEINLINE int64_t float2int8(v4f32 _vlow, v4f32 _vhigh)
{
    v4i32 _low = __ncnn_altivec_ftint_s(_vlow);
    v4i32 _high = __ncnn_altivec_ftint_s(_vhigh);
    _low = vec_min(_low, vec_splats(127));
    _low = vec_max(_low, vec_splats(-127));
    _high = vec_min(_high, vec_splats(127));
    _high = vec_max(_high, vec_splats(-127));

    v8i16 _i16 = vec_pack(_low, _high);
    v16i8 _i8 = vec_pack(_i16, _i16);

    // low 32 bit holds the first four elements, high 32 bit the last four
    int32_t _lo = vec_extract((v4i32)_i8, 0);
    int32_t _hi = vec_extract((v4i32)_i8, 1);
    return (int64_t)(((uint64_t)(uint32_t)_hi << 32) | (uint32_t)_lo);
}

static NCNN_FORCEINLINE v16i8 float2int8relu(v4f32 _v)
{
    v4i32 _i32 = __ncnn_altivec_ftint_s(_v);
    _i32 = vec_min(_i32, vec_splats(127));
    _i32 = vec_max(_i32, vec_splats(0));

    v8i16 _i16 = vec_pack(_i32, _i32);
    return vec_pack(_i16, _i16);
}

static NCNN_FORCEINLINE int64_t float2int8relu(v4f32 _vlow, v4f32 _vhigh)
{
    v4i32 _low = __ncnn_altivec_ftint_s(_vlow);
    v4i32 _high = __ncnn_altivec_ftint_s(_vhigh);
    _low = vec_min(_low, vec_splats(127));
    _low = vec_max(_low, vec_splats(0));
    _high = vec_min(_high, vec_splats(127));
    _high = vec_max(_high, vec_splats(0));

    v8i16 _i16 = vec_pack(_low, _high);
    v16i8 _i8 = vec_pack(_i16, _i16);

    int32_t _lo = vec_extract((v4i32)_i8, 0);
    int32_t _hi = vec_extract((v4i32)_i8, 1);
    return (int64_t)(((uint64_t)(uint32_t)_hi << 32) | (uint32_t)_lo);
}

static NCNN_FORCEINLINE v16i8 float2int8leakyrelu(v4f32 _v, v4f32 _slope)
{
    v4f32 _v_leaky = __ncnn_altivec_mul(_v, _slope);
    v4f32 _m = vec_max(_v, _v_leaky);
    return float2int8(_m);
}

static NCNN_FORCEINLINE int64_t float2int8leakyrelu(v4f32 _vlow, v4f32 _vhigh, v4f32 _slope)
{
    v4f32 _low = vec_max(_vlow, __ncnn_altivec_mul(_vlow, _slope));
    v4f32 _high = vec_max(_vhigh, __ncnn_altivec_mul(_vhigh, _slope));
    return float2int8(_low, _high);
}

#endif // __ALTIVEC__

#endif // POWERPC_USABILITY_H
