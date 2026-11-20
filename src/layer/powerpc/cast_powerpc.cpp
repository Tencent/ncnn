// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cast_powerpc.h"

#if __ALTIVEC__
#include <string.h>
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Cast_powerpc::Cast_powerpc()
{
    support_packing = true;
    support_any_packing = true;
}

#if __ALTIVEC__

// fp16 <-> float32 conversion without any fp16 hardware support
// these two helpers are bit-exact with ncnn::float32_to_float16 and
// ncnn::float16_to_float32 in src/mat.cpp

// convert 4 float lanes to 4 fp16 halves
// each result half sits in the low 16 bit of the corresponding 32bit lane
static NCNN_FORCEINLINE v4u32 __ncnn_altivec_f32_to_f16x4(v4f32 _v)
{
    v4u32 _u = (v4u32)_v;

    v4u32 _sign = vec_sr(vec_and(_u, vec_splats(0x80000000u)), vec_splats(16u)); // sign << 15
    v4u32 _exp = vec_and(vec_sr(_u, vec_splats(23u)), vec_splats(0xFFu));
    v4u32 _sig = vec_and(_u, vec_splats(0x7FFFFFu));

    v4i32 _newexp = vec_sub((v4i32)_exp, vec_splats(112));

    // normalized: sign<<15 | ((newexp<<10) + ((sig+0x1000)>>13))
    // the rounding carry of sig+0x1000 propagates into the exponent field exactly like the scalar code
    v4u32 _half_sig = vec_sr(vec_add(_sig, vec_splats(0x1000u)), vec_splats(13u));
    v4u32 _normal = vec_or(_sign, vec_add(vec_sl((v4u32)_newexp, vec_splats(10u)), _half_sig));

    v4u32 _ones = vec_splats(0xFFFFFFFFu);
    v4u32 _not_exp0 = vec_xor((v4u32)vec_cmpeq(_exp, vec_splats(0u)), _ones);
    v4u32 _not_expff = vec_xor((v4u32)vec_cmpeq(_exp, vec_splats(0xFFu)), _ones);
    v4u32 _gt0 = (v4u32)vec_cmpgt(_newexp, vec_splats(0));
    v4u32 _lt31 = (v4u32)vec_cmpgt(vec_splats(31), _newexp);

    v4u32 _valid = vec_and(vec_and(vec_and(_not_exp0, _not_expff), _gt0), _lt31);
    v4u32 _overflow = vec_and(vec_and(_not_exp0, _not_expff), (v4u32)vec_cmpgt(_newexp, vec_splats(30)));
    v4u32 _isnaninf = (v4u32)vec_cmpeq(_exp, vec_splats(0xFFu));
    v4u32 _sig_nonzero = (v4u32)vec_cmpgt((v4i32)_sig, vec_splats(0));

    // zero / subnormal / underflow: sign<<15
    v4u32 _res = _sign;
    // overflow: sign<<15 | 0x7C00
    _res = vec_sel(_res, vec_or(_sign, vec_splats(0x7C00u)), (v4b32)_overflow);
    // infinity or NaN: sign<<15 | 0x7C00 | (sig ? 0x200 : 0)
    v4u32 _naninf = vec_or(vec_or(_sign, vec_splats(0x7C00u)), vec_and(_sig_nonzero, vec_splats(0x200u)));
    _res = vec_sel(_res, _naninf, (v4b32)_isnaninf);
    // normalized
    _res = vec_sel(_res, _normal, (v4b32)_valid);

    return _res;
}

// convert 4 fp16 halves (in the low 16 bit of each 32bit lane) to 4 float lanes
// bit-exact with ncnn::float16_to_float32
static NCNN_FORCEINLINE v4f32 __ncnn_altivec_f16x4_to_f32(v4u32 _h)
{
    v4u32 _sign31 = vec_sl(vec_and(_h, vec_splats(0x8000u)), vec_splats(16u));
    v4u32 _exp = vec_and(vec_sr(_h, vec_splats(10u)), vec_splats(0x1Fu));
    v4u32 _sig = vec_and(_h, vec_splats(0x3FFu));

    // normalized: sign<<31 | ((exp+112)<<23) | (sig<<13)
    v4u32 _normal = vec_or(vec_or(_sign31, vec_sl(vec_add(_exp, vec_splats(112u)), vec_splats(23u))), vec_sl(_sig, vec_splats(13u)));

    // denormal, renormalize without the loop: sig * 2^-24 is exact since sig < 1024
    // xor the sign bit, this preserves -0 for sig==0 like the scalar code
    v4f32 _d = __ncnn_altivec_mul(vec_ctf((v4i32)_sig, 0), vec_splats(5.9604644775390625e-08f /* 2^-24 */));
    v4u32 _dres = vec_xor((v4u32)_d, _sign31);

    // infinity or NaN: sign<<31 | (0xFF<<23) | (sig<<13)
    v4u32 _naninf = vec_or(vec_or(_sign31, vec_splats(0x7F800000u)), vec_sl(_sig, vec_splats(13u)));

    v4u32 _is_exp0 = (v4u32)vec_cmpeq(_exp, vec_splats(0u));
    v4u32 _is_exp1f = (v4u32)vec_cmpeq(_exp, vec_splats(0x1Fu));

    v4u32 _res = _normal;
    _res = vec_sel(_res, _dres, (v4b32)_is_exp0);
    _res = vec_sel(_res, _naninf, (v4b32)_is_exp1f);

    return (v4f32)_res;
}

#endif // __ALTIVEC__

// same semantics as Cast::float32_to_int8 in cast.cpp
static signed char cast_f32_to_i8(float value)
{
    float tmp;
    if (value >= 0.f)
        tmp = value + 0.5f;
    else
        tmp = value - 0.5f;

    if (tmp > 127)
        return 127;
    if (tmp < -128)
        return -128;

    return static_cast<signed char>(tmp);
}

int Cast_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int batch = bottom_blob.n;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        if (type_from == 3)
        {
            return Cast::forward(bottom_blob, top_blob, opt);
        }

        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
        top_blob.create(w, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(w, h, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 4)
        top_blob.create(w, h, d, channels, out_elemsize, elempack, batch, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int size = w * h * d * elempack;

    if (type_from == 1 && type_to == 2)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const float* ptr = bottom_blob.batch(b).channel(q);
            unsigned short* outptr = top_blob.batch(b).channel(q);

            int i = 0;
#if __ALTIVEC__
            for (; i + 7 < size; i += 8)
            {
                v4f32 _p0 = __ncnn_altivec_load(ptr);
                v4f32 _p1 = __ncnn_altivec_load(ptr + 4);
                v8u16 _p = vec_pack(__ncnn_altivec_f32_to_f16x4(_p0), __ncnn_altivec_f32_to_f16x4(_p1));
                memcpy(outptr, &_p, 16);

                ptr += 8;
                outptr += 8;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *outptr = float32_to_float16(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 2 && type_to == 1)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const unsigned short* ptr = bottom_blob.batch(b).channel(q);
            float* outptr = top_blob.batch(b).channel(q);

            int i = 0;
#if __ALTIVEC__
            for (; i + 7 < size; i += 8)
            {
                v8u16 _p;
                memcpy(&_p, ptr, 16);
                v4u32 _h0 = (v4u32)vec_mergeh((v8u16)vec_splats((unsigned short)0), _p);
                v4u32 _h1 = (v4u32)vec_mergel((v8u16)vec_splats((unsigned short)0), _p);
                __ncnn_altivec_store(outptr, __ncnn_altivec_f16x4_to_f32(_h0));
                __ncnn_altivec_store(outptr + 4, __ncnn_altivec_f16x4_to_f32(_h1));

                ptr += 8;
                outptr += 8;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *outptr = float16_to_float32(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 1 && type_to == 3)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const float* ptr = bottom_blob.batch(b).channel(q);
            signed char* outptr = top_blob.batch(b).channel(q);

            int i = 0;
#if __ALTIVEC__
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                // round half away from zero, then saturate, same as Cast::float32_to_int8
                v4i32 _q = __ncnn_altivec_round_away_i32(_p);
                v16i8 _b = __ncnn_altivec_i32_to_i8(_q, -128, 127);
                int32_t _w = vec_extract((v4i32)_b, 0);
                memcpy(outptr, &_w, 4);

                ptr += 4;
                outptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *outptr = cast_f32_to_i8(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 4 && type_to == 1)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const unsigned short* ptr = bottom_blob.batch(b).channel(q);
            float* outptr = top_blob.batch(b).channel(q);

            int i = 0;
#if __ALTIVEC__
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                __ncnn_altivec_store(outptr, _p);
                ptr += 4;
                outptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *outptr = bfloat16_to_float32(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    if (type_from == 1 && type_to == 4)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const float* ptr = bottom_blob.batch(b).channel(q);
            unsigned short* outptr = top_blob.batch(b).channel(q);

            int i = 0;
#if __ALTIVEC__
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                __ncnn_altivec_f32_to_bf16(outptr, _p);
                ptr += 4;
                outptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *outptr = float32_to_bfloat16(*ptr);
                outptr++;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
