// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <algorithm>

namespace ncnn {

Quantize_powerpc::Quantize_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    const float scale0 = scale_data[0];

#if __ALTIVEC__
    v4f32 _scale = vec_splats(scale0);
    if (scale_data_size > 1 && elempack == 4)
    {
        _scale = __ncnn_altivec_load(scale_data);
    }

    int i = 0;
    for (; i + 7 < size; i += 8)
    {
        v4f32 _v0 = __ncnn_altivec_mul(__ncnn_altivec_load(ptr), _scale);
        v4f32 _v1 = __ncnn_altivec_mul(__ncnn_altivec_load(ptr + 4), _scale);
        // big-endian note: compose the 8 int8 via two endian-safe 4-byte stores
        v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
        v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
        memcpy(s8ptr, &_b0, 4);
        memcpy(s8ptr + 4, &_b1, 4);
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        v4f32 _v = __ncnn_altivec_mul(__ncnn_altivec_load(ptr), _scale);
        v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
        memcpy(s8ptr, &_b, 4);
        ptr += 4;
        s8ptr += 4;
    }
    for (; i < size; i++)
    {
        const float scale = scale_data_size == 1 ? scale0 : scale_data[i % elempack];
        *s8ptr = float2int8(*ptr * scale);
        ptr++;
        s8ptr++;
    }
#else
    for (int i = 0; i < size; i++)
    {
        const float scale = scale_data_size == 1 ? scale0 : scale_data[i % elempack];
        *s8ptr = float2int8(*ptr * scale);
        ptr++;
        s8ptr++;
    }
#endif
}

#if __ALTIVEC__
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const float scale0 = scale_data[0];
    v4f32 _scale0 = vec_splats(scale0);
    v4f32 _scale1 = _scale0;
    if (scale_data.w > 1)
    {
        _scale0 = __ncnn_altivec_load(scale_data);
        _scale1 = __ncnn_altivec_load((const float*)scale_data + 4);
    }

    for (int i = 0; i < elemcount; i++)
    {
        v4f32 _v0 = __ncnn_altivec_mul(__ncnn_altivec_load(ptr0), _scale0);
        v4f32 _v1 = __ncnn_altivec_mul(__ncnn_altivec_load(ptr1), _scale1);
        // big-endian note: compose the 8 int8 via two endian-safe 4-byte stores
        v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
        v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
        memcpy(s8ptr, &_b0, 4);
        memcpy(s8ptr + 4, &_b1, 4);
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const float scale0 = scale_data[0];
    v4f32 _scale = vec_splats(scale0);
    if (scale_data.w > 1)
    {
        _scale = __ncnn_altivec_load(scale_data);
    }

    for (int i = 0; i < elemcount; i++)
    {
        v4f32 _v = __ncnn_altivec_mul(__ncnn_altivec_load(ptr), _scale);
        v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
        signed char o[4];
        memcpy(o, &_b, 4);
        s8ptr0[0] = o[0];
        s8ptr1[0] = o[1];
        s8ptr2[0] = o[2];
        s8ptr3[0] = o[3];
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __ALTIVEC__

int Quantize_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        // int8 blobs stay pack1: the int8 consumers (generic int8 conv / innerproduct)
        // expect a pack1 quantized blob, and the packing layer converts on demand
        const int out_elempack = 1;
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        // int8 blobs stay pack1: the int8 consumers (generic int8 conv / innerproduct)
        // expect a pack1 quantized blob, and the packing layer converts on demand
        const int out_elempack = 1;
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ALTIVEC__
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* ptr0 = bottom_blob.row(i * 2);
                const float* ptr1 = bottom_blob.row(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 4);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 4 + 1);
                signed char* s8ptr2 = top_blob.row<signed char>(i * 4 + 2);
                signed char* s8ptr3 = top_blob.row<signed char>(i * 4 + 3);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __ALTIVEC__
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        // int8 blobs stay pack1: the int8 consumers (generic int8 conv / innerproduct)
        // expect a pack1 quantized blob, and the packing layer converts on demand
        const int out_elempack = 1;
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ALTIVEC__
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 2);
                const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h * d);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 4);
                signed char* s8ptr1 = top_blob.channel(q * 4 + 1);
                signed char* s8ptr2 = top_blob.channel(q * 4 + 2);
                signed char* s8ptr3 = top_blob.channel(q * 4 + 3);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h * d);
            }
        }
#endif // __ALTIVEC__
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_q, w * h * d, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
