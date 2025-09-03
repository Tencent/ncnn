// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Quantize_mips::Quantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __mips_msa
    v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = (v4f32)__msa_ld_w((const float*)scale_data, 0);
        }
    }
#endif // __mips_msa

    int i = 0;
#if __mips_msa
    for (; i + 7 < size; i += 8)
    {
        __builtin_prefetch(ptr + 32);
        v4f32 _v0 = (v4f32)__msa_ld_w(ptr, 0);
        v4f32 _v1 = (v4f32)__msa_ld_w(ptr + 4, 0);
        _v0 = __msa_fmul_w(_v0, _scale);
        _v1 = __msa_fmul_w(_v1, _scale);
        *((int64_t*)s8ptr) = float2int8(_v0, _v1);
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        v4f32 _v = (v4f32)__msa_ld_w(ptr, 0);
        _v = __msa_fmul_w(_v, _scale);
        v16i8 v = float2int8(_v);
        s8ptr[0] = v[0];
        s8ptr[1] = v[1];
        s8ptr[2] = v[2];
        s8ptr[3] = v[3];
        ptr += 4;
        s8ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __mips_msa
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    v4f32 _scale0 = (v4f32)__msa_fill_w_f32(scale);
    v4f32 _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        _scale0 = (v4f32)__msa_ld_w((const float*)scale_data, 0);
        _scale1 = (v4f32)__msa_ld_w((const float*)scale_data + 4, 0);
    }

    int i = 0;
    for (; i < elemcount; i++)
    {
        v4f32 _v0 = (v4f32)__msa_ld_w(ptr0, 0);
        v4f32 _v1 = (v4f32)__msa_ld_w(ptr1, 0);
        _v0 = __msa_fmul_w(_v0, _scale0);
        _v1 = __msa_fmul_w(_v1, _scale1);
        *((int64_t*)s8ptr) = float2int8(_v0, _v1);
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
    if (scale_data_size > 1)
    {
        _scale = (v4f32)__msa_ld_w((const float*)scale_data, 0);
    }

    int i = 0;
    for (; i < elemcount; i++)
    {
        v4f32 _v = (v4f32)__msa_ld_w(ptr, 0);
        _v = __msa_fmul_w(_v, _scale);
        v16i8 v = float2int8(_v);
        s8ptr0[0] = v[0];
        s8ptr1[0] = v[1];
        s8ptr2[0] = v[2];
        s8ptr3[0] = v[3];
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __mips_msa

int Quantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % 8 == 0 ? 8 : 1;
        }
#endif
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
        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __mips_msa
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
#endif // __mips_msa
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

    if (dims == 3)
    {
        int out_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __mips_msa
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 2);
                const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h);
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

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __mips_msa
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
