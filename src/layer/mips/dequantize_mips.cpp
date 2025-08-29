// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Dequantize_mips::Dequantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __mips_msa
    v4f32 _scale0 = (v4f32)__msa_fill_w_f32(scale);
    v4f32 _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale0 = (v4f32)__msa_ld_w((const float*)scale_data, 0);
            _scale1 = _scale0;
        }
        if (elempack == 8)
        {
            _scale0 = (v4f32)__msa_ld_w((const float*)scale_data, 0);
            _scale1 = (v4f32)__msa_ld_w((const float*)scale_data + 4, 0);
        }
    }
#endif // __mips_msa

    if (bias_data_size == 0)
    {
        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmul_w(_v0, _scale0);
            _v1 = __msa_fmul_w(_v1, _scale1);
            __msa_st_w((v4i32)_v0, ptr, 0);
            __msa_st_w((v4i32)_v1, ptr + 4, 0);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmul_w(_v, _scale0);
            __msa_st_w((v4i32)_v, ptr, 0);
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __mips_msa
        v4f32 _bias0 = (v4f32)__msa_fill_w_f32(bias);
        v4f32 _bias1 = _bias0;
        if (bias_data_size > 1)
        {
            if (elempack == 4)
            {
                _bias0 = (v4f32)__msa_ld_w((const float*)bias_data, 0);
                _bias1 = _bias0;
            }
            if (elempack == 8)
            {
                _bias0 = (v4f32)__msa_ld_w((const float*)bias_data, 0);
                _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + 4, 0);
            }
        }
#endif // __mips_msa

        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
            _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
            __msa_st_w((v4i32)_v0, ptr, 0);
            __msa_st_w((v4i32)_v1, ptr + 4, 0);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmadd_w(_bias0, _v, _scale0);
            __msa_st_w((v4i32)_v, ptr, 0);
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
}

int Dequantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            float* ptr = (float*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
