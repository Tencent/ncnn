// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_mips.h"

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

Scale_mips::Scale_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_mips::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

#if __mips_msa
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            int nn_size = size / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = ii * 4;
                __builtin_prefetch(ptr + i + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr + i, 0);
                v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
                v4f32 _bias = (v4f32)__msa_ld_w(bias + i, 0);
                _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                __msa_st_w((v4i32)_p, ptr + i, 0);
            }

            for (int i = nn_size * 4; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
        }
        else
        {
            int nn_size = size / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = ii * 4;
                __builtin_prefetch(ptr + i + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr + i, 0);
                v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
                _p = __msa_fmul_w(_p, _s);
                __msa_st_w((v4i32)_p, ptr + i, 0);
            }

            for (int i = nn_size * 4; i < size; i++)
            {
                ptr[i] *= scale[i];
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            float s = scale[i];
            v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale + i * 4, 0) : __msa_fill_w_f32(s);

            if (bias_term)
            {
                float b = bias[i];
                v4f32 _bias = (elempack == 4) ? (v4f32)__msa_ld_w(bias + i * 4, 0) : __msa_fill_w_f32(b);

                int j = 0;
                for (; j + 3 < size; j += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int j = 0;
                for (; j + 3 < size; j += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr *= s;
                    ptr++;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        const int size = w * bottom_top_blob.h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale[q];
            v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale + q * 4, 0) : __msa_fill_w_f32(s);

            if (bias_term)
            {
                float b = bias[q];
                v4f32 _bias = (elempack == 4) ? (v4f32)__msa_ld_w(bias + q * 4, 0) : __msa_fill_w_f32(b);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr *= s;
                    ptr++;
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    return Scale::forward_inplace(bottom_top_blobs, opt);
}

#if NCNN_BF16
static void scale_bf16s_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
    v4f32 _b = (elempack == 4) ? (v4f32)__msa_ld_w(bias, 0) : (v4f32)__msa_fill_w_f32(bias[0]);
#endif
    float s = scale[0];
    float b = bias[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __ncnn_msa_fmadd_w(_b, _p, _s);
        __msa_storel_d(float2bfloat_msa(_p), ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_msa(unsigned short* ptr, const float* scale, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
#endif
    float s = scale[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __msa_fmul_w(_p, _s);
        __msa_storel_d(float2bfloat_msa(_p), ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}

static void scale_bf16s_per_element_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        v4f32 _b = (v4f32)__msa_ld_w(bias + i, 0);
        _p = __ncnn_msa_fmadd_w(_b, _p, _s);
        __msa_storel_d(float2bfloat_msa(_p), ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
    }
}

static void scale_bf16s_no_bias_per_element_msa(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        _p = __msa_fmul_w(_p, _s);
        __msa_storel_d(float2bfloat_msa(_p), ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
    }
}

int Scale_mips::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    const float* scale = 0;
    Mat scale_fp32;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;

        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }
    const float* bias = bias_data;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            scale_bf16s_per_element_msa(ptr, scale, bias, size, opt.num_threads);
        }
        else
        {
            scale_bf16s_no_bias_per_element_msa(ptr, scale, size, opt.num_threads);
        }

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;

            if (bias_term)
            {
                const float* bptr = bias + i * elempack;
                scale_bf16s_msa(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_msa(ptr, sptr, size, elempack);
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;

            if (bias_term)
            {
                const float* bptr = bias + q * elempack;
                scale_bf16s_msa(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_msa(ptr, sptr, size, elempack);
            }
        }

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
