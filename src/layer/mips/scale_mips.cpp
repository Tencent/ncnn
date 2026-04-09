// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_mips.h"

#if __mips_msa
#include <msa.h>
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
    const Mat& scale_blob = bottom_top_blobs[1];

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            const int w = bottom_top_blob.w;

            const float* scale = scale_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _s = (v4f32)__msa_ld_w(scale + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w(bias + i * 4, 0);
                    _p = __msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _s = (v4f32)__msa_ld_w(scale + i * 4, 0);
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }

            return 0;
        }

        if (dims == 2)
        {
            const int w = bottom_top_blob.w;
            const int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }

        if (dims == 3)
        {
            const int size = bottom_top_blob.w * bottom_top_blob.h;
            const int channels = bottom_top_blob.c;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }
    }
#endif // __mips_msa

    return Scale::forward_inplace(bottom_top_blobs, opt);
}

#if NCNN_BF16
int Scale_mips::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    Mat scale_blob_fp32 = scale_blob;
    if (scale_blob.elembits() == 16)
    {
        cast_bfloat16_to_float32(scale_blob, scale_blob_fp32, opt_cast);
        if (scale_blob_fp32.empty())
            return -100;
    }

    std::vector<Mat> bottom_top_blobs_fp32(2);
    bottom_top_blobs_fp32[0] = bottom_top_blob_fp32;
    bottom_top_blobs_fp32[1] = scale_blob_fp32;

    int ret = forward_inplace(bottom_top_blobs_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
