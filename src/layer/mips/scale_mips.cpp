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

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    const int dims = bottom_top_blob.dims;

    const float* scale = scale_blob;
    const float* bias = bias_data;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            if (bias_term)
            {
                v4f32 _s = (v4f32)__msa_ld_w(scale, 0);
                v4f32 _bias = (v4f32)__msa_ld_w(bias, 0);
                int i = 0;
                for (; i + 3 < w; i += 4)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;
                    v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                    _p = __msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                }
                for (; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;
                    v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                    _p = __msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                }
            }
            else
            {
                v4f32 _s = (v4f32)__msa_ld_w(scale, 0);
                int i = 0;
                for (; i + 3 < w; i += 4)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;
                    v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                }
                for (; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;
                    v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
                }
            }
            return 0;
        }

        if (dims == 2)
        {
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                    int j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + j * 4, 0));
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + j * 4, 0);
                    }
                    for (; j < w; j++)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + j * 4, 0));
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + j * 4, 0);
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    int j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + j * 4, 0));
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + j * 4, 0);
                    }
                    for (; j < w; j++)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + j * 4, 0));
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + j * 4, 0);
                    }
                }
            }
            return 0;
        }

        if (dims == 3)
        {
            const int size = w * h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);
                    int i = 0;
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + i * 4, 0));
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + i * 4, 0);
                    }
                    for (; i < size; i++)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + i * 4, 0));
                        _p = __msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + i * 4, 0);
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    int i = 0;
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + i * 4, 0));
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + i * 4, 0);
                    }
                    for (; i < size; i++)
                    {
                        v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr + i * 4, 0));
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)float2bfloat_msa(_p), ptr + i * 4, 0);
                    }
                }
            }

            return 0;
        }
    }
#endif // __mips_msa

    // per-element or elempack == 1 path
    {
        Mat bottom_top_blob_unpacked = bottom_top_blob;
        Mat scale_blob_unpacked = scale_blob;
        if (elempack != 1)
        {
            Option opt_unpack = opt;
            opt_unpack.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt_unpack);
            if (bottom_top_blob_unpacked.empty())
                return -100;
            if (scale_blob.elempack != 1)
            {
                Mat scale_blob_unpacked2;
                convert_packing(scale_blob, scale_blob_unpacked2, 1, opt_unpack);
                if (scale_blob_unpacked2.empty())
                    return -100;
                scale_blob_unpacked = scale_blob_unpacked2;
            }
        }

        int size = bottom_top_blob_unpacked.total();

        if (bias_term)
        {
            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                unsigned short* ptr = bottom_top_blob_unpacked.row<unsigned short>(0) + i;
                const float* sptr = (const float*)scale_blob_unpacked + i;
                const float* bptr = (const float*)bias_data + i;
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                v4f32 _s = (v4f32)__msa_ld_w(sptr, 0);
                v4f32 _b = (v4f32)__msa_ld_w(bptr, 0);
                _p = __msa_fmadd_w(_b, _p, _s);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(((unsigned short*)bottom_top_blob_unpacked)[i]);
                v = v * ((const float*)scale_blob_unpacked)[i] + ((const float*)bias_data)[i];
                ((unsigned short*)bottom_top_blob_unpacked)[i] = float32_to_bfloat16(v);
            }
        }
        else
        {
            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                unsigned short* ptr = bottom_top_blob_unpacked.row<unsigned short>(0) + i;
                const float* sptr = (const float*)scale_blob_unpacked + i;
                v4f32 _p = bfloat2float_msa((v4i32)__msa_ld_w(ptr, 0));
                v4f32 _s = (v4f32)__msa_ld_w(sptr, 0);
                _p = __msa_fmul_w(_p, _s);
                __msa_st_w((v4i32)float2bfloat_msa(_p), ptr, 0);
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(((unsigned short*)bottom_top_blob_unpacked)[i]);
                v = v * ((const float*)scale_blob_unpacked)[i];
                ((unsigned short*)bottom_top_blob_unpacked)[i] = float32_to_bfloat16(v);
            }
        }

        if (elempack != 1)
        {
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_top_blob_unpacked, bottom_top_blob, elempack, opt_pack);
            if (bottom_top_blob.empty())
                return -100;
        }

        return 0;
    }
}
#endif // NCNN_BF16

} // namespace ncnn
