// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

GroupNorm_mips::GroupNorm_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void groupnorm_mips(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
#if __mips_msa
    v4f32 _mean = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float mean = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _mean = __msa_fadd_w(_mean, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

    {
#if __mips_msa
        mean += __msa_reduce_fadd_w(_mean);
#endif // __mips_msa
        mean = mean / (channels * size);
#if __mips_msa
        _mean = __msa_fill_w_f32(mean);
#endif // __mips_msa
    }

#if __mips_msa
    v4f32 _var = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float var = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _p = __msa_fsub_w(_p, _mean);
            _var = __msa_fmadd_w(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

    {
#if __mips_msa
        var += __msa_reduce_fadd_w(_var);
#endif // __mips_msa
        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = -mean * var;
#if __mips_msa
        _var = __msa_fill_w_f32(var);
        _mean = __msa_fill_w_f32(mean);
#endif // __mips_msa
    }

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

#if __mips_msa
            v4f32 _a = (v4f32)__msa_fill_w(0);
            v4f32 _b = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
            float a = 0.f;
            float b = 0.f;

#if __mips_msa
            if (elempack == 4)
            {
                v4f32 _gamma = (v4f32)__msa_ld_w(gamma_ptr + q * elempack, 0);
                v4f32 _beta = (v4f32)__msa_ld_w(beta_ptr + q * elempack, 0);

                _a = __msa_fmul_w(_var, _gamma);
                _b = __msa_fmadd_w(_beta, _mean, _gamma);
            }
#endif // __mips_msa
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
#if __mips_msa
                _a = __msa_fill_w_f32(a);
                _b = __msa_fill_w_f32(b);
#endif // __mips_msa
            }

            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                _p = __msa_fmadd_w(_b, _p, _a);
                __msa_st_w((v4i32)_p, ptr0, 0);
                ptr0 += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * a + b;
                ptr0++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                _p = __msa_fmadd_w(_mean, _p, _var);
                __msa_st_w((v4i32)_p, ptr0, 0);
                ptr0 += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * var + mean;
                ptr0++;
            }
        }
    }
}

int GroupNorm_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = elempack;
#if __mips_msa
    if (opt.use_packing_layout && elempack == 4 && channels_g % 4 != 0)
        g_elempack = 1;
#endif // __mips_msa

    Mat bottom_top_blob_packed = bottom_top_blob;
    if (elempack != g_elempack)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_top_blob, bottom_top_blob_packed, g_elempack, opt_pack);
        if (bottom_top_blob_packed.empty())
            return -100;
    }

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_packed.range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_mips(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, g_elempack, g_elempack, 1);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob_packed.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_packed.row_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_mips(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob_packed.w * bottom_top_blob_packed.h * bottom_top_blob_packed.d;
        const size_t cstep = bottom_top_blob_packed.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_packed.channel_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_mips(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_packed, bottom_top_blob, elempack, opt);
        if (bottom_top_blob.empty())
            return -100;
    }

    return 0;
}

#if NCNN_BF16
int GroupNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
