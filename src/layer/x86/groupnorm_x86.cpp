// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "groupnorm_bf16s.h"
#endif

GroupNorm_x86::GroupNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#include "groupnorm_fp32.h"

int GroupNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        g_elempack = channels_g % 16 == 0 ? 16 : channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#elif __AVX__
        g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#else
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    Mat bottom_top_blob_unpacked = bottom_top_blob;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, g_elempack, opt_p);
        if (bottom_top_blob_unpacked.empty())
            return -100;
    }

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, 1 * g_elempack, g_elempack, 1);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob_unpacked.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.row_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob_unpacked.w * bottom_top_blob_unpacked.h * bottom_top_blob_unpacked.d;
        const size_t cstep = bottom_top_blob_unpacked.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.channel_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob, elempack, opt);
    }

    return 0;
}

#if NCNN_BF16
int GroupNorm_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        g_elempack = channels_g % 16 == 0 ? 16 : channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#elif __AVX__
        g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#else
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    Mat bottom_top_blob_unpacked = bottom_top_blob;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, g_elempack, opt_p);
        if (bottom_top_blob_unpacked.empty())
            return -100;
    }

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_bf16s_sse(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, 1 * g_elempack, g_elempack, 1);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob_unpacked.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.row_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_bf16s_sse(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob_unpacked.w * bottom_top_blob_unpacked.h * bottom_top_blob_unpacked.d;
        const size_t cstep = bottom_top_blob_unpacked.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.channel_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm_bf16s_sse(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob, elempack, opt);
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
