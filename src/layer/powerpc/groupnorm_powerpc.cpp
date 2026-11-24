// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

GroupNorm_powerpc::GroupNorm_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
static void groupnorm_powerpc(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    v4f32 _sum = vec_splats(0.f);
    float sum = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = __ncnn_altivec_load(ptr0);
            _sum = vec_add(_sum, _p);
            ptr0 += 4;
        }
        for (; i < size; i++)
        {
            sum += ptr0[0];
            ptr0++;
        }
    }

    sum += __ncnn_altivec_reduce_add(_sum);
    float mean = sum / (channels * size);
    v4f32 _mean = vec_splats(mean);

    v4f32 _varsum = vec_splats(0.f);
    float varsum = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = __ncnn_altivec_load(ptr0);
            _p = vec_sub(_p, _mean);
            _varsum = vec_madd(_p, _p, _varsum);
            ptr0 += 4;
        }
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            varsum += v * v;
            ptr0++;
        }
    }

    varsum += __ncnn_altivec_reduce_add(_varsum);

    float var = 1.f / sqrtf(varsum / (channels * size) + eps);
    mean = -mean * var;
    _mean = vec_splats(mean);

    v4f32 _var = vec_splats(var);

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

            v4f32 _a = vec_splats(0.f);
            v4f32 _b = vec_splats(0.f);
            float a = 0.f;
            float b = 0.f;

            if (elempack == 4)
            {
                v4f32 _gamma = __ncnn_altivec_load(gamma_ptr + q * elempack);
                v4f32 _beta = __ncnn_altivec_load(beta_ptr + q * elempack);

                _a = __ncnn_altivec_mul(_var, _gamma);
                _b = vec_madd(_mean, _gamma, _beta);
            }
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
                _a = vec_splats(a);
                _b = vec_splats(b);
            }

            int i = 0;
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr0 + 16);

                v4f32 _p = __ncnn_altivec_load(ptr0);
                _p = vec_madd(_p, _a, _b);
                __ncnn_altivec_store(ptr0, _p);
                ptr0 += 4;
            }
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
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr0 + 16);

                v4f32 _p = __ncnn_altivec_load(ptr0);
                _p = vec_madd(_p, _var, _mean);
                __ncnn_altivec_store(ptr0, _p);
                ptr0 += 4;
            }
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * var + mean;
                ptr0++;
            }
        }
    }
}
#endif // __ALTIVEC__

int GroupNorm_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elembits() != 32)
    {
        // int8 / bf16 fall back to the generic implementation
        const int elempack = bottom_top_blob.elempack;
        if (elempack != 1)
        {
            Mat bottom_top_blob_unpacked;
            Option opt_unpacked = opt;
            opt_unpacked.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt_unpacked);
            if (bottom_top_blob_unpacked.empty())
                return -100;

            GroupNorm::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);

            if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
                if (bottom_top_blob.empty())
                    return -100;
            }
        }
        else
        {
            return GroupNorm::forward_inplace(bottom_top_blob, opt);
        }

        return 0;
    }

#if __ALTIVEC__
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = elempack;
    if (opt.use_packing_layout && elempack == 4 && channels_g % 4 != 0)
        g_elempack = 1;

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
            groupnorm_powerpc(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, g_elempack, g_elempack, 1);
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
            groupnorm_powerpc(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
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
            groupnorm_powerpc(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_packed, bottom_top_blob, elempack, opt);
        if (bottom_top_blob.empty())
            return -100;
    }

    return 0;
#else
    return GroupNorm::forward_inplace(bottom_top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
