// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

GroupNorm_loongarch::GroupNorm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void groupnorm_loongarch(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
#if __loongarch_asx
    __m256 _mean8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
#if __loongarch_sx
    __m128 _mean4 = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
    float mean = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _mean8 = __lasx_xvfadd_s(_mean8, _p);
            ptr0 += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _mean4 = __lsx_vfadd_s(_mean4, _p);
            ptr0 += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

    {
#if __loongarch_asx
        mean += __lasx_reduce_fadd_s(_mean8);
#endif // __loongarch_asx
#if __loongarch_sx
        mean += __lsx_reduce_fadd_s(_mean4);
#endif // __loongarch_sx
        mean = mean / (channels * size);
#if __loongarch_asx
        _mean8 = (__m256)__lasx_xvreplfr2vr_s(mean);
#endif // __loongarch_asx
#if __loongarch_sx
        _mean4 = (__m128)__lsx_vreplfr2vr_s(mean);
#endif // __loongarch_sx
    }

#if __loongarch_asx
    __m256 _var8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
#if __loongarch_sx
    __m128 _var4 = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
    float var = 0.f;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _p = __lasx_xvfsub_s(_p, _mean8);
            _var8 = __lasx_xvfmadd_s(_p, _p, _var8);
            ptr0 += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _p = __lsx_vfsub_s(_p, _mean4);
            _var4 = __lsx_vfmadd_s(_p, _p, _var4);
            ptr0 += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

    {
#if __loongarch_asx
        var += __lasx_reduce_fadd_s(_var8);
#endif // __loongarch_asx
#if __loongarch_sx
        var += __lsx_reduce_fadd_s(_var4);
#endif // __loongarch_sx
        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = -mean * var;
#if __loongarch_asx
        _var8 = (__m256)__lasx_xvreplfr2vr_s(var);
        _mean8 = (__m256)__lasx_xvreplfr2vr_s(mean);
#endif // __loongarch_asx
#if __loongarch_sx
        _var4 = (__m128)__lsx_vreplfr2vr_s(var);
        _mean4 = (__m128)__lsx_vreplfr2vr_s(mean);
#endif // __loongarch_sx
    }

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

#if __loongarch_asx
            __m256 _a8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
            __m256 _b8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
#if __loongarch_sx
            __m128 _a4 = (__m128)__lsx_vreplfr2vr_s(0.f);
            __m128 _b4 = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
            float a = 0.f;
            float b = 0.f;

#if __loongarch_asx
            if (elempack == 8)
            {
                __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr + q * elempack, 0);
                __m256 _beta = (__m256)__lasx_xvld(beta_ptr + q * elempack, 0);

                _a8 = __lasx_xvfmul_s(_var8, _gamma);
                _b8 = __lasx_xvfmadd_s(_mean8, _gamma, _beta);
            }
#endif // __loongarch_asx
#if __loongarch_sx
            if (elempack == 4)
            {
                __m128 _gamma = (__m128)__lsx_vld(gamma_ptr + q * elempack, 0);
                __m128 _beta = (__m128)__lsx_vld(beta_ptr + q * elempack, 0);

                _a4 = __lsx_vfmul_s(_var4, _gamma);
                _b4 = __lsx_vfmadd_s(_mean4, _gamma, _beta);
            }
#endif // __loongarch_sx
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
#if __loongarch_asx
                _a8 = (__m256)__lasx_xvreplfr2vr_s(a);
                _b8 = (__m256)__lasx_xvreplfr2vr_s(b);
#endif // __loongarch_asx
#if __loongarch_sx
                _a4 = (__m128)__lsx_vreplfr2vr_s(a);
                _b4 = (__m128)__lsx_vreplfr2vr_s(b);
#endif // __loongarch_sx
            }

            int i = 0;
#if __loongarch_asx
            if (elempack != 4)
            {
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                    _p = __lasx_xvfmadd_s(_p, _a8, _b8);
                    __lasx_xvst(_p, ptr0, 0);
                    ptr0 += 8;
                }
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                _p = __lsx_vfmadd_s(_p, _a4, _b4);
                __lsx_vst(_p, ptr0, 0);
                ptr0 += 4;
            }
#endif // __loongarch_sx
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
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                _p = __lasx_xvfmadd_s(_p, _var8, _mean8);
                __lasx_xvst(_p, ptr0, 0);
                ptr0 += 8;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                _p = __lsx_vfmadd_s(_p, _var4, _mean4);
                __lsx_vst(_p, ptr0, 0);
                ptr0 += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * var + mean;
                ptr0++;
            }
        }
    }
}

int GroupNorm_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = elempack;
#if __loongarch_asx
    if (opt.use_packing_layout && elempack == 8 && channels_g % 8 != 0)
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
#if __loongarch_sx
    if (opt.use_packing_layout && g_elempack == 4 && channels_g % 4 != 0)
        g_elempack = 1;
#endif // __loongarch_sx

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
            groupnorm_loongarch(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, g_elempack, g_elempack, 1);
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
            groupnorm_loongarch(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
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
            groupnorm_loongarch(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
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
int GroupNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
