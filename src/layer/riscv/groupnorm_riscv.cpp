// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

GroupNorm_riscv::GroupNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    // size is the number of floats per channel, already multiplied by elempack

#if __riscv_vector
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
#endif // __riscv_vector

    float mean = 0.f;
    {
#if __riscv_vector
        vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vlm8);
#endif // __riscv_vector
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __riscv_vector
            for (; i + (int)vlm8 - 1 < size; i += vlm8)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlm8);
                _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vlm8);
                ptr0 += vlm8;
            }
#endif // __riscv_vector
            for (; i < size; i++)
            {
                mean += ptr0[0];
                ptr0++;
            }
        }
#if __riscv_vector
        vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vlm8);
        mean += __riscv_vfmv_f_s_f32m1_f32(_sum0);
#endif // __riscv_vector

        mean = mean / ((float)channels * size);
    }

    float var = 0.f;
    {
#if __riscv_vector
        vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vlm8);
#endif // __riscv_vector
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __riscv_vector
            for (; i + (int)vlm8 - 1 < size; i += vlm8)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlm8);
                _p = __riscv_vfsub_vf_f32m8(_p, mean, vlm8);
                _sqsum = __riscv_vfmacc_vv_f32m8(_sqsum, _p, _p, vlm8);
                ptr0 += vlm8;
            }
#endif // __riscv_vector
            for (; i < size; i++)
            {
                float v = ptr0[0] - mean;
                var += v * v;
                ptr0++;
            }
        }
#if __riscv_vector
        vfloat32m1_t _sqsum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vlm8);
        var += __riscv_vfmv_f_s_f32m1_f32(_sqsum0);
#endif // __riscv_vector
    }

    // x * a + b
    const float a = 1.f / sqrtf(var / ((float)channels * size) + eps);
    const float b = -mean * a;

    for (int q = 0; q < channels; q++)
    {
        float* ptr0 = ptr + cstep * q * elempack;

#if __riscv_vector
        if (elempack != 1)
        {
            // elempack == packn, per-lane affine
            const size_t vl = __riscv_vsetvl_e32m1(elempack);

            vfloat32m1_t _a;
            vfloat32m1_t _b;
            if (gamma_ptr && beta_ptr)
            {
                vfloat32m1_t _gamma = __riscv_vle32_v_f32m1(gamma_ptr + q * elempack, vl);
                vfloat32m1_t _beta = __riscv_vle32_v_f32m1(beta_ptr + q * elempack, vl);
                _a = __riscv_vfmul_vf_f32m1(_gamma, a, vl);
                _b = __riscv_vfmacc_vf_f32m1(_beta, b, _gamma, vl);
            }
            else
            {
                _a = __riscv_vfmv_v_f_f32m1(a, vl);
                _b = __riscv_vfmv_v_f_f32m1(b, vl);
            }

            for (int i = 0; i < size; i += elempack)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0, vl);
                _p = __riscv_vfmadd_vv_f32m1(_p, _a, _b, vl);
                __riscv_vse32_v_f32m1(ptr0, _p, vl);
                ptr0 += elempack;
            }
            continue;
        }
#endif // __riscv_vector

        float aq = a;
        float bq = b;
        if (gamma_ptr && beta_ptr)
        {
            aq = a * gamma_ptr[q];
            bq = b * gamma_ptr[q] + beta_ptr[q];
        }

        int n = size;
#if __riscv_vector
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            _p = __riscv_vfmul_vf_f32m8(_p, aq, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, bq, vl);
            __riscv_vse32_v_f32m8(ptr0, _p, vl);
            ptr0 += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < n; i++)
        {
            ptr0[i] = ptr0[i] * aq + bq;
        }
#endif // __riscv_vector
    }
}

int GroupNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        g_elempack = channels_g % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

    Mat bottom_top_blob_unpacked = bottom_top_blob;
    if (elempack != g_elempack)
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

} // namespace ncnn
