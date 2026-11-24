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

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline)) static vfloat32m8_t
reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_xtheadvector
#endif // __riscv_vector

static void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    // size is the number of floats per channel, already multiplied by elempack

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();

    // reduce sum
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vlm8);
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int n = size / vlm8 * vlm8;
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlm8);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vlm8);
            ptr0 += vlm8;
            n -= vlm8;
        }
        int remain = size % vlm8;
        if (remain > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vl, 0.f);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vlm8);
#else  // __riscv_xtheadvector
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vl);
#endif // __riscv_xtheadvector
        }
    }

    vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vlm8);
    const float mean = __riscv_vfmv_f_s_f32m1_f32(_sum0) / ((float)channels * size);

    // reduce sqsum
    vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, vlm8);
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int n = size / vlm8 * vlm8;
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlm8);
            _p = __riscv_vfsub_vf_f32m8(_p, mean, vlm8);
            _sqsum = __riscv_vfmacc_vv_f32m8(_sqsum, _p, _p, vlm8);
            ptr0 += vlm8;
            n -= vlm8;
        }
        int remain = size % vlm8;
        if (remain > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vl, 0.f);
            _sqsum = __riscv_vfmacc_vv_f32m8(_sqsum, _p, _p, vlm8);
#else  // __riscv_xtheadvector
            _sqsum = __riscv_vfmacc_vv_f32m8_tu(_sqsum, _p, _p, vl);
#endif // __riscv_xtheadvector
        }
    }

    vfloat32m1_t _sqsum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, vlm8);
    const float var = __riscv_vfmv_f_s_f32m1_f32(_sqsum0) / ((float)channels * size);

    const float a = 1.f / sqrtf(var + eps);
    const float b = -mean * a;

    // v = v * a + b
    // v = (v * a + b) * gamma + beta
    //   = v * (a * gamma) + (b * gamma + beta)

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

            vfloat32m8_t _a;
            vfloat32m8_t _b;
            if (elempack == packn)
            {
                // broadcast the packn lanes of gamma and beta
                const size_t vlm1 = __riscv_vsetvlmax_e32m1();

                vfloat32m1_t _gamma = __riscv_vle32_v_f32m1(gamma_ptr + q * elempack, vlm1);
                vfloat32m1_t _beta = __riscv_vle32_v_f32m1(beta_ptr + q * elempack, vlm1);
                vfloat32m1_t _a0 = __riscv_vfmul_vf_f32m1(_gamma, a, vlm1);
                vfloat32m1_t _b0 = __riscv_vfmacc_vf_f32m1(_beta, b, _gamma, vlm1);
                _a = __riscv_vcreate_v_f32m1_f32m8(_a0, _a0, _a0, _a0, _a0, _a0, _a0, _a0);
                _b = __riscv_vcreate_v_f32m1_f32m8(_b0, _b0, _b0, _b0, _b0, _b0, _b0, _b0);
            }
            else // if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                _a = __riscv_vfmv_v_f_f32m8(a * gamma, vlm8);
                _b = __riscv_vfmv_v_f_f32m8(b * gamma + beta, vlm8);
            }

            // full vl chunks keep every chunk aligned with the packn lanes of _a and _b
            int n = size / vlm8 * vlm8;
            while (n > 0)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlm8);
                _p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vlm8);
                __riscv_vse32_v_f32m8(ptr0, _p, vlm8);
                ptr0 += vlm8;
                n -= vlm8;
            }
            int remain = size % vlm8;
            if (remain > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(remain);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vl);
                __riscv_vse32_v_f32m8(ptr0, _p, vl);
            }
        }
    }
    else
    {
        // a and b are the same for every lane, packed or not
        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vlm8);

        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
                _p = __riscv_vfmadd_vf_f32m8(_p, a, _b, vl);
                __riscv_vse32_v_f32m8(ptr0, _p, vl);
                ptr0 += vl;
                n -= vl;
            }
        }
    }
#else  // __riscv_vector
    float sum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;
        for (int i = 0; i < size; i++)
        {
            sum += ptr0[i];
        }
    }

    float mean = sum / (channels * size);

    float sqsum = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;
        for (int i = 0; i < size; i++)
        {
            float v = ptr0[i] - mean;
            sqsum += v * v;
        }
    }

    float var = sqsum / (channels * size);

    float a = 1.f / sqrtf(var + eps);
    float b = -mean * a;

    for (int q = 0; q < channels; q++)
    {
        float* ptr0 = ptr + cstep * q * elempack;

        float aq = a;
        float bq = b;
        if (gamma_ptr && beta_ptr)
        {
            aq = a * gamma_ptr[q];
            bq = b * gamma_ptr[q] + beta_ptr[q];
        }

        for (int i = 0; i < size; i++)
        {
            ptr0[i] = ptr0[i] * aq + bq;
        }
    }
#endif // __riscv_vector
}

int GroupNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    // unpack when a group does not hold whole packs
    const int g_elempack = channels_g % elempack == 0 ? elempack : 1;

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
        if (bottom_top_blob.empty())
            return -100;
    }

    return 0;
}

} // namespace ncnn
