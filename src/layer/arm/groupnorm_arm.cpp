// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

GroupNorm_arm::GroupNorm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
#if __ARM_NEON
    float32x4_t _mean = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float mean = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _mean = vaddq_f32(_mean, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

    {
#if __ARM_NEON
#if __aarch64__
        mean += vaddvq_f32(_mean);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_mean), vget_high_f32(_mean));
        _s2 = vpadd_f32(_s2, _s2);
        mean += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        mean = mean / (channels * size);
#if __ARM_NEON
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

#if __ARM_NEON
    float32x4_t _var = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float var = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _p = vsubq_f32(_p, _mean);
            _var = vmlaq_f32(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

    {
#if __ARM_NEON
#if __aarch64__
        var += vaddvq_f32(_var);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_var), vget_high_f32(_var));
        _s2 = vpadd_f32(_s2, _s2);
        var += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = -mean * var;
#if __ARM_NEON
        _var = vdupq_n_f32(var);
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(0.f);
            float32x4_t _b = vdupq_n_f32(0.f);
#endif // __ARM_NEON
            float a = 0.f;
            float b = 0.f;

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _gamma = vld1q_f32(gamma_ptr + q * elempack);
                float32x4_t _beta = vld1q_f32(beta_ptr + q * elempack);

                _a = vmulq_f32(_var, _gamma);
                _b = vmlaq_f32(_beta, _mean, _gamma);
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
#if __ARM_NEON
                _a = vdupq_n_f32(a);
                _b = vdupq_n_f32(b);
#endif // __ARM_NEON
            }

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr0);
                _p = vmlaq_f32(_b, _p, _a);
                vst1q_f32(ptr0, _p);
                ptr0 += 4;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr0);
                _p = vmlaq_f32(_mean, _p, _var);
                vst1q_f32(ptr0, _p);
                ptr0 += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * var + mean;
                ptr0++;
            }
        }
    }
}

int GroupNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

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
static void groupnorm_bf16s(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
#if __ARM_NEON
    float32x4_t _mean = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float mean = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _mean = vaddq_f32(_mean, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }
    }

    {
#if __ARM_NEON
#if __aarch64__
        mean += vaddvq_f32(_mean);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_mean), vget_high_f32(_mean));
        _s2 = vpadd_f32(_s2, _s2);
        mean += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        mean = mean / (channels * size);
#if __ARM_NEON
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

#if __ARM_NEON
    float32x4_t _var = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float var = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _p = vsubq_f32(_p, _mean);
            _var = vmlaq_f32(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]) - mean;
            var += v * v;
            ptr0++;
        }
    }

    {
#if __ARM_NEON
#if __aarch64__
        var += vaddvq_f32(_var);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_var), vget_high_f32(_var));
        _s2 = vpadd_f32(_s2, _s2);
        var += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = -mean * var;
#if __ARM_NEON
        _var = vdupq_n_f32(var);
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr0 = ptr + cstep * q * elempack;

#if __ARM_NEON
            float32x4_t _a = vdupq_n_f32(0.f);
            float32x4_t _b = vdupq_n_f32(0.f);
#endif // __ARM_NEON
            float a = 0.f;
            float b = 0.f;

#if __ARM_NEON
            if (elempack == 4)
            {
                float32x4_t _gamma = vld1q_f32(gamma_ptr + q * elempack);
                float32x4_t _beta = vld1q_f32(beta_ptr + q * elempack);

                _a = vmulq_f32(_var, _gamma);
                _b = vmlaq_f32(_beta, _mean, _gamma);
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
#if __ARM_NEON
                _a = vdupq_n_f32(a);
                _b = vdupq_n_f32(b);
#endif // __ARM_NEON
            }

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_u16(ptr0, float2bfloat(_p));
                ptr0 += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                ptr0++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr0));
                _p = vmlaq_f32(_mean, _p, _var);
                vst1_u16(ptr0, float2bfloat(_p));
                ptr0 += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * var + mean);
                ptr0++;
            }
        }
    }
}

int GroupNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

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
            groupnorm_bf16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, 1 * g_elempack, g_elempack, 1);
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
            groupnorm_bf16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
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
            groupnorm_bf16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
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
