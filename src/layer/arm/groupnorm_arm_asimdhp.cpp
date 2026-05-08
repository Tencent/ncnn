// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void groupnorm_fp16s(__fp16* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    float32x4_t _mean0 = vdupq_n_f32(0.f);
    float32x4_t _mean1 = vdupq_n_f32(0.f);
    float mean = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr0);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _mean0 = vaddq_f32(_mean0, _p0);
            _mean1 = vaddq_f32(_mean1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
            _mean0 = vaddq_f32(_mean0, _p);
            ptr0 += 4;
        }
        for (; i < size; i++)
        {
            mean += (float)ptr0[0];
            ptr0++;
        }
    }

    {
        _mean0 = vaddq_f32(_mean0, _mean1);
        mean += vaddvq_f32(_mean0);

        mean = mean / (channels * size);
        _mean0 = vdupq_n_f32(mean);
        _mean1 = _mean0;
    }

    float32x4_t _var0 = vdupq_n_f32(0.f);
    float32x4_t _var1 = vdupq_n_f32(0.f);
    float var = 0.f;
    for (int q = 0; q < channels; q++)
    {
        const __fp16* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr0);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _p0 = vsubq_f32(_p0, _mean0);
            _p1 = vsubq_f32(_p1, _mean1);
            _var0 = vfmaq_f32(_var0, _p0, _p0);
            _var1 = vfmaq_f32(_var1, _p1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
            _p = vsubq_f32(_p, _mean0);
            _var0 = vfmaq_f32(_var0, _p, _p);
            ptr0 += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

    {
        _var0 = vaddq_f32(_var0, _var1);
        var += vaddvq_f32(_var0);

        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = -mean * var;
        _var0 = vdupq_n_f32(var);
        _mean0 = vdupq_n_f32(mean);
        _var1 = _var0;
        _mean1 = _mean0;
    }

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr0 = ptr + cstep * q * elempack;

            float32x4_t _a0 = vdupq_n_f32(0.f);
            float32x4_t _b0 = vdupq_n_f32(0.f);
            float32x4_t _a1 = vdupq_n_f32(0.f);
            float32x4_t _b1 = vdupq_n_f32(0.f);
            float a = 0.f;
            float b = 0.f;

            if (elempack == 8)
            {
                float32x4_t _gamma0 = vld1q_f32(gamma_ptr + q * elempack);
                float32x4_t _gamma1 = vld1q_f32(gamma_ptr + q * elempack + 4);
                float32x4_t _beta0 = vld1q_f32(beta_ptr + q * elempack);
                float32x4_t _beta1 = vld1q_f32(beta_ptr + q * elempack + 4);

                _a0 = vmulq_f32(_var0, _gamma0);
                _a1 = vmulq_f32(_var1, _gamma1);
                _b0 = vfmaq_f32(_beta0, _mean0, _gamma0);
                _b1 = vfmaq_f32(_beta1, _mean1, _gamma1);
            }
            if (elempack == 4)
            {
                float32x4_t _gamma = vld1q_f32(gamma_ptr + q * elempack);
                float32x4_t _beta = vld1q_f32(beta_ptr + q * elempack);

                _a0 = vmulq_f32(_var0, _gamma);
                _b0 = vfmaq_f32(_beta, _mean0, _gamma);
                _a1 = _a0;
                _b1 = _b0;
            }
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma + beta;
                _a0 = vdupq_n_f32(a);
                _b0 = vdupq_n_f32(b);
                _a1 = _a0;
                _b1 = _b0;
            }

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr0);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                _p0 = vfmaq_f32(_b0, _p0, _a0);
                _p1 = vfmaq_f32(_b1, _p1, _a1);
                _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
                vst1q_f16(ptr0, _p);
                ptr0 += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vfmaq_f32(_b0, _p, _a0);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
            }
            for (; i < size; i++)
            {
                *ptr0 = (__fp16)((float)*ptr0 * a + b);
                ptr0++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr0);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                _p0 = vfmaq_f32(_mean0, _p0, _var0);
                _p1 = vfmaq_f32(_mean1, _p1, _var1);
                _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
                vst1q_f16(ptr0, _p);
                ptr0 += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vfmaq_f32(_mean0, _p, _var0);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
            }
            for (; i < size; i++)
            {
                *ptr0 = (__fp16)((float)*ptr0 * var + mean);
                ptr0++;
            }
        }
    }
}

int GroupNorm_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
    if (opt.use_packing_layout)
    {
        if (opt.use_fp16_arithmetic)
            g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
        else
            g_elempack = channels_g % 4 == 0 ? 4 : 1;
    }

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
            groupnorm_fp16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, 1 * g_elempack, g_elempack, 1);
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
            groupnorm_fp16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
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
            groupnorm_fp16s(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob, elempack, opt);
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
