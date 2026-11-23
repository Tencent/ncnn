// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

#include <algorithm>

namespace ncnn {

#if __ARM_NEON
static NCNN_FORCEINLINE float normalize_coeff(float ssum, float eps, int eps_mode)
{
    if (eps_mode == 0) // caffe/mxnet
        return 1.f / sqrtf(ssum + eps);

    if (eps_mode == 1) // pytorch
        return 1.f / std::max(sqrtf(ssum), eps);

    // if (eps_mode == 2) // tensorflow
    return 1.f / sqrtf(std::max(ssum, eps));
}

static float square_sum_ps(const float* ptr, int size)
{
    float32x4_t _sum0 = vdupq_n_f32(0.f);
    float32x4_t _sum1 = vdupq_n_f32(0.f);
    float32x4_t _sum2 = vdupq_n_f32(0.f);
    float32x4_t _sum3 = vdupq_n_f32(0.f);

    int i = 0;
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _p2 = vld1q_f32(ptr + 8);
        float32x4_t _p3 = vld1q_f32(ptr + 12);
        _sum0 = vmlaq_f32(_sum0, _p0, _p0);
        _sum1 = vmlaq_f32(_sum1, _p1, _p1);
        _sum2 = vmlaq_f32(_sum2, _p2, _p2);
        _sum3 = vmlaq_f32(_sum3, _p3, _p3);
        ptr += 16;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _sum0 = vmlaq_f32(_sum0, _p, _p);
        ptr += 4;
    }

    _sum0 = vaddq_f32(_sum0, _sum1);
    _sum0 = vaddq_f32(_sum0, _sum2);
    _sum0 = vaddq_f32(_sum0, _sum3);

#if __aarch64__
    float ssum = vaddvq_f32(_sum0);
#else
    float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
    _sum2 = vpadd_f32(_sum2, _sum2);
    float ssum = vget_lane_f32(_sum2, 0);
#endif
    for (; i < size; i++)
    {
        ssum += ptr[0] * ptr[0];
        ptr++;
    }

    return ssum;
}

static void scale_ps(float* ptr, int size, float32x4_t scale)
{
    int i = 0;
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _p2 = vld1q_f32(ptr + 8);
        float32x4_t _p3 = vld1q_f32(ptr + 12);
        _p0 = vmulq_f32(_p0, scale);
        _p1 = vmulq_f32(_p1, scale);
        _p2 = vmulq_f32(_p2, scale);
        _p3 = vmulq_f32(_p3, scale);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);
        vst1q_f32(ptr + 8, _p2);
        vst1q_f32(ptr + 12, _p3);
        ptr += 16;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        _p0 = vmulq_f32(_p0, scale);
        _p1 = vmulq_f32(_p1, scale);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);
        ptr += 8;
    }
#endif // __aarch64__
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _p = vmulq_f32(_p, scale);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }

    const float scale0 = vgetq_lane_f32(scale, 0);
    for (; i < size; i++)
    {
        ptr[0] *= scale0;
        ptr++;
    }
}
#endif // __ARM_NEON

Normalize_arm::Normalize_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int Normalize_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d;

    if (dims == 1 || dims == 2)
    {
        // packing is along the spatial dimensions, not channels
        size *= elempack;
        elempack = 1;
    }

    if (across_spatial && across_channel)
    {
        Mat square_sum_blob;
        square_sum_blob.create(channels, 4u, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            const int n = size * elempack;

#if __ARM_NEON
            ((float*)square_sum_blob)[q] = square_sum_ps(ptr, n);
#else
            float ssum = 0.f;
            for (int i = 0; i < n; i++)
            {
                ssum += ptr[i] * ptr[i];
            }
            ((float*)square_sum_blob)[q] = ssum;
#endif // __ARM_NEON
        }

        float ssum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            ssum += ((float*)square_sum_blob)[q];
        }

        const float a = normalize_coeff(ssum, eps, eps_mode);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * elempack);
            const int n = size * elempack;

#if __ARM_NEON
            float32x4_t _scale = vdupq_n_f32(a * scale_ptr[0]);
            if (!channel_shared && elempack == 4)
                _scale = vmulq_f32(vdupq_n_f32(a), vld1q_f32(scale_ptr));
            scale_ps(ptr, n, _scale);
#else
            const float scale = a * scale_ptr[0];
            for (int i = 0; i < n; i++)
            {
                ptr[i] *= scale;
            }
#endif // __ARM_NEON
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * 4);
                const int n = size * 4;

                float32x4_t _ssum0 = vdupq_n_f32(0.f);
                float32x4_t _ssum1 = vdupq_n_f32(0.f);
                float32x4_t _ssum2 = vdupq_n_f32(0.f);
                float32x4_t _ssum3 = vdupq_n_f32(0.f);
                int i = 0;
                for (; i + 15 < n; i += 16)
                {
                    float32x4_t _p0 = vld1q_f32(ptr + i);
                    float32x4_t _p1 = vld1q_f32(ptr + i + 4);
                    float32x4_t _p2 = vld1q_f32(ptr + i + 8);
                    float32x4_t _p3 = vld1q_f32(ptr + i + 12);
                    _ssum0 = vmlaq_f32(_ssum0, _p0, _p0);
                    _ssum1 = vmlaq_f32(_ssum1, _p1, _p1);
                    _ssum2 = vmlaq_f32(_ssum2, _p2, _p2);
                    _ssum3 = vmlaq_f32(_ssum3, _p3, _p3);
                }
                for (; i < n; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr + i);
                    _ssum0 = vmlaq_f32(_ssum0, _p, _p);
                }

                float32x4_t _ssum = vaddq_f32(vaddq_f32(_ssum0, _ssum1), vaddq_f32(_ssum2, _ssum3));
                const float32x4_t _one = vdupq_n_f32(1.f);
                const float32x4_t _eps_vec = vdupq_n_f32(eps);
                float32x4_t _coeff;
                if (eps_mode == 0)
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vaddq_f32(_ssum, _eps_vec)));
                else if (eps_mode == 1)
                    _coeff = vdivq_f32(_one, vmaxq_f32(vsqrtq_f32(_ssum), _eps_vec));
                else
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vmaxq_f32(_ssum, _eps_vec)));
                float32x4_t _scale = _coeff;
                _scale = vmulq_f32(_scale, channel_shared ? vdupq_n_f32(scale_ptr[0]) : vld1q_f32(scale_ptr));

                scale_ps(ptr, n, _scale);
            }

            return 0;
        }
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * elempack);
            const int n = size * elempack;

#if __ARM_NEON
            float ssum = square_sum_ps(ptr, n);
#else
            float ssum = 0.f;
            for (int i = 0; i < n; i++)
            {
                ssum += ptr[i] * ptr[i];
            }
#endif // __ARM_NEON

            const float a = normalize_coeff(ssum, eps, eps_mode);

#if __ARM_NEON
            scale_ps(ptr, n, vdupq_n_f32(a * scale_ptr[0]));
#else
            const float scale = a * scale_ptr[0];
            for (int i = 0; i < n; i++)
            {
                ptr[i] *= scale;
            }
#endif // __ARM_NEON
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        Mat square_sum_blob;
        square_sum_blob.create(size, 4u, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        float* square_sum = square_sum_blob;

        if (elempack == 4)
        {
#if __ARM_NEON
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                float32x4_t _ssum = vdupq_n_f32(0.f);
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i * 4;
                    float32x4_t _p = vld1q_f32(ptr);
                    _ssum = vmlaq_f32(_ssum, _p, _p);
                }
#if __aarch64__
                square_sum[i] = vaddvq_f32(_ssum);
#else
                float32x2_t _ssum2 = vadd_f32(vget_low_f32(_ssum), vget_high_f32(_ssum));
                _ssum2 = vpadd_f32(_ssum2, _ssum2);
                square_sum[i] = vget_lane_f32(_ssum2, 0);
#endif
            }
#else
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                float ssum = 0.f;
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i * 4;
                    ssum += ptr[0] * ptr[0] + ptr[1] * ptr[1] + ptr[2] * ptr[2] + ptr[3] * ptr[3];
                }
                square_sum[i] = ssum;
            }
#endif // __ARM_NEON

#if __ARM_NEON
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _ssum = vld1q_f32(square_sum + i);
                const float32x4_t _one = vdupq_n_f32(1.f);
                const float32x4_t _eps_vec = vdupq_n_f32(eps);
                float32x4_t _coeff;
                if (eps_mode == 0)
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vaddq_f32(_ssum, _eps_vec)));
                else if (eps_mode == 1)
                    _coeff = vdivq_f32(_one, vmaxq_f32(vsqrtq_f32(_ssum), _eps_vec));
                else
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vmaxq_f32(_ssum, _eps_vec)));
                _ssum = _coeff;
                vst1q_f32(square_sum + i, _ssum);
            }
            for (; i < size; i++)
            {
                square_sum[i] = normalize_coeff(square_sum[i], eps, eps_mode);
            }
#else
            for (int i = 0; i < size; i++)
            {
                square_sum[i] = normalize_coeff(square_sum[i], eps, eps_mode);
            }
#endif // __ARM_NEON

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * 4);

#if __ARM_NEON
                float32x4_t _scale = channel_shared ? vdupq_n_f32(scale_ptr[0]) : vld1q_f32(scale_ptr);
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _ssum = vdupq_n_f32(square_sum[i]);
                    _ssum = vmulq_f32(_ssum, _scale);
                    float32x4_t _p = vld1q_f32(ptr + i * 4);
                    _p = vmulq_f32(_p, _ssum);
                    vst1q_f32(ptr + i * 4, _p);
                }
#else
                for (int i = 0; i < size; i++)
                {
                    const float scale = square_sum[i] * scale_ptr[0];
                    float* p = ptr + i * 4;
                    p[0] *= scale;
                    p[1] *= scale;
                    p[2] *= scale;
                    p[3] *= scale;
                }
#endif // __ARM_NEON
            }

            return 0;
        }

#if __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i += 4)
        {
            const int remain = std::min(4, size - i);
            if (remain == 4)
            {
                float32x4_t _ssum = vdupq_n_f32(0.f);
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i;
                    float32x4_t _p = vld1q_f32(ptr);
                    _ssum = vmlaq_f32(_ssum, _p, _p);
                }
                const float32x4_t _one = vdupq_n_f32(1.f);
                const float32x4_t _eps_vec = vdupq_n_f32(eps);
                float32x4_t _coeff;
                if (eps_mode == 0)
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vaddq_f32(_ssum, _eps_vec)));
                else if (eps_mode == 1)
                    _coeff = vdivq_f32(_one, vmaxq_f32(vsqrtq_f32(_ssum), _eps_vec));
                else
                    _coeff = vdivq_f32(_one, vsqrtq_f32(vmaxq_f32(_ssum, _eps_vec)));
                _ssum = _coeff;
                vst1q_f32(square_sum + i, _ssum);
            }
            else
            {
                for (int j = 0; j < remain; j++)
                {
                    float ssum = 0.f;
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = (const float*)bottom_top_blob.channel(q) + i + j;
                        ssum += ptr[0] * ptr[0];
                    }
                    square_sum[i + j] = normalize_coeff(ssum, eps, eps_mode);
                }
            }
        }
#else
        for (int i = 0; i < size; i++)
        {
            float ssum = 0.f;
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = (const float*)bottom_top_blob.channel(q) + i;
                ssum += ptr[0] * ptr[0];
            }
            square_sum[i] = normalize_coeff(ssum, eps, eps_mode);
        }
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float scale = ((const float*)scale_data)[channel_shared ? 0 : q];

#if __ARM_NEON
            const float32x4_t _scale = vdupq_n_f32(scale);
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _ssum = vld1q_f32(square_sum + i);
                _ssum = vmulq_f32(_ssum, _scale);
                float32x4_t _p = vld1q_f32(ptr + i);
                _p = vmulq_f32(_p, _ssum);
                vst1q_f32(ptr + i, _p);
            }
            for (; i < size; i++)
            {
                ptr[i] *= square_sum[i] * scale;
            }
#else
            for (int i = 0; i < size; i++)
            {
                ptr[i] *= square_sum[i] * scale;
            }
#endif // __ARM_NEON
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
