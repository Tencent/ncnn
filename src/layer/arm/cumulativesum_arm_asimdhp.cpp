// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline float32x4_t load_fp16s(const __fp16* ptr)
{
    return vcvt_f32_f16(vld1_f16(ptr));
}

static inline void store_fp16s(__fp16* ptr, float32x4_t v)
{
    vst1_f16(ptr, vcvt_f16_f32(v));
}

static inline float load_fp16s1(const __fp16* ptr)
{
    return (float)*ptr;
}

static inline void store_fp16s1(__fp16* ptr, float v)
{
    *ptr = (__fp16)v;
}

static void cumulative_sum_fp16s(__fp16* ptr, int w)
{
    int i = 0;
    float sum = 0.f;
    const float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _sum = vdupq_n_f32(0.f);
    for (; i + 7 < w; i += 8)
    {
        float16x8_t _p = vld1q_f16(ptr);
        float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
        float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));

        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 3));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 3));
        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 2));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 2));

        _p1 = vaddq_f32(_p1, vdupq_lane_f32(vget_high_f32(_p0), 1));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p1), 1);

        _p0 = vaddq_f32(_p0, _sum);
        _p1 = vaddq_f32(_p1, _sum);
        vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1)));

        _sum = vaddq_f32(_sum, _total);

        ptr += 8;
    }
    for (; i + 3 < w; i += 4)
    {
        float32x4_t _p = load_fp16s(ptr);

        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 3));
        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 2));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p), 1);

        _p = vaddq_f32(_p, _sum);
        store_fp16s(ptr, _p);

        _sum = vaddq_f32(_sum, _total);

        ptr += 4;
    }
    sum = vgetq_lane_f32(_sum, 0);
    for (; i < w; i++)
    {
        sum += load_fp16s1(ptr);
        store_fp16s1(ptr, sum);
        ptr++;
    }
}

static void cumulative_sum_axis_fp16s_tile(__fp16* ptr, size_t stride, int count, int size)
{
    int i = 0;
    if (size == 16)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
        __fp16* p = ptr;
        for (int j = 0; j < count; j++)
        {
            float16x8_t _p0 = vld1q_f16(p);
            float16x8_t _p1 = vld1q_f16(p + 8);
            _sum0 = vaddq_f32(_sum0, vcvt_f32_f16(vget_low_f16(_p0)));
            _sum1 = vaddq_f32(_sum1, vcvt_f32_f16(vget_high_f16(_p0)));
            _sum2 = vaddq_f32(_sum2, vcvt_f32_f16(vget_low_f16(_p1)));
            _sum3 = vaddq_f32(_sum3, vcvt_f32_f16(vget_high_f16(_p1)));
            vst1q_f16(p, vcombine_f16(vcvt_f16_f32(_sum0), vcvt_f16_f32(_sum1)));
            vst1q_f16(p + 8, vcombine_f16(vcvt_f16_f32(_sum2), vcvt_f16_f32(_sum3)));
            p += stride;
        }
        return;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        __fp16* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            float16x8_t _p = vld1q_f16(p);
            _sum0 = vaddq_f32(_sum0, vcvt_f32_f16(vget_low_f16(_p)));
            _sum1 = vaddq_f32(_sum1, vcvt_f32_f16(vget_high_f16(_p)));
            vst1q_f16(p, vcombine_f16(vcvt_f16_f32(_sum0), vcvt_f16_f32(_sum1)));
            p += stride;
        }
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _sum = vdupq_n_f32(0.f);
        __fp16* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            _sum = vaddq_f32(_sum, load_fp16s(p));
            store_fp16s(p, _sum);
            p += stride;
        }
    }
    for (; i < size; i++)
    {
        float sum = 0.f;
        __fp16* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            sum += load_fp16s1(p);
            store_fp16s1(p, sum);
            p += stride;
        }
    }
}

static void cumulative_sum_axis_fp16s(__fp16* ptr, size_t stride, int count, int size)
{
    if (count <= 1)
        return;

    for (int i = 0; i < size; i += 16)
    {
        cumulative_sum_axis_fp16s_tile(ptr + i, stride, count, std::min(16, size - i));
    }
}

static void cumulative_sum_axis_fp16s_parallel(__fp16* ptr, size_t stride, int count, int size, const Option& opt)
{
    if (count <= 1)
        return;

    const int num_threads = std::min(opt.num_threads, std::max(1, (size + 15) / 16));
    const int nn_size = (size + 15) / 16;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < nn_size; i++)
    {
        const int start = i * 16;
        cumulative_sum_axis_fp16s_tile(ptr + start, stride, count, std::min(16, size - start));
    }
}

int CumulativeSum_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int c = bottom_top_blob.c;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum_fp16s((__fp16*)bottom_top_blob, w);

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        cumulative_sum_axis_fp16s_parallel((__fp16*)bottom_top_blob, w, h, w, opt);

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            cumulative_sum_fp16s(bottom_top_blob.row<__fp16>(i), w);
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const size_t cstep = bottom_top_blob.cstep;

        cumulative_sum_axis_fp16s_parallel((__fp16*)bottom_top_blob, cstep, c, size, opt);

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qz = 0; qz < c * d; qz++)
        {
            const int q = qz / d;
            const int z = qz - q * d;

            __fp16* ptr = bottom_top_blob.channel(q).depth(z);

            cumulative_sum_axis_fp16s(ptr, w, h, w);
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qi = 0; qi < c * d * h; qi++)
        {
            const int q = qi / (d * h);
            const int i = qi - q * (d * h);

            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q) + i * w;

            cumulative_sum_fp16s(ptr, w);
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            cumulative_sum_axis_fp16s(ptr, size, d, size);
        }

        return 0;
    }

    return -100;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
