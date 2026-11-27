// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if NCNN_BF16
#if __ARM_NEON
static inline float32x4_t load_bf16s(const unsigned short* ptr)
{
    return bfloat2float(vld1_u16(ptr));
}

static inline void store_bf16s(unsigned short* ptr, float32x4_t v)
{
    vst1_u16(ptr, float2bfloat(v));
}
#endif // __ARM_NEON

static inline float load_bf16s1(const unsigned short* ptr)
{
    return bfloat16_to_float32(*ptr);
}

static inline void store_bf16s1(unsigned short* ptr, float v)
{
    *ptr = float32_to_bfloat16(v);
}
#endif // NCNN_BF16

static void cumulative_sum(float* ptr, int w)
{
    int i = 0;
    float sum = 0.f;
#if __ARM_NEON
    const float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _sum = vdupq_n_f32(0.f);
    for (; i + 7 < w; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);

        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 3));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 3));
        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 2));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 2));

        _p1 = vaddq_f32(_p1, vdupq_lane_f32(vget_high_f32(_p0), 1));

        // broadcast the block total before adding the carry, so only one add is loop-carried
        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p1), 1);

        _p0 = vaddq_f32(_p0, _sum);
        _p1 = vaddq_f32(_p1, _sum);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);

        _sum = vaddq_f32(_sum, _total);

        ptr += 8;
    }
    for (; i + 3 < w; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);

        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 3));
        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 2));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p), 1);

        _p = vaddq_f32(_p, _sum);
        vst1q_f32(ptr, _p);

        _sum = vaddq_f32(_sum, _total);

        ptr += 4;
    }
    sum = vgetq_lane_f32(_sum, 0);
#endif // __ARM_NEON
    for (; i < w; i++)
    {
        sum += *ptr;
        *ptr++ = sum;
    }
}

static void cumulative_sum_axis_tile(float* ptr, size_t stride, int count, int size)
{
    int i = 0;
#if __ARM_NEON
    if (size == 16)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
        float* p = ptr;
        for (int j = 0; j < count; j++)
        {
            _sum0 = vaddq_f32(_sum0, vld1q_f32(p));
            _sum1 = vaddq_f32(_sum1, vld1q_f32(p + 4));
            _sum2 = vaddq_f32(_sum2, vld1q_f32(p + 8));
            _sum3 = vaddq_f32(_sum3, vld1q_f32(p + 12));
            vst1q_f32(p, _sum0);
            vst1q_f32(p + 4, _sum1);
            vst1q_f32(p + 8, _sum2);
            vst1q_f32(p + 12, _sum3);
            p += stride;
        }
        return;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            _sum0 = vaddq_f32(_sum0, vld1q_f32(p));
            _sum1 = vaddq_f32(_sum1, vld1q_f32(p + 4));
            vst1q_f32(p, _sum0);
            vst1q_f32(p + 4, _sum1);
            p += stride;
        }
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _sum = vdupq_n_f32(0.f);
        float* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            _sum = vaddq_f32(_sum, vld1q_f32(p));
            vst1q_f32(p, _sum);
            p += stride;
        }
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float sum = 0.f;
        float* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            sum += *p;
            *p = sum;
            p += stride;
        }
    }
}

static void cumulative_sum_axis(float* ptr, size_t stride, int count, int size)
{
    if (count <= 1)
        return;

    for (int i = 0; i < size; i += 16)
    {
        cumulative_sum_axis_tile(ptr + i, stride, count, std::min(16, size - i));
    }
}

static void cumulative_sum_axis_parallel(float* ptr, size_t stride, int count, int size, const Option& opt)
{
    if (count <= 1)
        return;

    const int num_threads = std::min(opt.num_threads, std::max(1, (size + 15) / 16));
    const int nn_size = (size + 15) / 16;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < nn_size; i++)
    {
        const int start = i * 16;
        cumulative_sum_axis_tile(ptr + start, stride, count, std::min(16, size - start));
    }
}

#if NCNN_BF16
static void cumulative_sum_bf16s(unsigned short* ptr, int w)
{
    int i = 0;
    float sum = 0.f;
#if __ARM_NEON
    const float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _sum = vdupq_n_f32(0.f);
    for (; i + 7 < w; i += 8)
    {
        uint16x8_t _p = vld1q_u16(ptr);
        float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
        float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 3));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 3));
        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 2));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 2));

        _p1 = vaddq_f32(_p1, vdupq_lane_f32(vget_high_f32(_p0), 1));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p1), 1);

        _p0 = vaddq_f32(_p0, _sum);
        _p1 = vaddq_f32(_p1, _sum);
        vst1q_u16(ptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

        _sum = vaddq_f32(_sum, _total);

        ptr += 8;
    }
    for (; i + 3 < w; i += 4)
    {
        float32x4_t _p = load_bf16s(ptr);

        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 3));
        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 2));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p), 1);

        _p = vaddq_f32(_p, _sum);
        store_bf16s(ptr, _p);

        _sum = vaddq_f32(_sum, _total);

        ptr += 4;
    }
    sum = vgetq_lane_f32(_sum, 0);
#endif // __ARM_NEON
    for (; i < w; i++)
    {
        sum += load_bf16s1(ptr);
        store_bf16s1(ptr, sum);
        ptr++;
    }
}

static void cumulative_sum_axis_bf16s_tile(unsigned short* ptr, size_t stride, int count, int size)
{
    int i = 0;
#if __ARM_NEON
    if (size == 16)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
        unsigned short* p = ptr;
        for (int j = 0; j < count; j++)
        {
            uint16x8_t _p0 = vld1q_u16(p);
            uint16x8_t _p1 = vld1q_u16(p + 8);
            _sum0 = vaddq_f32(_sum0, bfloat2float(vget_low_u16(_p0)));
            _sum1 = vaddq_f32(_sum1, bfloat2float(vget_high_u16(_p0)));
            _sum2 = vaddq_f32(_sum2, bfloat2float(vget_low_u16(_p1)));
            _sum3 = vaddq_f32(_sum3, bfloat2float(vget_high_u16(_p1)));
            vst1q_u16(p, vcombine_u16(float2bfloat(_sum0), float2bfloat(_sum1)));
            vst1q_u16(p + 8, vcombine_u16(float2bfloat(_sum2), float2bfloat(_sum3)));
            p += stride;
        }
        return;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        unsigned short* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            uint16x8_t _p = vld1q_u16(p);
            _sum0 = vaddq_f32(_sum0, bfloat2float(vget_low_u16(_p)));
            _sum1 = vaddq_f32(_sum1, bfloat2float(vget_high_u16(_p)));
            vst1q_u16(p, vcombine_u16(float2bfloat(_sum0), float2bfloat(_sum1)));
            p += stride;
        }
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _sum = vdupq_n_f32(0.f);
        unsigned short* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            _sum = vaddq_f32(_sum, load_bf16s(p));
            store_bf16s(p, _sum);
            p += stride;
        }
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float sum = 0.f;
        unsigned short* p = ptr + i;
        for (int j = 0; j < count; j++)
        {
            sum += load_bf16s1(p);
            store_bf16s1(p, sum);
            p += stride;
        }
    }
}

static void cumulative_sum_axis_bf16s(unsigned short* ptr, size_t stride, int count, int size)
{
    if (count <= 1)
        return;

    for (int i = 0; i < size; i += 16)
    {
        cumulative_sum_axis_bf16s_tile(ptr + i, stride, count, std::min(16, size - i));
    }
}

static void cumulative_sum_axis_bf16s_parallel(unsigned short* ptr, size_t stride, int count, int size, const Option& opt)
{
    if (count <= 1)
        return;

    const int num_threads = std::min(opt.num_threads, std::max(1, (size + 15) / 16));
    const int nn_size = (size + 15) / 16;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < nn_size; i++)
    {
        const int start = i * 16;
        cumulative_sum_axis_bf16s_tile(ptr + start, stride, count, std::min(16, size - start));
    }
}

#endif // NCNN_BF16

CumulativeSum_arm::CumulativeSum_arm()
{
#if __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int CumulativeSum_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int c = bottom_top_blob.c;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum(bottom_top_blob, w);

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        cumulative_sum_axis_parallel((float*)bottom_top_blob, w, h, w, opt);

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            cumulative_sum(bottom_top_blob.row(i), w);
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const size_t cstep = bottom_top_blob.cstep;

        cumulative_sum_axis_parallel((float*)bottom_top_blob, cstep, c, size, opt);

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qz = 0; qz < c * d; qz++)
        {
            const int q = qz / d;
            const int z = qz - q * d;

            float* ptr = bottom_top_blob.channel(q).depth(z);

            cumulative_sum_axis(ptr, w, h, w);
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

            float* ptr = (float*)bottom_top_blob.channel(q) + i * w;

            cumulative_sum(ptr, w);
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            cumulative_sum_axis(ptr, size, d, size);
        }

        return 0;
    }

    return -100;
}

#if NCNN_BF16
int CumulativeSum_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int c = bottom_top_blob.c;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum_bf16s((unsigned short*)bottom_top_blob, w);

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        cumulative_sum_axis_bf16s_parallel((unsigned short*)bottom_top_blob, w, h, w, opt);

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            cumulative_sum_bf16s(bottom_top_blob.row<unsigned short>(i), w);
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const size_t cstep = bottom_top_blob.cstep;

        cumulative_sum_axis_bf16s_parallel((unsigned short*)bottom_top_blob, cstep, c, size, opt);

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qz = 0; qz < c * d; qz++)
        {
            const int q = qz / d;
            const int z = qz - q * d;

            unsigned short* ptr = bottom_top_blob.channel(q).depth(z);

            cumulative_sum_axis_bf16s(ptr, w, h, w);
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

            unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q) + i * w;

            cumulative_sum_bf16s(ptr, w);
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            cumulative_sum_axis_bf16s(ptr, size, d, size);
        }

        return 0;
    }

    return -100;
}
#endif // NCNN_BF16

} // namespace ncnn
