// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

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

static void cumulative_sum_add(const float* ptr, float* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _outp0 = vld1q_f32(outptr);
        float32x4_t _outp1 = vld1q_f32(outptr + 4);
        vst1q_f32(outptr, vaddq_f32(_outp0, _p0));
        vst1q_f32(outptr + 4, vaddq_f32(_outp1, _p1));

        ptr += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = vld1q_f32(outptr);
        vst1q_f32(outptr, vaddq_f32(_outp, _p));

        ptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr++ += *ptr++;
    }
}

CumulativeSum_arm::CumulativeSum_arm()
{
}

int CumulativeSum_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
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
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* ptr = (float*)bottom_top_blob + i;

            for (int j = 1; j < h; j++)
            {
                cumulative_sum_add(ptr, ptr + w, size1);
                ptr += w;
            }
        }

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
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t cstep = bottom_top_blob.cstep;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* ptr = (float*)bottom_top_blob + i;

            for (int q = 1; q < c; q++)
            {
                cumulative_sum_add(ptr, ptr + cstep, size1);
                ptr += cstep;
            }
        }

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

            for (int i = 1; i < h; i++)
            {
                cumulative_sum_add(ptr, ptr + w, w);
                ptr += w;
            }
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

            for (int z = 1; z < d; z++)
            {
                cumulative_sum_add(ptr, ptr + size, size);
                ptr += size;
            }
        }

        return 0;
    }

    return -100;
}

} // namespace ncnn
