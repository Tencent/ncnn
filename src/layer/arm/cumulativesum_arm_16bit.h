// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static void cumulative_sum_16bit(unsigned short* ptr, int w)
{
    int i = 0;
    float sum = 0.f;
#if __ARM_NEON
    const float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _sum = vdupq_n_f32(0.f);
    for (; i + 7 < w; i += 8)
    {
        float32x4_t _p0 = Conv::load4(ptr);
        float32x4_t _p1 = Conv::load4(ptr + 4);

        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 3));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 3));
        _p0 = vaddq_f32(_p0, vextq_f32(_zero, _p0, 2));
        _p1 = vaddq_f32(_p1, vextq_f32(_zero, _p1, 2));

        _p1 = vaddq_f32(_p1, vdupq_lane_f32(vget_high_f32(_p0), 1));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p1), 1);

        _p0 = vaddq_f32(_p0, _sum);
        _p1 = vaddq_f32(_p1, _sum);
        Conv::store4(ptr, _p0);
        Conv::store4(ptr + 4, _p1);

        _sum = vaddq_f32(_sum, _total);

        ptr += 8;
    }
    for (; i + 3 < w; i += 4)
    {
        float32x4_t _p = Conv::load4(ptr);

        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 3));
        _p = vaddq_f32(_p, vextq_f32(_zero, _p, 2));

        float32x4_t _total = vdupq_lane_f32(vget_high_f32(_p), 1);

        _p = vaddq_f32(_p, _sum);
        Conv::store4(ptr, _p);

        _sum = vaddq_f32(_sum, _total);

        ptr += 4;
    }
    sum = vgetq_lane_f32(_sum, 0);
#endif // __ARM_NEON
    for (; i < w; i++)
    {
        sum += Conv::load1(ptr);
        Conv::store1(ptr, sum);
        ptr++;
    }
}

template<typename Conv>
static void cumulative_sum_axis_16bit_serial(unsigned short* ptr, size_t stride, int count, int size)
{
    const int nn = (size + 3) / 4;
    for (int ii = 0; ii < nn; ii++)
    {
        const int i = ii * 4;
        const int size1 = std::min(4, size - i);

        unsigned short* p = ptr + i;
#if __ARM_NEON
        if (size1 == 4)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (int j = 0; j < count; j++)
            {
                _sum = vaddq_f32(_sum, Conv::load4(p));
                Conv::store4(p, _sum);
                p += stride;
            }
            continue;
        }
#endif // __ARM_NEON

        float sum[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < count; j++)
        {
            for (int k = 0; k < size1; k++)
            {
                sum[k] += Conv::load1(p + k);
                Conv::store1(p + k, sum[k]);
            }
            p += stride;
        }
    }
}

template<typename Conv>
static void cumulative_sum_axis_16bit(unsigned short* ptr, size_t stride, int count, int size, const Option& opt)
{
    const int nn = (size + 3) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < nn; ii++)
    {
        const int i = ii * 4;
        const int size1 = std::min(4, size - i);

        unsigned short* p = ptr + i;
#if __ARM_NEON
        if (size1 == 4)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (int j = 0; j < count; j++)
            {
                _sum = vaddq_f32(_sum, Conv::load4(p));
                Conv::store4(p, _sum);
                p += stride;
            }
            continue;
        }
#endif // __ARM_NEON

        float sum[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < count; j++)
        {
            for (int k = 0; k < size1; k++)
            {
                sum[k] += Conv::load1(p + k);
                Conv::store1(p + k, sum[k]);
            }
            p += stride;
        }
    }
}

template<typename Conv>
static int cumulative_sum_forward_inplace_16bit(Mat& bottom_top_blob, int axis, const Option& opt)
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int c = bottom_top_blob.c;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum_16bit<Conv>((unsigned short*)bottom_top_blob, w);

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        cumulative_sum_axis_16bit<Conv>((unsigned short*)bottom_top_blob, w, h, w, opt);

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            cumulative_sum_16bit<Conv>(bottom_top_blob.row<unsigned short>(i), w);
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const size_t cstep = bottom_top_blob.cstep;

        cumulative_sum_axis_16bit<Conv>((unsigned short*)bottom_top_blob, cstep, c, size, opt);

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

            cumulative_sum_axis_16bit_serial<Conv>(ptr, w, h, w);
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

            cumulative_sum_16bit<Conv>(ptr, w);
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

            cumulative_sum_axis_16bit_serial<Conv>(ptr, size, d, size);
        }

        return 0;
    }

    return -100;
}
