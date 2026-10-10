// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

Softmax_arm::Softmax_arm()
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

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __ARM_NEON
    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _max = vmaxq_f32(_max, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

#if __ARM_NEON
    if (elempack == 1)
    {
#if __aarch64__
        max = std::max(max, vmaxvq_f32(_max));
#else
        float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
        float32x2_t _mm2 = vpmax_f32(_max2, _max2);
        max = std::max(max, vget_lane_f32(_mm2, 0));
#endif

        _max = vdupq_n_f32(max);
    }
#endif // __ARM_NEON

    // reduce exp(x - max)
#if __ARM_NEON
    float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float sum = 0.f;
    {
        float* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1q_f32(ptr, _p);
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

#if __ARM_NEON
    if (elempack == 1)
    {
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
        sum += vget_lane_f32(_ss2, 0);
#endif

        _sum = vdupq_n_f32(sum);
    }
#endif // __ARM_NEON

#if __ARM_NEON
    _sum = div_ps(vdupq_n_f32(1.f), _sum);
#endif // __ARM_NEON
    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmulq_f32(_p, _sum);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

#if __ARM_NEON
static void softmax_pack4(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            float32x4x4_t _p = vld4q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);
            float32x4_t _max2 = vmaxq_f32(_p.val[0], _p.val[1]);
            float32x4_t _max4 = vmaxq_f32(_p.val[2], _p.val[3]);
            _max = vmaxq_f32(_max, vmaxq_f32(_max2, _max4));
            vst1q_f32(maxptr, _max);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = vld1q_f32(ptr);
#if __aarch64__
            float max0 = vmaxvq_f32(_p);
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
            float32x2_t _mm2 = vpmax_f32(_max2, _max2);
            float max0 = vget_lane_f32(_mm2, 0);
#endif
            *maxptr = std::max(*maxptr, max0);
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            float32x4x4_t _p = vld4q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);
            float32x4_t _p0 = vsubq_f32(_p.val[0], _max);
            float32x4_t _p1 = vsubq_f32(_p.val[1], _max);
            float32x4_t _p2 = vsubq_f32(_p.val[2], _max);
            float32x4_t _p3 = vsubq_f32(_p.val[3], _max);
            _p.val[0] = exp_ps(_p0);
            _p.val[1] = exp_ps(_p1);
            _p.val[2] = exp_ps(_p2);
            _p.val[3] = exp_ps(_p3);
            vst4q_f32(ptr, _p);
            float32x4_t _sum = vld1q_f32(sumptr);
            float32x4_t _ss2 = vaddq_f32(_p.val[0], _p.val[1]);
            float32x4_t _ss4 = vaddq_f32(_p.val[2], _p.val[3]);
            _sum = vaddq_f32(_sum, vaddq_f32(_ss2, _ss4));
            vst1q_f32(sumptr, _sum);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vdupq_n_f32(*maxptr);
            _p = exp_ps(vsubq_f32(_p, _max));
            vst1q_f32(ptr, _p);
#if __aarch64__
            float sum0 = vaddvq_f32(_p);
#else
            float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
            float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
            float sum0 = vget_lane_f32(_ss2, 0);
#endif
            *sumptr += sum0;
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float32x4_t _one = vdupq_n_f32(1.f);
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = div_ps(_one, _sum);
            vst1q_f32(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            float32x4x4_t _p = vld4q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _p.val[0] = vmulq_f32(_p.val[0], _sum);
            _p.val[1] = vmulq_f32(_p.val[1], _sum);
            _p.val[2] = vmulq_f32(_p.val[2], _sum);
            _p.val[3] = vmulq_f32(_p.val[3], _sum);
            vst4q_f32(ptr, _p);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_dup_f32(sumptr);
            _p = vmulq_f32(_p, _sum);
            vst1q_f32(ptr, _p);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __ARM_NEON

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);
            _max = vmaxq_f32(_max, _p);
            vst1q_f32(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, *ptr);
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1q_f32(ptr, _p);
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            float v = expf(*ptr - *maxptr);
            *ptr = v;
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = div_ps(_one, _sum);
            vst1q_f32(sumptr, _sum);
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _p = vmulq_f32(_p, _sum);
            vst1q_f32(ptr, _p);
            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *ptr *= *sumptr;
            ptr++;
            sumptr++;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __ARM_NEON
        float32x4_t _negmax = vdupq_n_f32(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            vst1q_f32(maxptr, _negmax);
            maxptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // reduce exp(x - max)
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        float32x4_t _zero = vdupq_n_f32(0.f);
        for (; j + 3 < size1; j += 4)
        {
            vst1q_f32(sumptr, _zero);
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        softmax_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void softmax_bf16s(unsigned short* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __ARM_NEON
    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        float32x4_t _max1 = vdupq_n_f32(-FLT_MAX);
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _max = vmaxq_f32(_max, _p0);
            _max1 = vmaxq_f32(_max1, _p1);
            ptr += 8;
        }
        _max = vmaxq_f32(_max, _max1);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _max = vmaxq_f32(_max, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

#if __ARM_NEON
    if (elempack == 1)
    {
#if __aarch64__
        max = std::max(max, vmaxvq_f32(_max));
#else
        float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
        float32x2_t _mm2 = vpmax_f32(_max2, _max2);
        max = std::max(max, vget_lane_f32(_mm2, 0));
#endif

        _max = vdupq_n_f32(max);
    }
#endif // __ARM_NEON

    // reduce exp(x - max)
#if __ARM_NEON
    float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _p0 = vsubq_f32(_p0, _max);
            _p1 = vsubq_f32(_p1, _max);
            _p0 = exp_ps(_p0);
            _p1 = exp_ps(_p1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            _sum = vaddq_f32(_sum, _p0);
            _sum1 = vaddq_f32(_sum1, _p1);
            ptr += 8;
        }
        _sum = vaddq_f32(_sum, _sum1);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1_u16(ptr, float2bfloat(_p));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr++;
        }
    }

#if __ARM_NEON
    if (elempack == 1)
    {
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
        sum += vget_lane_f32(_ss2, 0);
#endif

        _sum = vdupq_n_f32(sum);
    }
#endif // __ARM_NEON

#if __ARM_NEON
    _sum = div_ps(vdupq_n_f32(1.f), _sum);
#endif // __ARM_NEON
    sum = 1.f / sum;

    // div sum
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _p0 = vmulq_f32(_p0, _sum);
            _p1 = vmulq_f32(_p1, _sum);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vmulq_f32(_p, _sum);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr++;
        }
    }
}

#if __ARM_NEON
static void softmax_bf16s_pack4(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            uint16x4x4_t _p = vld4_u16(ptr);
            float32x4_t _p0 = bfloat2float(_p.val[0]);
            float32x4_t _p1 = bfloat2float(_p.val[1]);
            float32x4_t _p2 = bfloat2float(_p.val[2]);
            float32x4_t _p3 = bfloat2float(_p.val[3]);
            float32x4_t _max = vld1q_f32(maxptr);
            float32x4_t _max2 = vmaxq_f32(_p0, _p1);
            float32x4_t _max4 = vmaxq_f32(_p2, _p3);
            _max = vmaxq_f32(_max, vmaxq_f32(_max2, _max4));
            vst1q_f32(maxptr, _max);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
#if __aarch64__
            float max0 = vmaxvq_f32(_p);
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
            float32x2_t _mm2 = vpmax_f32(_max2, _max2);
            float max0 = vget_lane_f32(_mm2, 0);
#endif
            *maxptr = std::max(*maxptr, max0);
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            uint16x4x4_t _p = vld4_u16(ptr);
            float32x4_t _p0 = bfloat2float(_p.val[0]);
            float32x4_t _p1 = bfloat2float(_p.val[1]);
            float32x4_t _p2 = bfloat2float(_p.val[2]);
            float32x4_t _p3 = bfloat2float(_p.val[3]);
            float32x4_t _max = vld1q_f32(maxptr);
            _p0 = vsubq_f32(_p0, _max);
            _p1 = vsubq_f32(_p1, _max);
            _p2 = vsubq_f32(_p2, _max);
            _p3 = vsubq_f32(_p3, _max);
            _p0 = exp_ps(_p0);
            _p1 = exp_ps(_p1);
            _p2 = exp_ps(_p2);
            _p3 = exp_ps(_p3);
            _p.val[0] = float2bfloat(_p0);
            _p.val[1] = float2bfloat(_p1);
            _p.val[2] = float2bfloat(_p2);
            _p.val[3] = float2bfloat(_p3);
            vst4_u16(ptr, _p);
            float32x4_t _sum = vld1q_f32(sumptr);
            float32x4_t _ss2 = vaddq_f32(_p0, _p1);
            float32x4_t _ss4 = vaddq_f32(_p2, _p3);
            _sum = vaddq_f32(_sum, vaddq_f32(_ss2, _ss4));
            vst1q_f32(sumptr, _sum);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _max = vdupq_n_f32(*maxptr);
            _p = exp_ps(vsubq_f32(_p, _max));
            vst1_u16(ptr, float2bfloat(_p));
#if __aarch64__
            float sum0 = vaddvq_f32(_p);
#else
            float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
            float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
            float sum0 = vget_lane_f32(_ss2, 0);
#endif
            *sumptr += sum0;
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float32x4_t _one = vdupq_n_f32(1.f);
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = div_ps(_one, _sum);
            vst1q_f32(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            uint16x4x4_t _p = vld4_u16(ptr);
            float32x4_t _p0 = bfloat2float(_p.val[0]);
            float32x4_t _p1 = bfloat2float(_p.val[1]);
            float32x4_t _p2 = bfloat2float(_p.val[2]);
            float32x4_t _p3 = bfloat2float(_p.val[3]);
            float32x4_t _sum = vld1q_f32(sumptr);
            _p0 = vmulq_f32(_p0, _sum);
            _p1 = vmulq_f32(_p1, _sum);
            _p2 = vmulq_f32(_p2, _sum);
            _p3 = vmulq_f32(_p3, _sum);
            _p.val[0] = float2bfloat(_p0);
            _p.val[1] = float2bfloat(_p1);
            _p.val[2] = float2bfloat(_p2);
            _p.val[3] = float2bfloat(_p3);
            vst4_u16(ptr, _p);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _sum = vld1q_dup_f32(sumptr);
            _p = vmulq_f32(_p, _sum);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __ARM_NEON

static void softmax_bf16s_pack1(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 7 < size1; j += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            float32x4_t _max0 = vld1q_f32(maxptr);
            float32x4_t _max1 = vld1q_f32(maxptr + 4);
            _max0 = vmaxq_f32(_max0, _p0);
            _max1 = vmaxq_f32(_max1, _p1);
            vst1q_f32(maxptr, _max0);
            vst1q_f32(maxptr + 4, _max1);
            ptr += 8;
            maxptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _max = vld1q_f32(maxptr);
            _max = vmaxq_f32(_max, _p);
            vst1q_f32(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 7 < size1; j += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            float32x4_t _max0 = vld1q_f32(maxptr);
            float32x4_t _max1 = vld1q_f32(maxptr + 4);
            float32x4_t _sum0 = vld1q_f32(sumptr);
            float32x4_t _sum1 = vld1q_f32(sumptr + 4);
            _p0 = vsubq_f32(_p0, _max0);
            _p1 = vsubq_f32(_p1, _max1);
            _p0 = exp_ps(_p0);
            _p1 = exp_ps(_p1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            _sum0 = vaddq_f32(_sum0, _p0);
            _sum1 = vaddq_f32(_sum1, _p1);
            vst1q_f32(sumptr, _sum0);
            vst1q_f32(sumptr + 4, _sum1);
            ptr += 8;
            maxptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _max = vld1q_f32(maxptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1_u16(ptr, float2bfloat(_p));
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - *maxptr);
            *ptr = float32_to_bfloat16(v);
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = div_ps(_one, _sum);
            vst1q_f32(sumptr, _sum);
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        for (; j + 7 < size1; j += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            float32x4_t _sum0 = vld1q_f32(sumptr);
            float32x4_t _sum1 = vld1q_f32(sumptr + 4);
            _p0 = vmulq_f32(_p0, _sum0);
            _p1 = vmulq_f32(_p1, _sum1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
            sumptr += 8;
        }
        for (; j + 3 < size1; j += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _sum = vld1q_f32(sumptr);
            _p = vmulq_f32(_p, _sum);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * *sumptr);
            ptr++;
            sumptr++;
        }
    }
}

static void softmax_bf16s(unsigned short* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __ARM_NEON
        float32x4_t _negmax = vdupq_n_f32(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            vst1q_f32(maxptr, _negmax);
            maxptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // reduce exp(x - max)
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __ARM_NEON
        float32x4_t _zero = vdupq_n_f32(0.f);
        for (; j + 3 < size1; j += 4)
        {
            vst1q_f32(sumptr, _zero);
            sumptr += 4;
        }
#endif // __ARM_NEON
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        softmax_bf16s_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
        softmax_bf16s_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        unsigned short* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_bf16s(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            softmax_bf16s(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax_bf16s(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax_bf16s(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax_bf16s(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_bf16s(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
