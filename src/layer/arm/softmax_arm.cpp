// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
static void softmax_unroll4(float* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _max = vmaxq_f32(_max, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce max 4 to 1
        // broadcast 1 to 4
        _max = vmaxq_f32(_max, vextq_f32(_max, _max, 1));
        _max = vmaxq_f32(_max, vextq_f32(_max, _max, 2));
    }
    if (elempack == 1)
    {
        // fine
    }

    // reduce exp(x - max)
    float32x4_t _sum = vdupq_n_f32(0.f);
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1q_f32(ptr, _p);
            _sum = vaddq_f32(_sum, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce sum 4 to 1
        // broadcast 1 to 4
        _sum = vaddq_f32(_sum, vextq_f32(_sum, _sum, 1));
        _sum = vaddq_f32(_sum, vextq_f32(_sum, _sum, 2));
    }
    if (elempack == 1)
    {
        // fine
    }

    _sum = div_ps(vdupq_n_f32(1.f), _sum);

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmulq_f32(_p, _sum);
            vst1q_f32(ptr, _p);
            ptr += stride;
        }
    }
}
#endif // __ARM_NEON

static void softmax_unroll2(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // TODO neon optimize
    // assert elempack == 1

    // reduce max
    float max0 = -FLT_MAX;
    float max1 = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max0 = std::max(max0, ptr[0]);
            max1 = std::max(max1, ptr[1]);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum0 = 0.f;
    float sum1 = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v0 = expf(ptr[0] - max0);
            float v1 = expf(ptr[1] - max1);
            ptr[0] = v0;
            ptr[1] = v1;
            sum0 += v0;
            sum1 += v1;
            ptr += stride;
        }
    }

    sum0 = 1.f / sum0;
    sum1 = 1.f / sum1;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            ptr[0] *= sum0;
            ptr[1] *= sum1;
            ptr += stride;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr += stride;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            *ptr *= sum;
            ptr += stride;
        }
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
        const int size = w * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll4(ptr, h, elempack, size);
        }
#endif // __ARM_NEON
        for (; i + 1 < size; i += 2)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, h, elempack, size);
        }
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, h, elempack, size);
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
        const int size = w * h * d * elempack;
        const int stride = bottom_top_blob.cstep * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll4(ptr, channels, elempack, stride);
        }
#endif // __ARM_NEON
        for (; i + 1 < size; i += 2)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax_unroll2(ptr, channels, elempack, stride);
        }
        for (; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, channels, elempack, stride);
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
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            const int size = w * h * elempack;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                softmax_unroll4(ptr, d, 1, size);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i + 1 < size; i += 2)
            {
                softmax_unroll2(ptr, d, 1, size);
                ptr += 2;
            }
            for (; i < size; i++)
            {
                softmax(ptr, d, 1, size);
                ptr++;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                const int size = w * elempack;

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    softmax_unroll4(ptr, h, 1, size);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j + 1 < size; j += 2)
                {
                    softmax_unroll2(ptr, h, 1, size);
                    ptr += 2;
                }
                for (; j < size; j++)
                {
                    softmax(ptr, h, 1, size);
                    ptr++;
                }
            }
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
static void softmax_bf16s_unroll4(unsigned short* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _max = vmaxq_f32(_max, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce max 4 to 1
        // broadcast 1 to 4
        _max = vmaxq_f32(_max, vextq_f32(_max, _max, 1));
        _max = vmaxq_f32(_max, vextq_f32(_max, _max, 2));
    }
    if (elempack == 1)
    {
        // fine
    }

    // reduce exp(x - max)
    float32x4_t _sum = vdupq_n_f32(0.f);
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vsubq_f32(_p, _max);
            _p = exp_ps(_p);
            vst1_u16(ptr, float2bfloat(_p));
            _sum = vaddq_f32(_sum, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce sum 4 to 1
        // broadcast 1 to 4
        _sum = vaddq_f32(_sum, vextq_f32(_sum, _sum, 1));
        _sum = vaddq_f32(_sum, vextq_f32(_sum, _sum, 2));
    }
    if (elempack == 1)
    {
        // fine
    }

    _sum = div_ps(vdupq_n_f32(1.f), _sum);

    // div sum
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vmulq_f32(_p, _sum);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += stride;
        }
    }
}
#endif // __ARM_NEON

static void softmax_bf16s_unroll2(unsigned short* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // TODO neon optimize
    // assert elempack == 1

    // reduce max
    float max0 = -FLT_MAX;
    float max1 = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max0 = std::max(max0, bfloat16_to_float32(ptr[0]));
            max1 = std::max(max1, bfloat16_to_float32(ptr[1]));
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum0 = 0.f;
    float sum1 = 0.f;
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v0 = expf(bfloat16_to_float32(ptr[0]) - max0);
            float v1 = expf(bfloat16_to_float32(ptr[1]) - max1);
            ptr[0] = float32_to_bfloat16(v0);
            ptr[1] = float32_to_bfloat16(v1);
            sum0 += v0;
            sum1 += v1;
            ptr += stride;
        }
    }

    sum0 = 1.f / sum0;
    sum1 = 1.f / sum1;

    // div sum
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * sum0);
            ptr[1] = float32_to_bfloat16(bfloat16_to_float32(ptr[1]) * sum1);
            ptr += stride;
        }
    }
}

static void softmax_bf16s(unsigned short* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr));
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr += stride;
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr += stride;
        }
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
        const int size = w * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s_unroll4(ptr, h, elempack, size);
        }
#endif // __ARM_NEON
        for (; i + 1 < size; i += 2)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s_unroll2(ptr, h, elempack, size);
        }
        for (; i < size; i++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s(ptr, h, elempack, size);
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
        const int size = w * h * d * elempack;
        const int stride = bottom_top_blob.cstep * elempack;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s_unroll4(ptr, channels, elempack, stride);
        }
#endif // __ARM_NEON
        for (; i + 1 < size; i += 2)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s_unroll2(ptr, channels, elempack, stride);
        }
        for (; i < size; i++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob + i;

            softmax_bf16s(ptr, channels, elempack, stride);
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
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            const int size = w * h * elempack;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                softmax_bf16s_unroll4(ptr, d, 1, size);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i + 1 < size; i += 2)
            {
                softmax_bf16s_unroll2(ptr, d, 1, size);
                ptr += 2;
            }
            for (; i < size; i++)
            {
                softmax_bf16s(ptr, d, 1, size);
                ptr++;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q).depth(i);

                const int size = w * elempack;

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    softmax_bf16s_unroll4(ptr, h, 1, size);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j + 1 < size; j += 2)
                {
                    softmax_bf16s_unroll2(ptr, h, 1, size);
                    ptr += 2;
                }
                for (; j < size; j++)
                {
                    softmax_bf16s(ptr, h, 1, size);
                    ptr++;
                }
            }
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
