// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "arm_usability.h"
#include "neon_mathfun_fp16s.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void softmax_fp16s(__fp16* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
    float16x8_t _max8 = vdupq_n_f16(-65504.f);
    float16x4_t _max4 = vdup_n_f16(-65504.f);
    __fp16 max = -65504.f;
    {
        const __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _max8 = vmaxq_f16(_max8, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _max4 = vmax_f16(_max4, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    if (elempack == 4)
    {
        _max4 = vmax_f16(_max4, vget_low_f16(_max8));
        _max4 = vmax_f16(_max4, vget_high_f16(_max8));

        _max8 = vcombine_f16(_max4, _max4);
    }
    if (elempack == 1)
    {
        max = std::max(max, vmaxvq_f16(_max8));
        max = std::max(max, vmaxv_f16(_max4));

        _max4 = vdup_n_f16(max);
        _max8 = vdupq_n_f16(max);
    }

    // reduce exp(x - max)
    float16x8_t _sum8 = vdupq_n_f16(0.f);
    float16x4_t _sum4 = vdup_n_f16(0.f);
    __fp16 sum = 0.f;
    {
        __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = exp_ps_f16(vsubq_f16(_p, _max8));
            vst1q_f16(ptr, _p);
            _sum8 = vaddq_f16(_sum8, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = exp_ps_f16(vsub_f16(_p, _max4));
            vst1_f16(ptr, _p);
            _sum4 = vadd_f16(_sum4, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = (__fp16)expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

    if (elempack == 4)
    {
        _sum4 = vadd_f16(_sum4, vget_low_f16(_sum8));
        _sum4 = vadd_f16(_sum4, vget_high_f16(_sum8));

        _sum8 = vcombine_f16(_sum4, _sum4);
    }
    if (elempack == 1)
    {
        _sum4 = vadd_f16(_sum4, vget_low_f16(_sum8));
        _sum4 = vadd_f16(_sum4, vget_high_f16(_sum8));
        _sum4 = vpadd_f16(_sum4, _sum4);
        _sum4 = vpadd_f16(_sum4, _sum4);
        sum += vget_lane_f16(_sum4, 0);

        _sum4 = vdup_n_f16(sum);
        _sum8 = vdupq_n_f16(sum);
    }

    _sum8 = vdivq_f16(vdupq_n_f16(1.f), _sum8);
    _sum4 = vdiv_f16(vdup_n_f16(1.f), _sum4);
    sum = (__fp16)(1.f / sum);

    // div sum
    {
        __fp16* ptr = _ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmulq_f16(_p, _sum8);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, _sum4);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

static void softmax_fp16s_unroll8(__fp16* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    float16x8_t _max = vdupq_n_f16(-65504.f);
    {
        const __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _max = vmaxq_f16(_max, _p);
            ptr += stride;
        }
    }

    if (elempack == 8)
    {
        // reduce max 8 to 1
        // broadcast 1 to 8
        _max = vmaxq_f16(_max, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_max))));
        _max = vmaxq_f16(_max, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_u16(_max))));
        _max = vmaxq_f16(_max, vextq_f16(_max, _max, 4));
    }
    if (elempack == 4)
    {
        // reduce max 4,4 to 1,1
        // broadcast 1,1 to 4,4
        _max = vmaxq_f16(_max, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_max))));
        _max = vmaxq_f16(_max, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_u16(_max))));
    }
    if (elempack == 1)
    {
        // fine
    }

    // reduce exp(x - max)
    float16x8_t _sum = vdupq_n_f16(0.f);
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vsubq_f16(_p, _max);
            _p = exp_ps_f16(_p);
            vst1q_f16(ptr, _p);
            _sum = vaddq_f16(_sum, _p);
            ptr += stride;
        }
    }

    if (elempack == 8)
    {
        // reduce sum 8 to 1
        // broadcast 1 to 8
        _sum = vaddq_f16(_sum, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_sum))));
        _sum = vaddq_f16(_sum, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_u16(_sum))));
        _sum = vaddq_f16(_sum, vextq_f16(_sum, _sum, 4));
    }
    if (elempack == 4)
    {
        // reduce sum 4,4 to 1,1
        // broadcast 1,1 to 4,4
        _sum = vaddq_f16(_sum, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_sum))));
        _sum = vaddq_f16(_sum, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_u16(_sum))));
    }
    if (elempack == 1)
    {
        // fine
    }

    _sum = vdivq_f16(vdupq_n_f16(1.f), _sum);

    // div sum
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmulq_f16(_p, _sum);
            vst1q_f16(ptr, _p);
            ptr += stride;
        }
    }
}

static void softmax_fp16s_unroll4(__fp16* _ptr, int elemcount, int elempack, int stride)
{
    // reduce max
    float16x4_t _max = vdup_n_f16(-65504.f);
    {
        const __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _max = vmax_f16(_max, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce max 4 to 1
        // broadcast 1 to 4
        _max = vmax_f16(_max, vreinterpret_f16_u16(vrev32_u16(vreinterpret_u16_f16(_max))));
        _max = vmax_f16(_max, vreinterpret_f16_u32(vrev64_u32(vreinterpret_u32_u16(_max))));
    }
    if (elempack == 1)
    {
        // fine
    }

    // reduce exp(x - max)
    float16x4_t _sum = vdup_n_f16(0.f);
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vsub_f16(_p, _max);
            _p = exp_ps_f16(_p);
            vst1_f16(ptr, _p);
            _sum = vadd_f16(_sum, _p);
            ptr += stride;
        }
    }

    if (elempack == 4)
    {
        // reduce sum 4 to 1
        // broadcast 1 to 4
        _sum = vadd_f16(_sum, vreinterpret_f16_u16(vrev32_u16(vreinterpret_u16_f16(_sum))));
        _sum = vadd_f16(_sum, vreinterpret_f16_u32(vrev64_u32(vreinterpret_u32_u16(_sum))));
    }
    if (elempack == 1)
    {
        // fine
    }

    _sum = vdiv_f16(vdup_n_f16(1.f), _sum);

    // div sum
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, _sum);
            vst1_f16(ptr, _p);
            ptr += stride;
        }
    }
}

static void softmax_fp16s_unroll2(__fp16* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    __fp16 max0 = -65504.f;
    __fp16 max1 = -65504.f;
    {
        const __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max0 = std::max(max0, ptr[0]);
            max1 = std::max(max1, ptr[1]);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    __fp16 sum0 = 0.f;
    __fp16 sum1 = 0.f;
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __fp16 v0 = (__fp16)expf(ptr[0] - max0);
            __fp16 v1 = (__fp16)expf(ptr[1] - max1);
            ptr[0] = v0;
            ptr[1] = v1;
            sum0 += v0;
            sum1 += v1;
            ptr += stride;
        }
    }

    sum0 = (__fp16)(1.f / sum0);
    sum1 = (__fp16)(1.f / sum1);

    // div sum
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            ptr[0] *= sum0;
            ptr[1] *= sum1;
            ptr += stride;
        }
    }
}

static void softmax_fp16s(__fp16* _ptr, int elemcount, int /*elempack*/, int stride)
{
    // assert elempack == 1

    // reduce max
    __fp16 max = -65504.f;
    {
        const __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    __fp16 sum = 0.f;
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            __fp16 v = (__fp16)expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr += stride;
        }
    }

    sum = (__fp16)(1.f / sum);

    // div sum
    {
        __fp16* ptr = _ptr;

        for (int i = 0; i < elemcount; i++)
        {
            *ptr *= sum;
            ptr += stride;
        }
    }
}

int Softmax_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
        __fp16* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_fp16s(ptr, size, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w * elempack;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll8(ptr, h, elempack, size);
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll4(ptr, h, elempack, size);
        }
        for (; i + 1 < size; i += 2)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll2(ptr, h, elempack, size);
        }
        for (; i < size; i++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s(ptr, h, elempack, size);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);

            softmax_fp16s(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d * elempack;
        const int stride = bottom_top_blob.cstep * elempack;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll8(ptr, channels, elempack, stride);
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll4(ptr, channels, elempack, stride);
        }
        for (; i + 1 < size; i += 2)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unroll2(ptr, channels, elempack, stride);
        }
        for (; i < size; i++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s(ptr, channels, elempack, stride);
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax_fp16s(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            const int size = w * h * elempack;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                softmax_fp16s_unroll8(ptr, d, 1, size);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                softmax_fp16s_unroll4(ptr, d, 1, size);
                ptr += 4;
            }
            for (; i + 1 < size; i += 2)
            {
                softmax_fp16s_unroll2(ptr, d, 1, size);
                ptr += 2;
            }
            for (; i < size; i++)
            {
                softmax_fp16s(ptr, d, 1, size);
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
                __fp16* ptr = bottom_top_blob.channel(q).depth(i);

                const int size = w * elempack;

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    softmax_fp16s_unroll8(ptr, h, 1, size);
                    ptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    softmax_fp16s_unroll4(ptr, h, 1, size);
                    ptr += 4;
                }
                for (; j + 1 < size; j += 2)
                {
                    softmax_fp16s_unroll2(ptr, h, 1, size);
                    ptr += 2;
                }
                for (; j < size; j++)
                {
                    softmax_fp16s(ptr, h, 1, size);
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
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_fp16s(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
