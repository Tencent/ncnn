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

#include "cpu.h"

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
    sum = (__fp16)1.f / sum;

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

static void softmax_fp16s_unrolln(__fp16* _ptr, int elemcount, int elempack, int stride, int unroll, __fp16* _maxptr, __fp16* _sumptr)
{
    // reduce max
    // __fp16 max[unroll];
    {
        float16x8_t _negmax = vdupq_n_f16(-65504.f);

        __fp16* maxptr = _maxptr;

        int k = 0;
        for (; k + 7 < unroll; k += 8)
        {
            vst1q_f16(maxptr, _negmax);
            maxptr += 8;
        }
        for (; k + 3 < unroll; k += 4)
        {
            vst1_f16(maxptr, vget_low_f16(_negmax));
            maxptr += 4;
        }
        for (; k < unroll; k++)
        {
            *maxptr++ = -65504.f;
        }
    }
    {
        for (int i = 0; i < elemcount; i++)
        {
            const __fp16* ptr = _ptr + i * stride;
            __fp16* maxptr = _maxptr;

            int k = 0;
            for (; k + 7 < unroll; k += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _max = vld1q_f16(maxptr);
                _max = vmaxq_f16(_max, _p);
                vst1q_f16(maxptr, _max);
                ptr += 8;
                maxptr += 8;
            }
            for (; k + 3 < unroll; k += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _max = vld1_f16(maxptr);
                _max = vmax_f16(_max, _p);
                vst1_f16(maxptr, _max);
                ptr += 4;
                maxptr += 4;
            }
            for (; k < unroll; k++)
            {
                *maxptr = std::max(*maxptr, *ptr);
                ptr++;
                maxptr++;
            }
        }
    }

    if (elempack == 8)
    {
        // reduce max 8,8,8... to 1,1,1...
        // broadcast 1,1,1... to 8,8,8...
        __fp16* maxptr = _maxptr;
        for (int k = 0; k + 7 < unroll; k += 8)
        {
            float16x8_t _max = vld1q_f16(maxptr);
            _max = vmaxq_f16(_max, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_max))));
            _max = vmaxq_f16(_max, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_f16(_max))));
            _max = vmaxq_f16(_max, vextq_f16(_max, _max, 4));
            vst1q_f16(maxptr, _max);
            maxptr += 8;
        }
    }
    if (elempack == 4)
    {
        // reduce max 4,4,4,4,4,4... to 1,1,1,1,1,1...
        // broadcast 1,1,1,1,1,1... to 4,4,4,4,4,4...
        __fp16* maxptr = _maxptr;
        int k = 0;
        for (; k + 7 < unroll; k += 8)
        {
            float16x8_t _max = vld1q_f16(maxptr);
            _max = vmaxq_f16(_max, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_max))));
            _max = vmaxq_f16(_max, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_f16(_max))));
            vst1q_f16(maxptr, _max);
            maxptr += 8;
        }
        for (; k + 3 < unroll; k += 4)
        {
            float16x4_t _max = vld1_f16(maxptr);
            _max = vmax_f16(_max, vreinterpret_f16_u16(vrev32_u16(vreinterpret_u16_f16(_max))));
            _max = vmax_f16(_max, vreinterpret_f16_u32(vrev64_u32(vreinterpret_u32_f16(_max))));
            vst1_f16(maxptr, _max);
            maxptr += 4;
        }
    }

    // reduce exp(x - max)
    // __fp16 sum[unroll];
    {
        float16x8_t _zero = vdupq_n_f16(0.f);

        __fp16* sumptr = _sumptr;

        int k = 0;
        for (; k + 7 < unroll; k += 8)
        {
            vst1q_f16(sumptr, _zero);
            sumptr += 8;
        }
        for (; k + 3 < unroll; k += 4)
        {
            vst1_f16(sumptr, vget_low_f16(_zero));
            sumptr += 4;
        }
        for (; k < unroll; k++)
        {
            *sumptr++ = 0.f;
        }
    }
    {
        for (int i = 0; i < elemcount; i++)
        {
            __fp16* ptr = _ptr + i * stride;
            const __fp16* maxptr = _maxptr;
            __fp16* sumptr = _sumptr;

            int k = 0;
            for (; k + 7 < unroll; k += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _max = vld1q_f16(maxptr);
                float16x8_t _sum = vld1q_f16(sumptr);
                _p = vsubq_f16(_p, _max);
                _p = exp_ps_f16(_p);
                vst1q_f16(ptr, _p);
                _sum = vaddq_f16(_sum, _p);
                vst1q_f16(sumptr, _sum);
                ptr += 8;
                maxptr += 8;
                sumptr += 8;
            }
            for (; k + 3 < unroll; k += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _max = vld1_f16(maxptr);
                float16x4_t _sum = vld1_f16(sumptr);
                _p = vsub_f16(_p, _max);
                _p = exp_ps_f16(_p);
                vst1_f16(ptr, _p);
                _sum = vadd_f16(_sum, _p);
                vst1_f16(sumptr, _sum);
                ptr += 4;
                maxptr += 4;
                sumptr += 4;
            }
            for (; k < unroll; k++)
            {
                __fp16 v = expf(*ptr - *maxptr);
                *ptr = v;
                *sumptr += v;
                ptr++;
                maxptr++;
                sumptr++;
            }
        }
    }

    if (elempack == 8)
    {
        // reduce sum 8,8,8... to 1,1,1...
        // broadcast 1,1,1... to 8,8,8...
        __fp16* sumptr = _sumptr;
        for (int k = 0; k + 7 < unroll; k += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vaddq_f16(_sum, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_sum))));
            _sum = vaddq_f16(_sum, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_f16(_sum))));
            _sum = vaddq_f16(_sum, vextq_f16(_sum, _sum, 4));
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
    }
    if (elempack == 4)
    {
        // reduce sum 4,4,4,4,4,4... to 1,1,1,1,1,1...
        // broadcast 1,1,1,1,1,1... to 4,4,4,4,4,4...
        __fp16* sumptr = _sumptr;
        int k = 0;
        for (; k + 7 < unroll; k += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vaddq_f16(_sum, vreinterpretq_f16_u16(vrev32q_u16(vreinterpretq_u16_f16(_sum))));
            _sum = vaddq_f16(_sum, vreinterpretq_f16_u32(vrev64q_u32(vreinterpretq_u32_f16(_sum))));
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
        for (; k + 3 < unroll; k += 4)
        {
            float16x4_t _sum = vld1_f16(sumptr);
            _sum = vadd_f16(_sum, vreinterpret_f16_u16(vrev32_u16(vreinterpret_u16_f16(_sum))));
            _sum = vadd_f16(_sum, vreinterpret_f16_u32(vrev64_u32(vreinterpret_u32_f16(_sum))));
            vst1_f16(sumptr, _sum);
            sumptr += 4;
        }
    }

    {
        float16x8_t _one = vdupq_n_f16(1.f);
        __fp16* sumptr = _sumptr;
        int k = 0;
        for (; k + 7 < unroll; k += 8)
        {
            float16x8_t _sum = vld1q_f16(sumptr);
            _sum = vdivq_f16(_one, _sum);
            vst1q_f16(sumptr, _sum);
            sumptr += 8;
        }
        for (; k + 3 < unroll; k += 4)
        {
            float16x4_t _sum = vld1_f16(sumptr);
            _sum = vdiv_f16(vget_low_f16(_one), _sum);
            vst1_f16(sumptr, _sum);
            sumptr += 4;
        }
        for (; k < unroll; k++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    {
        for (int i = 0; i < elemcount; i++)
        {
            __fp16* ptr = _ptr + i * stride;
            const __fp16* sumptr = _sumptr;

            int k = 0;
            for (; k + 7 < unroll; k += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _sum = vld1q_f16(sumptr);
                _p = vmulq_f16(_p, _sum);
                vst1q_f16(ptr, _p);
                ptr += 8;
                sumptr += 8;
            }
            for (; k + 3 < unroll; k += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _sum = vld1_f16(sumptr);
                _p = vmul_f16(_p, _sum);
                vst1_f16(ptr, _p);
                ptr += 4;
                sumptr += 4;
            }
            for (; k < unroll; k++)
            {
                *ptr *= *sumptr;
                ptr++;
                sumptr++;
            }
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

        const int sizen = (w + (opt.num_threads - 1)) / opt.num_threads * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        int nn_size = std::max(size / sizen, 1);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + sizen;

            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unrolln(ptr, h, elempack, size, size1, maxptr, sumptr);
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

        const int sizen = (w * h * d + (opt.num_threads - 1)) / opt.num_threads * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        int nn_size = std::max(size / sizen, 1);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + sizen;

            __fp16* ptr = (__fp16*)bottom_top_blob + i;

            softmax_fp16s_unrolln(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                __fp16* ptr = bottom_top_blob.channel(q).depth(i);

                __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
                __fp16* maxptr = maxsumptr;
                __fp16* sumptr = maxptr + size;

                softmax_fp16s_unrolln(ptr, h, 1, size, size, maxptr, sumptr);
            }
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
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            __fp16* maxsumptr = maxsum.channel(get_omp_thread_num());
            __fp16* maxptr = maxsumptr;
            __fp16* sumptr = maxptr + size;

            softmax_fp16s_unrolln(ptr, d, 1, size, size, maxptr, sumptr);
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
