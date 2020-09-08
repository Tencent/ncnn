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

#include "prelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

PReLU_arm::PReLU_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int PReLU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _zero = vdupq_n_f32(0.f);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _slope = vld1q_f32(slope + i * 4);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);
                }
            }
            else
            {
                float32x4_t _slope = vdupq_n_f32(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + i * 4) : vdupq_n_f32(slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + q * 4) : vdupq_n_f32(slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope[i];
            }
        }
        else
        {
            const float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < w; j++)
            {
                float v = *ptr;
                if (v < 0.f)
                    *ptr = v * slope;

                ptr++;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "veor       q1, q0, q0          \n"
                    "vdup.f32   q2, %4              \n"
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vcle.f32   q3, q0, q1          \n"
                    "vmul.f32   q4, q0, q2          \n"
                    "vbit.32    q0, q4, q3          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(ptr) // %1
                    : "0"(nn),
                    "1"(ptr),
                    "r"(slope) // %4
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int PReLU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        float32x4_t _zero = vdupq_n_f32(0.f);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    float32x4_t _slope = vld1q_f32(slope + i * 4);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_f16(ptr, vcvt_f16_f32(_p));
                }
            }
            else
            {
                float32x4_t _slope = vdupq_n_f32(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_f16(ptr, vcvt_f16_f32(_p));
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + i * 4) : vdupq_n_f32(slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_f16(ptr, vcvt_f16_f32(_p));

                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + q * 4) : vdupq_n_f32(slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_f16(ptr, vcvt_f16_f32(_p));

                    ptr += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        __fp16* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = (float)ptr[i];
                if (v < 0.f)
                    ptr[i] = (__fp16)(v * slope[i]);
            }
        }
        else
        {
            const float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = (float)ptr[i];
                if (v < 0.f)
                    ptr[i] = (__fp16)(v * slope);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
            for (; j < w; j++)
            {
                float v = (float)*ptr;
                if (v < 0.f)
                    *ptr = (__fp16)(v * slope);

                ptr++;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            const float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            int j = 0;
            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
            for (; j < size; j++)
            {
                float v = (float)*ptr;
                if (v < 0.f)
                    *ptr = (__fp16)(v * slope);

                ptr++;
            }
        }
    }

    return 0;
}

int PReLU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        float16x8_t _zero = vdupq_n_f16(0.f);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 8;

                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _slope = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)slope + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)slope + i * 8 + 4)));
                    uint16x8_t _lemask = vcleq_f16(_p, _zero);
                    float16x8_t _ps = vmulq_f16(_p, _slope);
                    _p = vbslq_f16(_lemask, _ps, _p);
                    vst1q_f16(ptr, _p);
                }
            }
            else
            {
                float16x8_t _slope = vdupq_n_f16((__fp16)slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 8;

                    float16x8_t _p = vld1q_f16(ptr);
                    uint16x8_t _lemask = vcleq_f16(_p, _zero);
                    float16x8_t _ps = vmulq_f16(_p, _slope);
                    _p = vbslq_f16(_lemask, _ps, _p);
                    vst1q_f16(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                float16x8_t _slope = num_slope > 1 ? vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)slope_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)slope_data + i * 8 + 4))) : vdupq_n_f16((__fp16)slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    uint16x8_t _lemask = vcleq_f16(_p, _zero);
                    float16x8_t _ps = vmulq_f16(_p, _slope);
                    _p = vbslq_f16(_lemask, _ps, _p);
                    vst1q_f16(ptr, _p);

                    ptr += 8;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                float16x8_t _slope = num_slope > 1 ? vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)slope_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)slope_data + q * 8 + 4))) : vdupq_n_f16((__fp16)slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    uint16x8_t _lemask = vcleq_f16(_p, _zero);
                    float16x8_t _ps = vmulq_f16(_p, _slope);
                    _p = vbslq_f16(_lemask, _ps, _p);
                    vst1q_f16(ptr, _p);

                    ptr += 8;
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        float16x4_t _zero = vdup_n_f16(0.f);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _slope = vcvt_f16_f32(vld1q_f32(slope + i * 4));
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
                    float16x4_t _ps = vmul_f16(_p, _slope);
                    _p = vbsl_f16(_lemask, _ps, _p);
                    vst1_f16(ptr, _p);
                }
            }
            else
            {
                float16x4_t _slope = vdup_n_f16((__fp16)slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                    float16x4_t _p = vld1_f16(ptr);
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
                    float16x4_t _ps = vmul_f16(_p, _slope);
                    _p = vbsl_f16(_lemask, _ps, _p);
                    vst1_f16(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                float16x4_t _slope = num_slope > 1 ? vcvt_f16_f32(vld1q_f32((const float*)slope_data + i * 4)) : vdup_n_f16((__fp16)slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
                    float16x4_t _ps = vmul_f16(_p, _slope);
                    _p = vbsl_f16(_lemask, _ps, _p);
                    vst1_f16(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                float16x4_t _slope = num_slope > 1 ? vcvt_f16_f32(vld1q_f32((const float*)slope_data + q * 4)) : vdup_n_f16((__fp16)slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
                    float16x4_t _ps = vmul_f16(_p, _slope);
                    _p = vbsl_f16(_lemask, _ps, _p);
                    vst1_f16(ptr, _p);

                    ptr += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        __fp16* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = (float)ptr[i];
                if (v < (__fp16)0.f)
                    ptr[i] = (__fp16)(v * slope[i]);
            }
        }
        else
        {
            const float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = (float)ptr[i];
                if (v < (__fp16)0.f)
                    ptr[i] = (__fp16)(v * slope);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            float16x4_t _zero = vdup_n_f16(0.f);
            float16x4_t _slope = vdup_n_f16((__fp16)slope);

            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                uint16x4_t _lemask = vcle_f16(_p, _zero);
                float16x4_t _ps = vmul_f16(_p, _slope);
                _p = vbsl_f16(_lemask, _ps, _p);
                vst1_f16(ptr, _p);

                ptr += 4;
            }
            for (; j < w; j++)
            {
                float v = (float)*ptr;
                if (v < (__fp16)0.f)
                    *ptr = (__fp16)(v * slope);

                ptr++;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            const float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            float16x4_t _zero = vdup_n_f16(0.f);
            float16x4_t _slope = vdup_n_f16((__fp16)slope);

            int j = 0;
            for (; j + 3 < size; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                uint16x4_t _lemask = vcle_f16(_p, _zero);
                float16x4_t _ps = vmul_f16(_p, _slope);
                _p = vbsl_f16(_lemask, _ps, _p);
                vst1_f16(ptr, _p);

                ptr += 4;
            }
            for (; j < size; j++)
            {
                float v = (float)*ptr;
                if (v < (__fp16)0.f)
                    *ptr = (__fp16)(v * slope);

                ptr++;
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int PReLU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _zero = vdupq_n_f32(0.f);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _slope = vld1q_f32(slope + i * 4);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, vcvt_bf16_f32(_p));
                }
            }
            else
            {
                float32x4_t _slope = vdupq_n_f32(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, vcvt_bf16_f32(_p));
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + i * 4) : vdupq_n_f32(slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, vcvt_bf16_f32(_p));

                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                float32x4_t _slope = num_slope > 1 ? vld1q_f32((const float*)slope_data + q * 4) : vdupq_n_f32(slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, vcvt_bf16_f32(_p));

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        unsigned short* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope[i]);
            }
        }
        else
        {
            const float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_u16(ptr, vcvt_bf16_f32(_p));

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < w; j++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    *ptr = float32_to_bfloat16(v * slope);

                ptr++;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            const float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            int j = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_u16(ptr, vcvt_bf16_f32(_p));

                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < size; j++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    *ptr = float32_to_bfloat16(v * slope);

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
