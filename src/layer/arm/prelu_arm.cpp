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

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

PReLU_arm::PReLU_arm()
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

int PReLU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

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

            int j = 0;
#if __ARM_NEON
            float32x4_t _zero = vdupq_n_f32(0.f);
            float32x4_t _slope = vdupq_n_f32(slope);

            for (; j + 15 < size; j += 16)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _p2 = vld1q_f32(ptr + 8);
                float32x4_t _p3 = vld1q_f32(ptr + 12);
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
                uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                float32x4_t _ps2 = vmulq_f32(_p2, _slope);
                float32x4_t _ps3 = vmulq_f32(_p3, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                _p2 = vbslq_f32(_lemask2, _ps2, _p2);
                _p3 = vbslq_f32(_lemask3, _ps3, _p3);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                vst1q_f32(ptr + 8, _p2);
                vst1q_f32(ptr + 12, _p3);
                ptr += 16;
            }
            for (; j + 7 < size; j += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
                uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
                float32x4_t _ps0 = vmulq_f32(_p0, _slope);
                float32x4_t _ps1 = vmulq_f32(_p1, _slope);
                _p0 = vbslq_f32(_lemask0, _ps0, _p0);
                _p1 = vbslq_f32(_lemask1, _ps1, _p1);
                vst1q_f32(ptr, _p0);
                vst1q_f32(ptr + 4, _p1);
                ptr += 8;
            }
            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; j < size; j++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
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

                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _slope = vld1q_f32(slope + i * 4);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, float2bfloat(_p));
                }
            }
            else
            {
                float32x4_t _slope = vdupq_n_f32(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, float2bfloat(_p));
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
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, float2bfloat(_p));

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
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1_u16(ptr, float2bfloat(_p));

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
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_u16(ptr, float2bfloat(_p));

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
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                uint32x4_t _lemask = vcleq_f32(_p, _zero);
                float32x4_t _ps = vmulq_f32(_p, _slope);
                _p = vbslq_f32(_lemask, _ps, _p);
                vst1_u16(ptr, float2bfloat(_p));

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
#endif // NCNN_BF16

} // namespace ncnn
