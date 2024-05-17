// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

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
#if _MSC_VER
                float16x4_t _slope0 = vcvt_f16_f32(vdupq_n_f32(slope_data[0]));
                float16x8_t _slope = vcombine_f16(_slope0, _slope0);
#else
                float16x8_t _slope = vdupq_n_f16((__fp16)slope_data[0]);
#endif

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
        if (dims == 1)
        {
#if _MSC_VER
            float16x8_t _zero = vdupq_n_f16(0.f);
#else
            float16x4_t _zero = vdup_n_f16(0.f);
#endif

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
#if _MSC_VER
                    uint16x4_t _lemask = vcle_f16(_p, vget_low_f16(_zero));
#else
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
#endif
                    float16x4_t _ps = vmul_f16(_p, _slope);
                    _p = vbsl_f16(_lemask, _ps, _p);
                    vst1_f16(ptr, _p);
                }
            }
            else
            {
#if _MSC_VER
                float16x8_t _slope = vdupq_n_f16((__fp16)slope_data[0]);
#else
                float16x4_t _slope = vdup_n_f16((__fp16)slope_data[0]);
#endif

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                    float16x4_t _p = vld1_f16(ptr);
#if _MSC_VER
                    uint16x4_t _lemask = vcle_f16(_p, vget_low_f16(_zero));
                    float16x4_t _ps = vmul_f16(_p, vget_low_f16(_slope));
#else
                    uint16x4_t _lemask = vcle_f16(_p, _zero);
                    float16x4_t _ps = vmul_f16(_p, _slope);
#endif
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
                float16x4_t _zero = vdup_n_f16(0.f);
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
                float16x4_t _zero = vdup_n_f16(0.f);
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
#if _MSC_VER
            float16x4_t _slope = vcvt_f16_f32(vdupq_n_f32(slope));
#else
            float16x4_t _slope = vdup_n_f16((__fp16)slope);
#endif

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
#if _MSC_VER
            float16x4_t _slope = vcvt_f16_f32(vdupq_n_f32(slope));
#else
            float16x4_t _slope = vdup_n_f16((__fp16)slope);
#endif

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

} // namespace ncnn
