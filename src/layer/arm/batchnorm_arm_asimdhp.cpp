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

#include "batchnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int BatchNorm_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _p = vfmaq_f32(_a, _p, _b);
                vst1_f16(ptr, vcvt_f16_f32(_p));
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + i * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + i * 4);

                __fp16* ptr = bottom_top_blob.row<__fp16>(i);

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    _p = vfmaq_f32(_a, _p, _b);
                    vst1_f16(ptr, vcvt_f16_f32(_p));

                    ptr += 4;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float32x4_t _a = vld1q_f32((const float*)a_data + q * 4);
                float32x4_t _b = vld1q_f32((const float*)b_data + q * 4);

                __fp16* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    _p = vfmaq_f32(_a, _p, _b);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * (float)ptr[i] + a_data[i];
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

            float a = a_data[i];
            float b = b_data[i];

            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _p = vfmaq_f32(_a, _p, _b);
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
            for (; j < w; j++)
            {
                *ptr = b * (float)*ptr + a;

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            float a = a_data[q];
            float b = b_data[q];

            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            int j = 0;
            for (; j + 3 < size; j += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _p = vfmaq_f32(_a, _p, _b);
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
            for (; j < size; j++)
            {
                *ptr = b * (float)*ptr + a;

                ptr++;
            }
        }
    }

    return 0;
}

int BatchNorm_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                __fp16* ptr = (__fp16*)bottom_top_blob + i * 8;

                float16x8_t _a = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 8 + 4)));
                float16x8_t _b = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 8 + 4)));

                float16x8_t _p = vld1q_f16(ptr);
                _p = vfmaq_f16(_a, _p, _b);
                vst1q_f16(ptr, _p);
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float16x8_t _a = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 8 + 4)));
                float16x8_t _b = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 8 + 4)));

                __fp16* ptr = bottom_top_blob.row<__fp16>(i);

                for (int j = 0; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _p = vfmaq_f16(_a, _p, _b);
                    vst1q_f16(ptr, _p);

                    ptr += 8;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float16x8_t _a = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)a_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)a_data + q * 8 + 4)));
                float16x8_t _b = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)b_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)b_data + q * 8 + 4)));

                __fp16* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _p = vfmaq_f16(_a, _p, _b);
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
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                __fp16* ptr = (__fp16*)bottom_top_blob + i * 4;

                float16x4_t _a = vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 4));
                float16x4_t _b = vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 4));

                float16x4_t _p = vld1_f16(ptr);
                _p = vfma_f16(_a, _p, _b);
                vst1_f16(ptr, _p);
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float16x4_t _a = vcvt_f16_f32(vld1q_f32((const float*)a_data + i * 4));
                float16x4_t _b = vcvt_f16_f32(vld1q_f32((const float*)b_data + i * 4));

                __fp16* ptr = bottom_top_blob.row<__fp16>(i);

                for (int j = 0; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _p = vfma_f16(_a, _p, _b);
                    vst1_f16(ptr, _p);

                    ptr += 4;
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int c = bottom_top_blob.c;
            int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float16x4_t _a = vcvt_f16_f32(vld1q_f32((const float*)a_data + q * 4));
                float16x4_t _b = vcvt_f16_f32(vld1q_f32((const float*)b_data + q * 4));

                __fp16* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _p = vfma_f16(_a, _p, _b);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            ptr[i] = (__fp16)b_data[i] * ptr[i] + (__fp16)a_data[i];
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

            __fp16 a = (__fp16)a_data[i];
            __fp16 b = (__fp16)b_data[i];

            float16x4_t _a = vdup_n_f16(a);
#if _MSC_VER
            float16x4_t _b = vcvt_f16_f32(vdupq_n_f32(b_data[i]));
#else
            float16x4_t _b = vdup_n_f16(b);
#endif

            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vfma_f16(_a, _p, _b);
                vst1_f16(ptr, _p);

                ptr += 4;
            }
            for (; j < w; j++)
            {
                *ptr = b * *ptr + a;

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            __fp16 a = (__fp16)a_data[q];
            __fp16 b = (__fp16)b_data[q];

            float16x4_t _a = vdup_n_f16(a);
#if _MSC_VER
            float16x4_t _b = vcvt_f16_f32(vdupq_n_f32(b_data[q]));
#else
            float16x4_t _b = vdup_n_f16(b);
#endif

            int j = 0;
            for (; j + 3 < size; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vfma_f16(_a, _p, _b);
                vst1_f16(ptr, _p);

                ptr += 4;
            }
            for (; j < size; j++)
            {
                *ptr = b * *ptr + a;

                ptr++;
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
