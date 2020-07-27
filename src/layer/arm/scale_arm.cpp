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

#include "scale_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Scale_arm::Scale_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int Scale_arm::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            const float* scale = scale_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _s = vld1q_f32(scale + i * 4);
                    float32x4_t _bias = vld1q_f32(bias + i * 4);
                    _p = vmlaq_f32(_bias, _p, _s);
                    vst1q_f32(ptr, _p);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _s = vld1q_f32(scale + i * 4);
                    _p = vmulq_f32(_p, _s);
                    vst1q_f32(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + i * 4);
                    float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmlaq_f32(_bias, _p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + q * 4);
                    float32x4_t _bias = vld1q_f32((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmlaq_f32(_bias, _p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims != 3)
        return Scale::forward_inplace(bottom_top_blobs, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* scale_ptr = scale_blob;
        const float* bias_ptr = bias_data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];
            float bias = bias_ptr[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _s = vdupq_n_f32(s);
            float32x4_t _bias = vdupq_n_f32(bias);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmlaq_f32(_bias, _p, _s);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                *ptr = *ptr * s + bias;

                ptr++;
            }
        }
    }
    else
    {
        const float* scale_ptr = scale_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _s = vdupq_n_f32(s);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmulq_f32(_p, _s);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                *ptr *= s;

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
