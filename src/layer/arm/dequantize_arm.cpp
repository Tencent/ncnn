// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dequantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Dequantize_arm::Dequantize_arm()
{
#if __ARM_NEON
    support_packing = true;
// #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
//     support_fp16_storage = true;
// #endif
#endif // __ARM_NEON

    //     support_bf16_storage = true;
}

int Dequantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    //     int elembits = bottom_blob.elembits();

    // #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    //     if (opt.use_fp16_storage && elembits == 16)
    //     {
    //         if (opt.use_fp16_arithmetic)
    //             return forward_fp16sa(bottom_blob, top_blob, opt);
    //         else
    //             return forward_fp16s(bottom_blob, top_blob, opt);
    //     }
    // #endif

    //     if (opt.use_bf16_storage && elembits == 16)
    //         return forward_bf16s(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1q_f32(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                float32x4_t _bias = vdupq_n_f32(bias);
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmlaq_f32(_bias, _v, _scale);
                    vst1q_f32(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1q_f32(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                float32x4_t _bias = vdupq_n_f32(bias);
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmlaq_f32(_bias, _v, _scale);
                    vst1q_f32(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
