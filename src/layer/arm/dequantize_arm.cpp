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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Dequantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
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
                    for (int i = 0; i < outw; i++)
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
                    for (int i = 0; i < outw; i++)
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
                    for (int i = 0; i < outw; i++)
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
                    for (int i = 0; i < outw; i++)
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
                    for (int i = 0; i < outw; i++)
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
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1q_f32(ptr0, _v0);
                        vst1q_f32(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1q_f32(ptr0, _v0);
                        vst1q_f32(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1q_f32(ptr0, _v0);
                        vst1q_f32(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1q_f32(ptr0, _v0);
                        vst1q_f32(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        return 0;
    }

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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Dequantize_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr0 = top_blob.row<__fp16>(i * 2);
                    __fp16* ptr1 = top_blob.row<__fp16>(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_f16(ptr0, vcvt_f16_f32(_v0));
                        vst1_f16(ptr1, vcvt_f16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr0 = top_blob.row<__fp16>(i * 2);
                    __fp16* ptr1 = top_blob.row<__fp16>(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1_f16(ptr0, vcvt_f16_f32(_v0));
                        vst1_f16(ptr1, vcvt_f16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr0 = top_blob.channel(q * 2);
                    __fp16* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_f16(ptr0, vcvt_f16_f32(_v0));
                        vst1_f16(ptr1, vcvt_f16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr0 = top_blob.channel(q * 2);
                    __fp16* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1_f16(ptr0, vcvt_f16_f32(_v0));
                        vst1_f16(ptr1, vcvt_f16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, elempack, opt.blob_allocator);
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
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
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
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));

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
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));

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

            top_blob.create(w, h, channels, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));

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
                    __fp16* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_f16(ptr, vcvt_f16_f32(_v));

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        __fp16* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias_data[i]);
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
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1_f16(ptr, vcvt_f16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
                float32x4_t _scale = vdupq_n_f32(scale);
                float32x4_t _bias = vdupq_n_f32(bias);
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmlaq_f32(_bias, _v, _scale);
                    vst1_f16(ptr, vcvt_f16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale + bias);
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
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1_f16(ptr, vcvt_f16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
                float32x4_t _scale = vdupq_n_f32(scale);
                float32x4_t _bias = vdupq_n_f32(bias);
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmlaq_f32(_bias, _v, _scale);
                    vst1_f16(ptr, vcvt_f16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale + bias);
                }
            }
        }
    }

    return 0;
}

int Dequantize_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)16u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float16x8_t _scale = vdupq_n_f16((__fp16)scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vmulq_f16(_v, _scale);
                        vst1q_f16(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float16x8_t _bias = vdupq_n_f16((__fp16)bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8 + 4)));
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);
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
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vmulq_f16(_v, _scale);
                        vst1q_f16(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float16x8_t _bias = vdupq_n_f16((__fp16)bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                        float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8 + 4)));
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));

                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vmulq_f16(_v, _scale);
                        vst1q_f16(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                    float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8 + 4)));

                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);

                        intptr += 8;
                        ptr += 8;
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

            top_blob.create(w, h, channels, (size_t)16u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8 + 4)));

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vmulq_f16(_v, _scale);
                        vst1q_f16(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8 + 4)));
                    float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + q * 8 + 4)));

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr + 4))));
                        _v = vfmaq_f16(_bias, _v, _scale);
                        vst1q_f16(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 4;
            size_t out_elemsize = out_elempack * 2u;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float16x4_t _scale = vdup_n_f16((__fp16)scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vmul_f16(_v, _scale);
                        vst1_f16(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float16x4_t _bias = vdup_n_f16((__fp16)bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vfma_f16(_bias, _v, _scale);
                        vst1_f16(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _bias = vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 4));
                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vfma_f16(_bias, _v, _scale);
                        vst1_f16(ptr, _v);
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
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _scale = vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 4));
                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vmul_f16(_v, _scale);
                        vst1_f16(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float16x4_t _bias = vdup_n_f16((__fp16)bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _scale = vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 4));
                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vfma_f16(_bias, _v, _scale);
                        vst1_f16(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        float16x4_t _scale = vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 4));
                        float16x4_t _bias = vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 4));
                        float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                        _v = vfma_f16(_bias, _v, _scale);
                        vst1_f16(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 4;
            size_t out_elemsize = out_elempack * 2u;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const int* intptr0 = bottom_blob.row<const int>(i * 2);
                        const int* intptr1 = bottom_blob.row<const int>(i * 2 + 1);
                        __fp16* ptr = top_blob.row<__fp16>(i);

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));

                        for (int j = 0; j < w; j++)
                        {
                            float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr0))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr1))));
                            _v = vmulq_f16(_v, _scale);
                            vst1q_f16(ptr, _v);

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const int* intptr0 = bottom_blob.row<const int>(i * 2);
                        const int* intptr1 = bottom_blob.row<const int>(i * 2 + 1);
                        __fp16* ptr = top_blob.row<__fp16>(i);

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                        float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 8 + 4)));

                        for (int j = 0; j < w; j++)
                        {
                            float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr0))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr1))));
                            _v = vfmaq_f16(_bias, _v, _scale);
                            vst1q_f16(ptr, _v);

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 4)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const int* intptr = bottom_blob.row<const int>(i);
                        __fp16* ptr = top_blob.row<__fp16>(i);

                        float16x4_t _scale = scale_data_size == 1 ? vdup_n_f16((__fp16)scale_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 4));

                        for (int j = 0; j < w; j++)
                        {
                            float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                            _v = vmul_f16(_v, _scale);
                            vst1_f16(ptr, _v);

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
                        __fp16* ptr = top_blob.row<__fp16>(i);

                        float16x4_t _scale = scale_data_size == 1 ? vdup_n_f16((__fp16)scale_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 4));
                        float16x4_t _bias = bias_data_size == 1 ? vdup_n_f16((__fp16)bias_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)bias_data + i * 4));

                        for (int j = 0; j < w; j++)
                        {
                            float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                            _v = vfma_f16(_bias, _v, _scale);
                            vst1_f16(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
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
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 4;
            size_t out_elemsize = out_elempack * 2u;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const int* intptr0 = bottom_blob.channel(q * 2);
                        const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                        __fp16* ptr = top_blob.channel(q);

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8 + 4)));

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr0))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr1))));
                            _v = vmulq_f16(_v, _scale);
                            vst1q_f16(ptr, _v);

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const int* intptr0 = bottom_blob.channel(q * 2);
                        const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                        __fp16* ptr = top_blob.channel(q);

                        float16x8_t _scale = scale_data_size == 1 ? vdupq_n_f16((__fp16)scale_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8 + 4)));
                        float16x8_t _bias = bias_data_size == 1 ? vdupq_n_f16((__fp16)bias_data[0]) : vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)bias_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)bias_data + q * 8 + 4)));

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _v = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr0))), vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr1))));
                            _v = vfmaq_f16(_bias, _v, _scale);
                            vst1q_f16(ptr, _v);

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 4)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const int* intptr = bottom_blob.channel(q);
                        __fp16* ptr = top_blob.channel(q);

                        float16x4_t _scale = scale_data_size == 1 ? vdup_n_f16((__fp16)scale_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 4));

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                            _v = vmul_f16(_v, _scale);
                            vst1_f16(ptr, _v);

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
                        __fp16* ptr = top_blob.channel(q);

                        float16x4_t _scale = scale_data_size == 1 ? vdup_n_f16((__fp16)scale_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 4));
                        float16x4_t _bias = bias_data_size == 1 ? vdup_n_f16((__fp16)bias_data[0]) : vcvt_f16_f32(vld1q_f32((const float*)bias_data + q * 4));

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                            _v = vfma_f16(_bias, _v, _scale);
                            vst1_f16(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        __fp16* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const __fp16 scale = (__fp16)scale_data[0];

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
                const __fp16 bias = (__fp16)bias_data[0];

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
                    ptr[i] = intptr[i] * scale + (__fp16)bias_data[i];
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
                    ptr[i] = intptr[i] * (__fp16)scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const __fp16 bias = (__fp16)bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * (__fp16)scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * (__fp16)scale_data[i] + (__fp16)bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const __fp16 scale = scale_data_size == 1 ? (__fp16)scale_data[0] : (__fp16)scale_data[i];

                int j = 0;
                float16x4_t _scale = vdup_n_f16(scale);
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                    _v = vmul_f16(_v, _scale);
                    vst1_f16(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
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
                __fp16* ptr = top_blob.row<__fp16>(i);

                const __fp16 scale = scale_data_size == 1 ? (__fp16)scale_data[0] : (__fp16)scale_data[i];
                const __fp16 bias = bias_data_size == 1 ? (__fp16)bias_data[0] : (__fp16)bias_data[i];

                int j = 0;
                float16x4_t _scale = vdup_n_f16(scale);
                float16x4_t _bias = vdup_n_f16(bias);
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                    _v = vfma_f16(_bias, _v, _scale);
                    vst1_f16(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
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
                __fp16* ptr = top_blob.channel(q);

                const __fp16 scale = scale_data_size == 1 ? (__fp16)scale_data[0] : (__fp16)scale_data[q];

                int i = 0;
                float16x4_t _scale = vdup_n_f16(scale);
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                    _v = vmul_f16(_v, _scale);
                    vst1_f16(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
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
                __fp16* ptr = top_blob.channel(q);

                const __fp16 scale = scale_data_size == 1 ? (__fp16)scale_data[0] : (__fp16)scale_data[q];
                const __fp16 bias = bias_data_size == 1 ? (__fp16)bias_data[0] : (__fp16)bias_data[q];

                int i = 0;
                float16x4_t _scale = vdup_n_f16(scale);
                float16x4_t _bias = vdup_n_f16(bias);
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _v = vcvt_f16_f32(vcvtq_f32_s32(vld1q_s32(intptr)));
                    _v = vfma_f16(_bias, _v, _scale);
                    vst1_f16(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int Dequantize_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    unsigned short* ptr0 = top_blob.row<unsigned short>(i * 2);
                    unsigned short* ptr1 = top_blob.row<unsigned short>(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_u16(ptr0, vcvt_bf16_f32(_v0));
                        vst1_u16(ptr1, vcvt_bf16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    unsigned short* ptr0 = top_blob.row<unsigned short>(i * 2);
                    unsigned short* ptr1 = top_blob.row<unsigned short>(i * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1_u16(ptr0, vcvt_bf16_f32(_v0));
                        vst1_u16(ptr1, vcvt_bf16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    unsigned short* ptr0 = top_blob.channel(q * 2);
                    unsigned short* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_u16(ptr0, vcvt_bf16_f32(_v0));
                        vst1_u16(ptr1, vcvt_bf16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    unsigned short* ptr0 = top_blob.channel(q * 2);
                    unsigned short* ptr1 = top_blob.channel(q * 2 + 1);

                    float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                        vst1_u16(ptr0, vcvt_bf16_f32(_v0));
                        vst1_u16(ptr1, vcvt_bf16_f32(_v1));

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, elempack, opt.blob_allocator);
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
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
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
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        unsigned short* ptr = (unsigned short*)top_blob + i * 4;

                        float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    unsigned short* ptr = top_blob.row<unsigned short>(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));

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
                    unsigned short* ptr = top_blob.row<unsigned short>(i);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));

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

            top_blob.create(w, h, channels, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    unsigned short* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));

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
                    unsigned short* ptr = top_blob.channel(q);

                    float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale);
                        vst1_u16(ptr, vcvt_bf16_f32(_v));

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

        top_blob.create(w, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        unsigned short* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale + bias_data[i]);
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
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale_data[i] + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = float32_to_bfloat16(intptr[i] * scale_data[i] + bias_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                unsigned short* ptr = top_blob.row<unsigned short>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1_u16(ptr, vcvt_bf16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr++ = float32_to_bfloat16(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                unsigned short* ptr = top_blob.row<unsigned short>(i);

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
                    vst1_u16(ptr, vcvt_bf16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr++ = float32_to_bfloat16(*intptr++ * scale + bias);
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

        top_blob.create(w, h, channels, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                unsigned short* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    _v = vmulq_f32(_v, _scale);
                    vst1_u16(ptr, vcvt_bf16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr++ = float32_to_bfloat16(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                unsigned short* ptr = top_blob.channel(q);

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
                    vst1_u16(ptr, vcvt_bf16_f32(_v));

                    intptr += 4;
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr++ = float32_to_bfloat16(*intptr++ * scale + bias);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
