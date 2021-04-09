// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "requantize_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#if __ARM_NEON
#include "requantize_relu_pack4.h"
#include "requantize_relu_pack8.h"
#endif // __ARM_NEON

Requantize_arm::Requantize_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int Requantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_in_data_size == 1 && scale_out_data_size == 1)
            {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data[0]);
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in);
                        _v1 = vmulq_f32(_v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias, _v0, _scale_in);
                        _v1 = vmlaq_f32(_bias, _v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
            }
            else if (scale_in_data_size == 1 && scale_out_data_size > 1)
            {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in);
                        _v1 = vmulq_f32(_v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias, _v0, _scale_in);
                        _v1 = vmlaq_f32(_bias, _v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
            }
            else if (scale_in_data_size > 1 && scale_out_data_size == 1)
            {
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in0);
                        _v1 = vmulq_f32(_v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out);
                        _v1 = vmulq_f32(_v1, _scale_out);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
            }
            else // if (scale_in_data_size > 1 && scale_out_data_size > 1)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in0);
                        _v1 = vmulq_f32(_v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        signed char* ptr = (signed char*)top_blob + i * 8;

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    signed char* ptr = top_blob.row<signed char>(i);

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in0);
                        _v1 = vmulq_f32(_v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));

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
                    signed char* ptr = top_blob.row<signed char>(i);

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));

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

            top_blob.create(w, h, channels, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (activation_type == 1)
            {
                requantize_relu_pack8_neon(bottom_blob, top_blob, scale_in_data, scale_out_data, bias_data, opt);
                return 0;
            }

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    signed char* ptr = top_blob.channel(q);

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmulq_f32(_v0, _scale_in0);
                        _v1 = vmulq_f32(_v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));

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
                    signed char* ptr = top_blob.channel(q);

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                        _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                        _v0 = activation_ps(_v0, activation_type, activation_params);
                        _v1 = activation_ps(_v1, activation_type, activation_params);
                        _v0 = vmulq_f32(_v0, _scale_out0);
                        _v1 = vmulq_f32(_v1, _scale_out1);
                        vst1_s8(ptr, float2int8(_v0, _v1));

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
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_in_data_size == 1 && scale_out_data_size == 1)
            {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data[0]);
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
            }
            else if (scale_in_data_size == 1 && scale_out_data_size > 1)
            {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
            }
            else if (scale_in_data_size > 1 && scale_out_data_size == 1)
            {
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
            }
            else // if (scale_in_data_size > 1 && scale_out_data_size > 1)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else if (bias_data_size == 1)
                {
                    float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        signed char* ptr = (signed char*)top_blob + i * 4;

                        float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmlaq_f32(_bias, _v, _scale_in);
                        _v = activation_ps(_v, activation_type, activation_params);
                        _v = vmulq_f32(_v, _scale_out);
                        int8x8_t v = float2int8(_v, _v);
                        ptr[0] = vget_lane_s8(v, 0);
                        ptr[1] = vget_lane_s8(v, 1);
                        ptr[2] = vget_lane_s8(v, 2);
                        ptr[3] = vget_lane_s8(v, 3);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
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
                        signed char* ptr = top_blob.row<signed char>(i);

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

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
                        signed char* ptr = top_blob.row<signed char>(i);

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const int* intptr = bottom_blob.row<const int>(i);
                        signed char* ptr0 = top_blob.row<signed char>(i * 4);
                        signed char* ptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* ptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* ptr3 = top_blob.row<signed char>(i * 4 + 3);

                        float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr0[0] = vget_lane_s8(v, 0);
                            ptr1[0] = vget_lane_s8(v, 1);
                            ptr2[0] = vget_lane_s8(v, 2);
                            ptr3[0] = vget_lane_s8(v, 3);

                            intptr += 4;
                            ptr0 += 1;
                            ptr1 += 1;
                            ptr2 += 1;
                            ptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const int* intptr = bottom_blob.row<const int>(i);
                        signed char* ptr0 = top_blob.row<signed char>(i * 4);
                        signed char* ptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* ptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* ptr3 = top_blob.row<signed char>(i * 4 + 3);

                        float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + i * 4);
                        float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + i * 4);
                        float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr0[0] = vget_lane_s8(v, 0);
                            ptr1[0] = vget_lane_s8(v, 1);
                            ptr2[0] = vget_lane_s8(v, 2);
                            ptr3[0] = vget_lane_s8(v, 3);

                            intptr += 4;
                            ptr0 += 1;
                            ptr1 += 1;
                            ptr2 += 1;
                            ptr3 += 1;
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
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (activation_type == 1)
            {
                requantize_relu_pack4_neon(bottom_blob, top_blob, scale_in_data, scale_out_data, bias_data, opt);
                return 0;
            }

            if (out_elempack == 8)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const int* intptr0 = bottom_blob.channel(q * 2);
                        const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* ptr = top_blob.channel(q);

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const int* intptr0 = bottom_blob.channel(q * 2);
                        const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* ptr = top_blob.channel(q);

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr0 += 4;
                            intptr1 += 4;
                            ptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const int* intptr = bottom_blob.channel(q);
                        signed char* ptr0 = top_blob.channel(q * 4);
                        signed char* ptr1 = top_blob.channel(q * 4 + 1);
                        signed char* ptr2 = top_blob.channel(q * 4 + 2);
                        signed char* ptr3 = top_blob.channel(q * 4 + 3);

                        float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 4);
                        float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr0[0] = vget_lane_s8(v, 0);
                            ptr1[0] = vget_lane_s8(v, 1);
                            ptr2[0] = vget_lane_s8(v, 2);
                            ptr3[0] = vget_lane_s8(v, 3);

                            intptr += 4;
                            ptr0 += 1;
                            ptr1 += 1;
                            ptr2 += 1;
                            ptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const int* intptr = bottom_blob.channel(q);
                        signed char* ptr0 = top_blob.channel(q * 4);
                        signed char* ptr1 = top_blob.channel(q * 4 + 1);
                        signed char* ptr2 = top_blob.channel(q * 4 + 2);
                        signed char* ptr3 = top_blob.channel(q * 4 + 3);

                        float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data[0]) : vld1q_f32((const float*)scale_in_data + q * 4);
                        float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data[0]) : vld1q_f32((const float*)scale_out_data + q * 4);
                        float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr0[0] = vget_lane_s8(v, 0);
                            ptr1[0] = vget_lane_s8(v, 1);
                            ptr2[0] = vget_lane_s8(v, 2);
                            ptr3[0] = vget_lane_s8(v, 3);

                            intptr += 4;
                            ptr0 += 1;
                            ptr1 += 1;
                            ptr2 += 1;
                            ptr3 += 1;
                        }
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

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        signed char* ptr = top_blob;

        if (scale_in_data_size == 1 && scale_out_data_size == 1)
        {
            const float scale_in = scale_in_data[0];
            const float scale_out = scale_out_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else if (scale_in_data_size == 1 && scale_out_data_size > 1)
        {
            const float scale_in = scale_in_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
        }
        else if (scale_in_data_size > 1 && scale_out_data_size == 1)
        {
            const float scale_out = scale_out_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else // if (scale_in_data_size > 1 && scale_out_data_size > 1)
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[i];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[i];

                for (int j = 0; j < w; j++)
                {
                    float v = intptr[j] * scale_in;
                    ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[i];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                for (int j = 0; j < w; j++)
                {
                    float v = intptr[j] * scale_in + bias;
                    ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
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

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];

                for (int i = 0; i < size; i++)
                {
                    float v = intptr[i] * scale_in;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                for (int i = 0; i < size; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
