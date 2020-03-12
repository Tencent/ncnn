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

#include "innerproduct_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON
#include "neon_activation.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct_arm)

InnerProduct_arm::InnerProduct_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;

    flatten = 0;
}

int InnerProduct_arm::create_pipeline(const Option& opt)
{
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }
#endif // __ARM_NEON

    if (opt.use_bf16_storage)
    {
        ncnn::cast_float32_to_bfloat16(weight_data, weight_data_bf16, opt);
    }

    return 0;
}

int InnerProduct_arm::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int InnerProduct_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }

    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __ARM_NEON
    if (elempack == 4)
    {
        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        if (bottom_blob.dims != 1)
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        }

        // pack1
        {
            bottom_blob_flattened.w *= bottom_blob_flattened.elempack;
            bottom_blob_flattened.elemsize = 4u;
            bottom_blob_flattened.elempack = 1;
        }

        return forward(bottom_blob_flattened, top_blob, opt);
    }
#endif // __ARM_NEON

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;

    int nn_num_output = num_output >> 2;
    int remain_num_output_start = nn_num_output << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_num_output; pp++)
    {
        int p = pp * 4;

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;

        if (bias_term)
        {
            sum0 = bias_data[p];
            sum1 = bias_data[p+1];
            sum2 = bias_data[p+2];
            sum3 = bias_data[p+3];
        }

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p+1);
        const float* w2 = weight_data_ptr + size * channels * (p+2);
        const float* w3 = weight_data_ptr + size * channels * (p+3);

#if __ARM_NEON
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size & 3;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            for (; nn>0; nn--)
            {
                float32x4_t _m = vld1q_f32(m);

                float32x4_t _w0 = vld1q_f32(w0);
                _sum0 = vmlaq_f32(_sum0, _m, _w0);

                float32x4_t _w1 = vld1q_f32(w1);
                _sum1 = vmlaq_f32(_sum1, _m, _w1);

                float32x4_t _w2 = vld1q_f32(w2);
                _sum2 = vmlaq_f32(_sum2, _m, _w2);

                float32x4_t _w3 = vld1q_f32(w3);
                _sum3 = vmlaq_f32(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                sum0 += *m * *w0;
                sum1 += *m * *w1;
                sum2 += *m * *w2;
                sum3 += *m * *w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

        }

#if __ARM_NEON
        float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
        float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
        float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
        float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

        float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
        float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

        sum0 += vget_lane_f32(_sum01ss, 0);
        sum1 += vget_lane_f32(_sum01ss, 1);
        sum2 += vget_lane_f32(_sum23ss, 0);
        sum3 += vget_lane_f32(_sum23ss, 1);

#endif // __ARM_NEON

        if (activation_type == 1)
        {
            sum0 = std::max(sum0, 0.f);
            sum1 = std::max(sum1, 0.f);
            sum2 = std::max(sum2, 0.f);
            sum3 = std::max(sum3, 0.f);
        }
        else if (activation_type == 2)
        {
            float slope = activation_params[0];
            sum0 = sum0 > 0.f ? sum0 : sum0 * slope;
            sum1 = sum1 > 0.f ? sum1 : sum1 * slope;
            sum2 = sum2 > 0.f ? sum2 : sum2 * slope;
            sum3 = sum3 > 0.f ? sum3 : sum3 * slope;
        }
        else if (activation_type == 3)
        {
            float min = activation_params[0];
            float max = activation_params[1];
            if (sum0 < min) sum0 = min;
            if (sum0 > max) sum0 = max;
            if (sum1 < min) sum1 = min;
            if (sum1 > max) sum1 = max;
            if (sum2 < min) sum2 = min;
            if (sum2 > max) sum2 = max;
            if (sum3 < min) sum3 = min;
            if (sum3 > max) sum3 = max;
        }

        top_blob[p] = sum0;
        top_blob[p+1] = sum1;
        top_blob[p+2] = sum2;
        top_blob[p+3] = sum3;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_num_output_start; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        const float* w = weight_data_ptr + size * channels * p;

#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                                   \n"
                "prfm       pldl1keep, [%1, #256]     \n"
                "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                "prfm       pldl1keep, [%2, #256]     \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                "fmla       %3.4s, v0.4s, v2.4s       \n"
                "subs       %w0, %w0, #1              \n"
                "fmla       %4.4s, v1.4s, v3.4s       \n"
                "bne        0b                        \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.f32   {d0-d3}, [%1 :128]! \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2]!      \n"
                "vmla.f32   %q3, q0, q2         \n"
                "subs       %0, #1              \n"
                "vmla.f32   %q4, q1, q3         \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                sum += *m * *w;

                m++;
                w++;
            }
        }

#if __ARM_NEON
        _sum = vaddq_f32(_sum, _sum2);
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _sumss = vpadd_f32(_sumss, _sumss);
        sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

        sum = activation_ss(sum, activation_type, activation_params);

        top_blob[p] = sum;
    }

    return 0;
}

int InnerProduct_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __ARM_NEON
    if (elempack == 4)
    {
        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        if (bottom_blob.dims != 1)
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        }

        // pack1
        {
            bottom_blob_flattened.w *= bottom_blob_flattened.elempack;
            bottom_blob_flattened.elemsize = 2u;
            bottom_blob_flattened.elempack = 1;
        }

        return forward(bottom_blob_flattened, top_blob, opt);
    }
#endif // __ARM_NEON

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const unsigned short* weight_data_ptr = weight_data_bf16;

    int nn_num_output = num_output >> 2;
    int remain_num_output_start = nn_num_output << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_num_output; pp++)
    {
        int p = pp * 4;

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;

        if (bias_term)
        {
            sum0 = bias_data[p];
            sum1 = bias_data[p+1];
            sum2 = bias_data[p+2];
            sum3 = bias_data[p+3];
        }

        const unsigned short* w0 = weight_data_ptr + size * channels * p;
        const unsigned short* w1 = weight_data_ptr + size * channels * (p+1);
        const unsigned short* w2 = weight_data_ptr + size * channels * (p+2);
        const unsigned short* w3 = weight_data_ptr + size * channels * (p+3);

#if __ARM_NEON
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const unsigned short* m = bottom_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i+3<size; i+=4)
            {
                float32x4_t _m = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(m), 16));
                float32x4_t _w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w0), 16));
                float32x4_t _w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w1), 16));
                float32x4_t _w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w2), 16));
                float32x4_t _w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w3), 16));

                _sum0 = vmlaq_f32(_sum0, _m, _w0);
                _sum1 = vmlaq_f32(_sum1, _m, _w1);
                _sum2 = vmlaq_f32(_sum2, _m, _w2);
                _sum3 = vmlaq_f32(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __ARM_NEON
            for (; i<size; i++)
            {
                float _m = bfloat16_to_float32(*m);
                float _w0 = bfloat16_to_float32(*w0);
                float _w1 = bfloat16_to_float32(*w1);
                float _w2 = bfloat16_to_float32(*w2);
                float _w3 = bfloat16_to_float32(*w3);

                sum0 += _m * _w0;
                sum1 += _m * _w1;
                sum2 += _m * _w2;
                sum3 += _m * _w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }
        }

#if __ARM_NEON
        float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
        float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
        float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
        float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

        float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
        float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

        sum0 += vget_lane_f32(_sum01ss, 0);
        sum1 += vget_lane_f32(_sum01ss, 1);
        sum2 += vget_lane_f32(_sum23ss, 0);
        sum3 += vget_lane_f32(_sum23ss, 1);

#endif // __ARM_NEON

        if (activation_type == 1)
        {
            sum0 = std::max(sum0, 0.f);
            sum1 = std::max(sum1, 0.f);
            sum2 = std::max(sum2, 0.f);
            sum3 = std::max(sum3, 0.f);
        }
        else if (activation_type == 2)
        {
            float slope = activation_params[0];
            sum0 = sum0 > 0.f ? sum0 : sum0 * slope;
            sum1 = sum1 > 0.f ? sum1 : sum1 * slope;
            sum2 = sum2 > 0.f ? sum2 : sum2 * slope;
            sum3 = sum3 > 0.f ? sum3 : sum3 * slope;
        }
        else if (activation_type == 3)
        {
            float min = activation_params[0];
            float max = activation_params[1];
            if (sum0 < min) sum0 = min;
            if (sum0 > max) sum0 = max;
            if (sum1 < min) sum1 = min;
            if (sum1 > max) sum1 = max;
            if (sum2 < min) sum2 = min;
            if (sum2 > max) sum2 = max;
            if (sum3 < min) sum3 = min;
            if (sum3 > max) sum3 = max;
        }

        unsigned short* outptr = (unsigned short*)top_blob + p;
        outptr[0] = float32_to_bfloat16(sum0);
        outptr[1] = float32_to_bfloat16(sum1);
        outptr[2] = float32_to_bfloat16(sum2);
        outptr[3] = float32_to_bfloat16(sum3);
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_num_output_start; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        const unsigned short* w = weight_data_ptr + size * channels * p;

#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const unsigned short* m = bottom_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i+3<size; i+=4)
            {
                float32x4_t _m = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(m), 16));
                float32x4_t _w = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16));

                _sum = vmlaq_f32(_sum, _m, _w);

                m += 4;
                w += 4;
            }
#endif // __ARM_NEON
            for (; i<size; i++)
            {
                float _m = bfloat16_to_float32(*m);
                float _w = bfloat16_to_float32(*w);

                sum += _m * _w;

                m++;
                w++;
            }
        }

#if __ARM_NEON
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _sumss = vpadd_f32(_sumss, _sumss);
        sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

        sum = activation_ss(sum, activation_type, activation_params);

        unsigned short* outptr = (unsigned short*)top_blob + p;
        outptr[0] = float32_to_bfloat16(sum);
    }

    return 0;
}

} // namespace ncnn
