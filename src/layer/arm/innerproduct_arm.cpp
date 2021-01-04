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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON
#include "cpu.h"
#include "neon_activation.h"

namespace ncnn {

InnerProduct_arm::InnerProduct_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
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

    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    if (opt.use_bf16_storage && elembits == 16)
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
            bottom_blob_flattened.cstep = bottom_blob_flattened.w;
            bottom_blob_flattened.elemsize = 4u;
            bottom_blob_flattened.elempack = 1;
        }

        return forward(bottom_blob_flattened, top_blob, opt);
    }
#endif

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;

    int nn_num_output = num_output >> 2;
    int remain_num_output_start = nn_num_output << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = pp * 4;

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;

        if (bias_term)
        {
            sum0 = bias_data[p];
            sum1 = bias_data[p + 1];
            sum2 = bias_data[p + 2];
            sum3 = bias_data[p + 3];
        }

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p + 1);
        const float* w2 = weight_data_ptr + size * channels * (p + 2);
        const float* w3 = weight_data_ptr + size * channels * (p + 3);

#if __ARM_NEON
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size & 3;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            for (; nn > 0; nn--)
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
            for (; remain > 0; remain--)
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
        else if (activation_type == 4)
        {
            sum0 = static_cast<float>(1.f / (1.f + exp(-sum0)));
            sum1 = static_cast<float>(1.f / (1.f + exp(-sum1)));
            sum2 = static_cast<float>(1.f / (1.f + exp(-sum2)));
            sum3 = static_cast<float>(1.f / (1.f + exp(-sum3)));
        }
        else if (activation_type == 5)
        {
            sum0 = static_cast<float>(sum0 * tanh(log(exp(sum0) + 1.f)));
            sum1 = static_cast<float>(sum1 * tanh(log(exp(sum1) + 1.f)));
            sum2 = static_cast<float>(sum2 * tanh(log(exp(sum2) + 1.f)));
            sum3 = static_cast<float>(sum3 * tanh(log(exp(sum3) + 1.f)));
        }

        top_blob[p] = sum0;
        top_blob[p + 1] = sum1;
        top_blob[p + 2] = sum2;
        top_blob[p + 3] = sum3;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_num_output_start; p < num_output; p++)
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
        for (int q = 0; q < channels; q++)
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
                    : "=r"(nn),   // %0
                    "=r"(m),    // %1
                    "=r"(w),    // %2
                    "=w"(_sum), // %3
                    "=w"(_sum2) // %4
                    : "0"(nn),
                    "1"(m),
                    "2"(w),
                    "3"(_sum),
                    "4"(_sum2)
                    : "cc", "memory", "v0", "v1", "v2", "v3");
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
                    : "=r"(nn),   // %0
                    "=r"(m),    // %1
                    "=r"(w),    // %2
                    "=w"(_sum), // %3
                    "=w"(_sum2) // %4
                    : "0"(nn),
                    "1"(m),
                    "2"(w),
                    "3"(_sum),
                    "4"(_sum2)
                    : "cc", "memory", "q0", "q1", "q2", "q3");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int InnerProduct_arm::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        elempack = opt.use_fp16_arithmetic && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }

    // src = inch-outch
    // dst = pb-pa-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_fp16.create(num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g0 = weight_data_fp16.row<__fp16>(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int i = 0; i < elempack; i++)
                {
                    for (int j = 0; j < out_elempack; j++)
                    {
                        *g0++ = (__fp16)(weight_data_r2.row(q + j)[p + i]);
                    }
                }
            }
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int InnerProduct_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    int size = bottom_blob_flattened.w;
    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr));
                float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr + 4));
                float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr + 8));
                float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr + 12));

                _sum = vfmaq_laneq_f32(_sum, _w0, _val, 0);
                _sum = vfmaq_laneq_f32(_sum, _w1, _val, 1);
                _sum = vfmaq_laneq_f32(_sum, _w2, _val, 2);
                _sum = vfmaq_laneq_f32(_sum, _w3, _val, 3);

                sptr += 4;
                kptr += 16;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, vcvt_f16_f32(_sum));
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vdupq_n_f32((float)sptr[0]);

                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));

                _sum = vfmaq_f32(_sum, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, vcvt_f16_f32(_sum));
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            float32x4_t _sum = vdupq_n_f32(0.f);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));

                _sum = vfmaq_f32(_sum, _val, _w);

                sptr += 4;
                kptr += 4;
            }

            sum += vaddvq_f32(_sum); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_fp16.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            float32x4_t _sum = vdupq_n_f32(0.f);
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _m = vcvt_f32_f16(vld1_f16(sptr));
                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));

                _sum = vfmaq_f32(_sum, _m, _w);

                sptr += 4;
                kptr += 4;
            }
            for (; i < size; i++)
            {
                float v = (float)(*sptr);
                float k = (float)(*kptr);

                sum += v * k;

                sptr++;
                kptr++;
            }

            sum += vaddvq_f32(_sum);

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}

int InnerProduct_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    int size = bottom_blob_flattened.w;
    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 8 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x8_t _sum = vdupq_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int nn = size; // size always > 0

            asm volatile(
                "eor    v1.16b, v1.16b, v1.16b      \n"
                "eor    v2.16b, v2.16b, v2.16b      \n"
                "eor    v3.16b, v3.16b, v3.16b      \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.8h}, [%2], #16          \n" // _val

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                "fmla   %1.8h, v8.8h, v0.h[0]       \n"
                "fmla   v1.8h, v9.8h, v0.h[1]       \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                "fmla   v2.8h, v10.8h, v0.h[2]      \n"
                "fmla   v3.8h, v11.8h, v0.h[3]      \n"
                "fmla   %1.8h, v12.8h, v0.h[4]      \n"
                "fmla   v1.8h, v13.8h, v0.h[5]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v2.8h, v14.8h, v0.h[6]      \n"
                "fmla   v3.8h, v15.8h, v0.h[7]      \n"

                "bne    0b                          \n"

                "fadd   %1.8h, %1.8h, v1.8h         \n"
                "fadd   v2.8h, v2.8h, v3.8h         \n"
                "fadd   %1.8h, %1.8h, v2.8h         \n"

                : "=r"(nn),   // %0
                "=w"(_sum), // %1
                "=r"(sptr), // %2
                "=r"(kptr)  // %3
                : "0"(nn),
                "1"(_sum),
                "2"(sptr),
                "3"(kptr)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1q_f16(outptr + p * 8, _sum);
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x8_t _sum = vdupq_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float16x8_t _val = vdupq_n_f16(sptr[0]);

                float16x8_t _w = vld1q_f16(kptr);

                _sum = vfmaq_f16(_sum, _val, _w);

                sptr += 1;
                kptr += 8;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1q_f16(outptr + p * 8, _sum);
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x8_t _sum = vdupq_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float16x4_t _val = vld1_f16(sptr);

                float16x8_t _w0 = vld1q_f16(kptr);
                float16x8_t _w1 = vld1q_f16(kptr + 8);
                float16x8_t _w2 = vld1q_f16(kptr + 16);
                float16x8_t _w3 = vld1q_f16(kptr + 24);

                _sum = vfmaq_lane_f16(_sum, _w0, _val, 0);
                _sum = vfmaq_lane_f16(_sum, _w1, _val, 1);
                _sum = vfmaq_lane_f16(_sum, _w2, _val, 2);
                _sum = vfmaq_lane_f16(_sum, _w3, _val, 3);

                sptr += 4;
                kptr += 32;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1q_f16(outptr + p * 8, _sum);
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            float16x8_t _sum = vdupq_n_f16(0.f);

            for (int i = 0; i < size; i++)
            {
                float16x8_t _val = vld1q_f16(sptr);

                float16x8_t _w = vld1q_f16(kptr);

                _sum = vfmaq_f16(_sum, _val, _w);

                sptr += 8;
                kptr += 8;
            }

            float16x4_t _s4 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
            sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = sum;
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x4_t _sum = vdup_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float16x8_t _val = vld1q_f16(sptr);

                float16x4_t _w0 = vld1_f16(kptr);
                float16x4_t _w1 = vld1_f16(kptr + 4);
                float16x4_t _w2 = vld1_f16(kptr + 8);
                float16x4_t _w3 = vld1_f16(kptr + 12);
                float16x4_t _w4 = vld1_f16(kptr + 16);
                float16x4_t _w5 = vld1_f16(kptr + 20);
                float16x4_t _w6 = vld1_f16(kptr + 24);
                float16x4_t _w7 = vld1_f16(kptr + 28);

                _sum = vfma_laneq_f16(_sum, _w0, _val, 0);
                _sum = vfma_laneq_f16(_sum, _w1, _val, 1);
                _sum = vfma_laneq_f16(_sum, _w2, _val, 2);
                _sum = vfma_laneq_f16(_sum, _w3, _val, 3);
                _sum = vfma_laneq_f16(_sum, _w4, _val, 4);
                _sum = vfma_laneq_f16(_sum, _w5, _val, 5);
                _sum = vfma_laneq_f16(_sum, _w6, _val, 6);
                _sum = vfma_laneq_f16(_sum, _w7, _val, 7);

                sptr += 8;
                kptr += 32;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, _sum);
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x4_t _sum = vdup_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float16x4_t _val = vld1_f16(sptr);

                float16x4_t _w0 = vld1_f16(kptr);
                float16x4_t _w1 = vld1_f16(kptr + 4);
                float16x4_t _w2 = vld1_f16(kptr + 8);
                float16x4_t _w3 = vld1_f16(kptr + 12);

                _sum = vfma_lane_f16(_sum, _w0, _val, 0);
                _sum = vfma_lane_f16(_sum, _w1, _val, 1);
                _sum = vfma_lane_f16(_sum, _w2, _val, 2);
                _sum = vfma_lane_f16(_sum, _w3, _val, 3);

                sptr += 4;
                kptr += 16;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, _sum);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x4_t _sum = vdup_n_f16(0.f);

            if (bias_term)
            {
                _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float16x4_t _val = vdup_n_f16(sptr[0]);

                float16x4_t _w = vld1_f16(kptr);

                _sum = vfma_f16(_sum, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, _sum);
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            float16x4_t _sum = vdup_n_f16(0.f);

            for (int i = 0; i < size; i++)
            {
                float16x4_t _val = vld1_f16(sptr);

                float16x4_t _w = vld1_f16(kptr);

                _sum = vfma_f16(_sum, _val, _w);

                sptr += 4;
                kptr += 4;
            }

            sum += vaddvq_f32(vcvt_f32_f16(_sum)); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_fp16.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            float16x8_t _sum = vdupq_n_f16(0.f);
            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _m = vld1q_f16(sptr);
                float16x8_t _w = vld1q_f16(kptr);

                _sum = vfmaq_f16(_sum, _m, _w);

                sptr += 8;
                kptr += 8;
            }
            for (; i < size; i++)
            {
                __fp16 v = *sptr;
                __fp16 k = *kptr;

                sum += (float)(v * k);

                sptr++;
                kptr++;
            }

            float16x4_t _s4 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
            sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int InnerProduct_arm::create_pipeline_bf16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int elempack = opt.use_packing_layout && num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;

    // src = inch-outch
    // dst = pb-pa-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_bf16.create(num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g0 = weight_data_bf16.row<unsigned short>(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int i = 0; i < elempack; i++)
                {
                    for (int j = 0; j < out_elempack; j++)
                    {
                        *g0++ = float32_to_bfloat16(weight_data_r2.row(q + j)[p + i]);
                    }
                }
            }
        }
    }

    return 0;
}

int InnerProduct_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    int size = bottom_blob_flattened.w;
    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr));
                float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr + 4));
                float32x4_t _w2 = vcvt_f32_bf16(vld1_u16(kptr + 8));
                float32x4_t _w3 = vcvt_f32_bf16(vld1_u16(kptr + 12));

#if __aarch64__
                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                sptr += 4;
                kptr += 16;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            vst1_u16(outptr + p * 4, vcvt_bf16_f32(_sum));
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[0]));

                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));

                _sum = vmlaq_f32(_sum, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            vst1_u16(outptr + p * 4, vcvt_bf16_f32(_sum));
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            float32x4_t _sum = vdupq_n_f32(0.f);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));

                _sum = vmlaq_f32(_sum, _val, _w);

                sptr += 4;
                kptr += 4;
            }

#if __aarch64__
            sum += vaddvq_f32(_sum); // dot
#else
            float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _ss = vpadd_f32(_ss, _ss);
            sum += vget_lane_f32(_ss, 0);
#endif

            sum = activation_ss(sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            outptr[p] = float32_to_bfloat16(sum);
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const unsigned short* kptr = weight_data_bf16.row<unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            int i = 0;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _m = vcvt_f32_bf16(vld1_u16(sptr));
                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));

                _sum = vmlaq_f32(_sum, _m, _w);

                sptr += 4;
                kptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*sptr);
                float k = bfloat16_to_float32(*kptr);

                sum += v * k;

                sptr++;
                kptr++;
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

            unsigned short* outptr = (unsigned short*)top_blob;
            outptr[p] = float32_to_bfloat16(sum);
        }
    }

    return 0;
}

} // namespace ncnn
