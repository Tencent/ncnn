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
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

InnerProduct_arm::InnerProduct_arm()
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

    flatten = 0;
}

int InnerProduct_arm::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_arm(opt);
    }
#endif

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

#if NCNN_VFPV4
    if (cpu_support_arm_vfpv4() && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    if (out_elempack == 4)
    {
        // src = inch-outch
        // dst = pb-inch-outch/pb
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_tm.create(num_input, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                float* g0 = weight_data_tm.row(q / out_elempack);

                for (int p = 0; p < num_input; p++)
                {
                    for (int j = 0; j < out_elempack; j++)
                    {
                        *g0++ = weight_data_r2.row(q + j)[p];
                    }
                }
            }
        }
    }
    else
    {
        weight_data_tm = weight_data;
    }

    weight_data.release();

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
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_arm(bottom_blob, top_blob, opt);
    }
#endif

    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

#if NCNN_VFPV4
    if (cpu_support_arm_vfpv4() && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __ARM_NEON
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum0 = vdupq_n_f32(bias_data[p * 4 + 0]);
                        _sum1 = vdupq_n_f32(bias_data[p * 4 + 1]);
                        _sum2 = vdupq_n_f32(bias_data[p * 4 + 2]);
                        _sum3 = vdupq_n_f32(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        float32x4_t _val = vld1q_f32(m);
                        float32x4_t _w = vld1q_f32(kptr);
#if __aarch64__
                        _sum0 = vfmaq_laneq_f32(_sum0, _val, _w, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _val, _w, 1);
                        _sum2 = vfmaq_laneq_f32(_sum2, _val, _w, 2);
                        _sum3 = vfmaq_laneq_f32(_sum3, _val, _w, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _val, vget_low_f32(_w), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _val, vget_low_f32(_w), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _val, vget_high_f32(_w), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _val, vget_high_f32(_w), 1);
#endif
                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    vst1q_f32(outptr, _sum0);
                    vst1q_f32(outptr + 4, _sum1);
                    vst1q_f32(outptr + 8, _sum2);
                    vst1q_f32(outptr + 12, _sum3);
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum0 = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        float32x4_t _val = vld1q_f32(m);

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 8);
                        float32x4_t _w3 = vld1q_f32(kptr + 12);

#if __aarch64__
                        _sum0 = vfmaq_laneq_f32(_sum0, _w0, _val, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _w1, _val, 1);
                        _sum2 = vfmaq_laneq_f32(_sum2, _w2, _val, 2);
                        _sum3 = vfmaq_laneq_f32(_sum3, _w3, _val, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_val), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_val), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_val), 1);
#endif

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        float32x4_t _val = vld1q_dup_f32(m);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum0 = vmlaq_f32(_sum0, _val, _k);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = vaddq_f32(_sum0, _sum1);
                    _sum2 = vaddq_f32(_sum2, _sum3);
                    _sum0 = vaddq_f32(_sum0, _sum2);

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);

                    vst1q_f32(outptr, _sum0);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vdupq_n_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vld1q_f32(m);
                        float32x4_t _k = vdupq_n_f32(kptr[0]);
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f32(outptr, _sum);
                    outptr += 4;
                }
            }
#endif // __ARM_NEON

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    int i = 0;
#if __ARM_NEON
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (; i + 3 < num_input; i += 4)
                    {
                        float32x4_t _val = vld1q_f32(m);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 4;
                        kptr += 4;
                    }
#if __aarch64__
                    sum += vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);
                    sum += vget_lane_f32(_ss, 0);
#endif
#endif // __ARM_NEON
                    for (; i < num_input; i++)
                    {
                        sum += *m * *kptr;

                        m += 1;
                        kptr += 1;
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum0 = bias_term ? vld1q_f32((const float*)bias_data + p * 4) : vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
#if NCNN_GNU_INLINE_ASM
            for (; i + 7 < num_input; i += 8)
            {
#if __aarch64__
                asm volatile(
                    "prfm       pldl1keep, [%0, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%0], #32 \n"
                    "prfm       pldl1keep, [%1, #512]     \n"
                    "ld1        {v2.4s, v3.4s, v4.4s, v5.4s}, [%1], #64 \n"
                    "prfm       pldl1keep, [%1, #512]     \n"
                    "ld1        {v6.4s, v7.4s, v8.4s, v9.4s}, [%1], #64 \n"
                    "fmla       %2.4s, v2.4s, v0.s[0]     \n"
                    "fmla       %3.4s, v3.4s, v0.s[1]     \n"
                    "fmla       %4.4s, v4.4s, v0.s[2]     \n"
                    "fmla       %5.4s, v5.4s, v0.s[3]     \n"
                    "fmla       %2.4s, v6.4s, v1.s[0]     \n"
                    "fmla       %3.4s, v7.4s, v1.s[1]     \n"
                    "fmla       %4.4s, v8.4s, v1.s[2]     \n"
                    "fmla       %5.4s, v9.4s, v1.s[3]     \n"
                    : "=r"(sptr),  // %0
                    "=r"(kptr),  // %1
                    "=w"(_sum0), // %2
                    "=w"(_sum1), // %3
                    "=w"(_sum2), // %4
                    "=w"(_sum3)  // %5
                    : "0"(sptr),
                    "1"(kptr),
                    "2"(_sum0),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]! \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d4-d11}       \n"
                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d12-d19}      \n"
                    "vmla.f32   %q2, q2, d0[0]      \n"
                    "vmla.f32   %q3, q3, d0[1]      \n"
                    "vmla.f32   %q4, q4, d1[0]      \n"
                    "vmla.f32   %q5, q5, d1[1]      \n"
                    "vmla.f32   %q2, q6, d2[0]      \n"
                    "vmla.f32   %q3, q7, d2[1]      \n"
                    "vmla.f32   %q4, q8, d3[0]      \n"
                    "vmla.f32   %q5, q9, d3[1]      \n"
                    : "=r"(sptr),  // %0
                    "=r"(kptr),  // %1
                    "=w"(_sum0), // %2
                    "=w"(_sum1), // %3
                    "=w"(_sum2), // %4
                    "=w"(_sum3)  // %5
                    : "0"(sptr),
                    "1"(kptr),
                    "2"(_sum0),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9");
#endif
            }
#endif // NCNN_GNU_INLINE_ASM
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _val = vld1q_f32(sptr);

                float32x4_t _w0 = vld1q_f32(kptr);
                float32x4_t _w1 = vld1q_f32(kptr + 4);
                float32x4_t _w2 = vld1q_f32(kptr + 8);
                float32x4_t _w3 = vld1q_f32(kptr + 12);

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _w0, _val, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _w1, _val, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _w2, _val, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _w3, _val, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_val), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_val), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_val), 1);
#endif

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                float32x4_t _val = vld1q_dup_f32(sptr);
                float32x4_t _w = vld1q_f32(kptr);
                _sum0 = vmlaq_f32(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _sum0 = vaddq_f32(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            vst1q_f32(outptr + p * 4, _sum0);
        }
    }
#endif // __ARM_NEON

    if (out_elempack == 1)
    {
        const float* weight_data_ptr = weight_data_tm;

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

            const float* w0 = weight_data_ptr + num_input * p;
            const float* w1 = weight_data_ptr + num_input * (p + 1);
            const float* w2 = weight_data_ptr + num_input * (p + 2);
            const float* w3 = weight_data_ptr + num_input * (p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __ARM_NEON
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
#if NCNN_GNU_INLINE_ASM
            for (; i + 7 < num_input; i += 8)
            {
#if __aarch64__
                asm volatile(
                    "prfm       pldl1keep, [%0, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%0], #32 \n"
                    "prfm       pldl1keep, [%1, #256]     \n"
                    "ld1        {v2.4s, v3.4s}, [%1], #32 \n"
                    "prfm       pldl1keep, [%2, #256]     \n"
                    "ld1        {v4.4s, v5.4s}, [%2], #32 \n"
                    "prfm       pldl1keep, [%3, #256]     \n"
                    "ld1        {v6.4s, v7.4s}, [%3], #32 \n"
                    "prfm       pldl1keep, [%4, #256]     \n"
                    "ld1        {v8.4s, v9.4s}, [%4], #32 \n"
                    "fmla       %5.4s, v0.4s, v2.4s       \n"
                    "fmla       %6.4s, v0.4s, v4.4s       \n"
                    "fmla       %7.4s, v0.4s, v6.4s       \n"
                    "fmla       %8.4s, v0.4s, v8.4s       \n"
                    "fmla       %5.4s, v1.4s, v3.4s       \n"
                    "fmla       %6.4s, v1.4s, v5.4s       \n"
                    "fmla       %7.4s, v1.4s, v7.4s       \n"
                    "fmla       %8.4s, v1.4s, v9.4s       \n"
                    : "=r"(m),     // %0
                    "=r"(w0),    // %1
                    "=r"(w1),    // %2
                    "=r"(w2),    // %3
                    "=r"(w3),    // %4
                    "=w"(_sum0), // %5
                    "=w"(_sum1), // %6
                    "=w"(_sum2), // %7
                    "=w"(_sum3)  // %8
                    : "0"(m),
                    "1"(w0),
                    "2"(w1),
                    "3"(w2),
                    "4"(w3),
                    "5"(_sum0),
                    "6"(_sum1),
                    "7"(_sum2),
                    "8"(_sum3)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]! \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d4-d7}, [%1]!      \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d8-d11}, [%2]!     \n"
                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d12-d15}, [%3]!    \n"
                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d16-d19}, [%4]!    \n"
                    "vmla.f32   %q5, q0, q2         \n"
                    "vmla.f32   %q6, q0, q4         \n"
                    "vmla.f32   %q7, q0, q6         \n"
                    "vmla.f32   %q8, q0, q8         \n"
                    "vmla.f32   %q5, q1, q3         \n"
                    "vmla.f32   %q6, q1, q5         \n"
                    "vmla.f32   %q7, q1, q7         \n"
                    "vmla.f32   %q8, q1, q9         \n"
                    : "=r"(m),     // %0
                    "=r"(w0),    // %1
                    "=r"(w1),    // %2
                    "=r"(w2),    // %3
                    "=r"(w3),    // %4
                    "=w"(_sum0), // %5
                    "=w"(_sum1), // %6
                    "=w"(_sum2), // %7
                    "=w"(_sum3)  // %8
                    : "0"(m),
                    "1"(w0),
                    "2"(w1),
                    "3"(w2),
                    "4"(w3),
                    "5"(_sum0),
                    "6"(_sum1),
                    "7"(_sum2),
                    "8"(_sum3)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9");
#endif // __aarch64__
            }
#endif // NCNN_GNU_INLINE_ASM
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _val = vld1q_f32(m);

                float32x4_t _w0 = vld1q_f32(w0);
                float32x4_t _w1 = vld1q_f32(w1);
                float32x4_t _w2 = vld1q_f32(w2);
                float32x4_t _w3 = vld1q_f32(w3);

                _sum0 = vmlaq_f32(_sum0, _val, _w0);
                _sum1 = vmlaq_f32(_sum1, _val, _w1);
                _sum2 = vmlaq_f32(_sum2, _val, _w2);
                _sum3 = vmlaq_f32(_sum3, _val, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }

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
            for (; i < num_input; i++)
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

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);
            sum2 = activation_ss(sum2, activation_type, activation_params);
            sum3 = activation_ss(sum3, activation_type, activation_params);

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

            const float* w = weight_data_ptr + num_input * p;

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
#if NCNN_GNU_INLINE_ASM
            for (; i + 7 < num_input; i += 8)
            {
#if __aarch64__
                asm volatile(
                    "prfm       pldl1keep, [%0, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%0], #32 \n"
                    "prfm       pldl1keep, [%1, #256]     \n"
                    "ld1        {v2.4s, v3.4s}, [%1], #32 \n"
                    "fmla       %2.4s, v0.4s, v2.4s       \n"
                    "fmla       %3.4s, v1.4s, v3.4s       \n"
                    : "=r"(m),    // %0
                    "=r"(w),    // %1
                    "=w"(_sum), // %2
                    "=w"(_sum2) // %3
                    : "0"(m),
                    "1"(w),
                    "2"(_sum),
                    "3"(_sum2)
                    : "cc", "memory", "v0", "v1", "v2", "v3");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]! \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d4-d7}, [%1]!      \n"
                    "vmla.f32   %q2, q0, q2         \n"
                    "vmla.f32   %q3, q1, q3         \n"
                    : "=r"(m),    // %0
                    "=r"(w),    // %1
                    "=w"(_sum), // %2
                    "=w"(_sum2) // %3
                    : "0"(m),
                    "1"(w),
                    "2"(_sum),
                    "3"(_sum2)
                    : "cc", "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
            }
#endif // NCNN_GNU_INLINE_ASM
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _val = vld1q_f32(m);
                float32x4_t _w = vld1q_f32(w);
                _sum = vmlaq_f32(_sum, _val, _w);
                m += 4;
                w += 4;
            }

            _sum = vaddq_f32(_sum, _sum2);
#if __aarch64__
            sum += vaddvq_f32(_sum);
#else
            float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _sumss = vpadd_f32(_sumss, _sumss);
            sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON
            for (; i < num_input; i++)
            {
                sum += *m * *w;

                m++;
                w++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            top_blob[p] = sum;
        }
    }

    return 0;
}

#if NCNN_BF16
int InnerProduct_arm::create_pipeline_bf16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = float32_to_bfloat16(weight_data_r2.row(q + j)[p]);
                }
            }
        }
    }

    weight_data.release();

    return 0;
}

int InnerProduct_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __ARM_NEON
            if (elempack == 4 && num_output_elempack == 4)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_tm + num_input * p * 4;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum0 = vdupq_n_f32(bias_data[p * 4 + 0]);
                        _sum1 = vdupq_n_f32(bias_data[p * 4 + 1]);
                        _sum2 = vdupq_n_f32(bias_data[p * 4 + 2]);
                        _sum3 = vdupq_n_f32(bias_data[p * 4 + 3]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = bfloat2float(vld1_u16(m));
                        float32x4_t _k = bfloat2float(vld1_u16(kptr));
#if __aarch64__
                        _sum0 = vfmaq_laneq_f32(_sum0, _val, _k, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _val, _k, 1);
                        _sum2 = vfmaq_laneq_f32(_sum2, _val, _k, 2);
                        _sum3 = vfmaq_laneq_f32(_sum3, _val, _k, 3);
#else
                        _sum0 = vmlaq_lane_f32(_sum0, _val, vget_low_f32(_k), 0);
                        _sum1 = vmlaq_lane_f32(_sum1, _val, vget_low_f32(_k), 1);
                        _sum2 = vmlaq_lane_f32(_sum2, _val, vget_high_f32(_k), 0);
                        _sum3 = vmlaq_lane_f32(_sum3, _val, vget_high_f32(_k), 1);
#endif

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    vst1_u16(outptr, float2bfloat(_sum0));
                    vst1_u16(outptr + 4, float2bfloat(_sum1));
                    vst1_u16(outptr + 8, float2bfloat(_sum2));
                    vst1_u16(outptr + 12, float2bfloat(_sum3));
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_tm + num_input * p * 4;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(m[0]));
                        float32x4_t _k = bfloat2float(vld1_u16(kptr));
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 1;
                        kptr += 4;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_u16(outptr, float2bfloat(_sum));
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_tm + num_input * p;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vdupq_n_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = bfloat2float(vld1_u16(m));
                        float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(kptr[0]));
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_u16(outptr, float2bfloat(_sum));
                    outptr += 4;
                }
            }
#endif // __ARM_NEON

            if (elempack == 1 && num_output_elempack == 1)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_tm + num_input * p;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += bfloat16_to_float32(*m) * bfloat16_to_float32(*kptr);

                        m += 1;
                        kptr += 1;
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = float32_to_bfloat16(sum);
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum0 = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _val = bfloat2float(vld1_u16(sptr));

                float32x4_t _w0 = bfloat2float(vld1_u16(kptr));
                float32x4_t _w1 = bfloat2float(vld1_u16(kptr + 4));
                float32x4_t _w2 = bfloat2float(vld1_u16(kptr + 8));
                float32x4_t _w3 = bfloat2float(vld1_u16(kptr + 12));

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w1, _val, 1);
                _sum2 = vmlaq_laneq_f32(_sum2, _w2, _val, 2);
                _sum3 = vmlaq_laneq_f32(_sum3, _w3, _val, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_val), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_val), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_val), 1);
#endif

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[0]));

                float32x4_t _w = bfloat2float(vld1_u16(kptr));

                _sum0 = vmlaq_f32(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _sum0 = vaddq_f32(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            vst1_u16(outptr + p * 4, float2bfloat(_sum0));
        }
    }
#endif // __ARM_NEON

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const unsigned short* kptr = weight_data_tm.row<unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            int i = 0;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _m = bfloat2float(vld1_u16(sptr));
                float32x4_t _w = bfloat2float(vld1_u16(kptr));

                _sum = vmlaq_f32(_sum, _m, _w);

                sptr += 4;
                kptr += 4;
            }
#endif // __ARM_NEON
            for (; i < num_input; i++)
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
#endif // NCNN_BF16

#if NCNN_INT8
int InnerProduct_arm::create_pipeline_int8_arm(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g0 = weight_data_tm.row<signed char>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<signed char>(q + j)[p];
                }
            }
        }
    }

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // dequantize
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_in_data[p] = scale_in;
    }

    weight_data.release();

    return 0;
}

int InnerProduct_arm::forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    if (bottom_blob_int8.dims == 2 && bottom_blob_int8.w == num_input)
    {
        // gemm
        Mat bottom_blob_int8_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_int8, bottom_blob_int8_unpacked, 1, opt_unpack);

        int h = bottom_blob_int8_unpacked.h;

        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = h % 4 == 0 ? 4 : 1;
        }
#endif

        int outh = h / out_elempack;

        top_blob.create(num_output, outh, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif

#if __ARM_NEON
        if (num_output_elempack == 8 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    int32x4_t _sum00 = vdupq_n_s32(0);
                    int32x4_t _sum01 = vdupq_n_s32(0);
                    int32x4_t _sum10 = vdupq_n_s32(0);
                    int32x4_t _sum11 = vdupq_n_s32(0);
                    int32x4_t _sum20 = vdupq_n_s32(0);
                    int32x4_t _sum21 = vdupq_n_s32(0);
                    int32x4_t _sum30 = vdupq_n_s32(0);
                    int32x4_t _sum31 = vdupq_n_s32(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        int8x8_t _val0 = vld1_dup_s8(m0);
                        int8x8_t _val1 = vld1_dup_s8(m1);
                        int8x8_t _val2 = vld1_dup_s8(m2);
                        int8x8_t _val3 = vld1_dup_s8(m3);

                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_val0, _w);
                        int16x8_t _s1 = vmull_s8(_val1, _w);
                        int16x8_t _s2 = vmull_s8(_val2, _w);
                        int16x8_t _s3 = vmull_s8(_val3, _w);
                        _sum00 = vaddw_s16(_sum00, vget_low_s16(_s0));
                        _sum01 = vaddw_s16(_sum01, vget_high_s16(_s0));
                        _sum10 = vaddw_s16(_sum10, vget_low_s16(_s1));
                        _sum11 = vaddw_s16(_sum11, vget_high_s16(_s1));
                        _sum20 = vaddw_s16(_sum20, vget_low_s16(_s2));
                        _sum21 = vaddw_s16(_sum21, vget_high_s16(_s2));
                        _sum30 = vaddw_s16(_sum30, vget_low_s16(_s3));
                        _sum31 = vaddw_s16(_sum31, vget_high_s16(_s3));

                        m0++;
                        m1++;
                        m2++;
                        m3++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
                    float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);

                    float32x4_t _sumfp32_00 = vcvtq_f32_s32(_sum00);
                    float32x4_t _sumfp32_01 = vcvtq_f32_s32(_sum01);
                    float32x4_t _sumfp32_10 = vcvtq_f32_s32(_sum10);
                    float32x4_t _sumfp32_11 = vcvtq_f32_s32(_sum11);
                    float32x4_t _sumfp32_20 = vcvtq_f32_s32(_sum20);
                    float32x4_t _sumfp32_21 = vcvtq_f32_s32(_sum21);
                    float32x4_t _sumfp32_30 = vcvtq_f32_s32(_sum30);
                    float32x4_t _sumfp32_31 = vcvtq_f32_s32(_sum31);
                    if (bias_term)
                    {
                        float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                        float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                        _sumfp32_00 = vmlaq_f32(_bias0, _sumfp32_00, _scale_in0);
                        _sumfp32_01 = vmlaq_f32(_bias1, _sumfp32_01, _scale_in1);
                        _sumfp32_10 = vmlaq_f32(_bias0, _sumfp32_10, _scale_in0);
                        _sumfp32_11 = vmlaq_f32(_bias1, _sumfp32_11, _scale_in1);
                        _sumfp32_20 = vmlaq_f32(_bias0, _sumfp32_20, _scale_in0);
                        _sumfp32_21 = vmlaq_f32(_bias1, _sumfp32_21, _scale_in1);
                        _sumfp32_30 = vmlaq_f32(_bias0, _sumfp32_30, _scale_in0);
                        _sumfp32_31 = vmlaq_f32(_bias1, _sumfp32_31, _scale_in1);
                    }
                    else
                    {
                        _sumfp32_00 = vmulq_f32(_sumfp32_00, _scale_in0);
                        _sumfp32_01 = vmulq_f32(_sumfp32_01, _scale_in1);
                        _sumfp32_10 = vmulq_f32(_sumfp32_10, _scale_in0);
                        _sumfp32_11 = vmulq_f32(_sumfp32_11, _scale_in1);
                        _sumfp32_20 = vmulq_f32(_sumfp32_20, _scale_in0);
                        _sumfp32_21 = vmulq_f32(_sumfp32_21, _scale_in1);
                        _sumfp32_30 = vmulq_f32(_sumfp32_30, _scale_in0);
                        _sumfp32_31 = vmulq_f32(_sumfp32_31, _scale_in1);
                    }

                    _sumfp32_00 = activation_ps(_sumfp32_00, activation_type, activation_params);
                    _sumfp32_01 = activation_ps(_sumfp32_01, activation_type, activation_params);
                    _sumfp32_10 = activation_ps(_sumfp32_10, activation_type, activation_params);
                    _sumfp32_11 = activation_ps(_sumfp32_11, activation_type, activation_params);
                    _sumfp32_20 = activation_ps(_sumfp32_20, activation_type, activation_params);
                    _sumfp32_21 = activation_ps(_sumfp32_21, activation_type, activation_params);
                    _sumfp32_30 = activation_ps(_sumfp32_30, activation_type, activation_params);
                    _sumfp32_31 = activation_ps(_sumfp32_31, activation_type, activation_params);

                    // transpose 4x8
                    float32x4x4_t _sumfp32_0;
                    _sumfp32_0.val[0] = _sumfp32_00;
                    _sumfp32_0.val[1] = _sumfp32_10;
                    _sumfp32_0.val[2] = _sumfp32_20;
                    _sumfp32_0.val[3] = _sumfp32_30;
                    float32x4x4_t _sumfp32_1;
                    _sumfp32_1.val[0] = _sumfp32_01;
                    _sumfp32_1.val[1] = _sumfp32_11;
                    _sumfp32_1.val[2] = _sumfp32_21;
                    _sumfp32_1.val[3] = _sumfp32_31;

                    vst4q_f32(outptr, _sumfp32_0);
                    vst4q_f32(outptr + 16, _sumfp32_1);

                    outptr += 32;
                }
            }
        }

        if (num_output_elempack == 1 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    int i = 0;

                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    int32x4_t _sum2 = vdupq_n_s32(0);
                    int32x4_t _sum3 = vdupq_n_s32(0);
                    for (; i + 7 < num_input; i += 8)
                    {
                        int8x8_t _val0 = vld1_s8(m0);
                        int8x8_t _val1 = vld1_s8(m1);
                        int8x8_t _val2 = vld1_s8(m2);
                        int8x8_t _val3 = vld1_s8(m3);
                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_val0, _w);
                        int16x8_t _s1 = vmull_s8(_val1, _w);
                        int16x8_t _s2 = vmull_s8(_val2, _w);
                        int16x8_t _s3 = vmull_s8(_val3, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                        _sum2 = vaddw_s16(_sum2, vget_low_s16(_s2));
                        _sum3 = vaddw_s16(_sum3, vget_low_s16(_s3));
                        _sum0 = vaddw_s16(_sum0, vget_high_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s1));
                        _sum2 = vaddw_s16(_sum2, vget_high_s16(_s2));
                        _sum3 = vaddw_s16(_sum3, vget_high_s16(_s3));

                        m0 += 8;
                        m1 += 8;
                        m2 += 8;
                        m3 += 8;
                        kptr += 8;
                    }
#if __aarch64__
                    sum0 = vaddvq_s32(_sum0);
                    sum1 = vaddvq_s32(_sum1);
                    sum2 = vaddvq_s32(_sum2);
                    sum3 = vaddvq_s32(_sum3);
#else
                    int32x2_t _s20 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                    int32x2_t _s21 = vadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                    int32x2_t _s22 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                    int32x2_t _s23 = vadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));
                    int32x2_t _s201 = vpadd_s32(_s20, _s21);
                    int32x2_t _s223 = vpadd_s32(_s22, _s23);
                    sum0 = vget_lane_s32(_s201, 0);
                    sum1 = vget_lane_s32(_s201, 1);
                    sum2 = vget_lane_s32(_s223, 0);
                    sum3 = vget_lane_s32(_s223, 1);
#endif
                    for (; i < num_input; i++)
                    {
                        sum0 += *m0++ * kptr[0];
                        sum1 += *m1++ * kptr[0];
                        sum2 += *m2++ * kptr[0];
                        sum3 += *m3++ * kptr[0];
                        kptr += 1;
                    }

                    // dequantize and relu
                    float sumfp32_0 = sum0 * scale_in_data[p];
                    float sumfp32_1 = sum1 * scale_in_data[p];
                    float sumfp32_2 = sum2 * scale_in_data[p];
                    float sumfp32_3 = sum3 * scale_in_data[p];

                    if (bias_term)
                    {
                        sumfp32_0 += bias_data[p];
                        sumfp32_1 += bias_data[p];
                        sumfp32_2 += bias_data[p];
                        sumfp32_3 += bias_data[p];
                    }

                    outptr[0] = activation_ss(sumfp32_0, activation_type, activation_params);
                    outptr[1] = activation_ss(sumfp32_1, activation_type, activation_params);
                    outptr[2] = activation_ss(sumfp32_2, activation_type, activation_params);
                    outptr[3] = activation_ss(sumfp32_3, activation_type, activation_params);
                    outptr += 4;
                }
            }
        }

        if (num_output_elempack == 8 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        int8x8_t _val0 = vdup_n_s8(m[0]);
                        int8x8_t _val1 = vdup_n_s8(m[1]);
                        int8x8_t _val2 = vdup_n_s8(m[2]);
                        int8x8_t _val3 = vdup_n_s8(m[3]);

                        int8x16_t _w0 = vld1q_s8(kptr);
                        int8x16_t _w1 = vld1q_s8(kptr + 16);

                        int16x8_t _s0 = vmull_s8(_val0, vget_low_s8(_w0));
                        int16x8_t _s1 = vmull_s8(_val2, vget_low_s8(_w1));
                        _s0 = vmlal_s8(_s0, _val1, vget_high_s8(_w0));
                        _s1 = vmlal_s8(_s1, _val3, vget_high_s8(_w1));

                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s1));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s1));

                        m += 4;
                        kptr += 32;
                    }
                    for (; i < num_input; i++)
                    {
                        int8x8_t _val = vld1_dup_s8(m);
                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        m++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
                    float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);

                    float32x4_t _sumfp32_0 = vcvtq_f32_s32(_sum0);
                    float32x4_t _sumfp32_1 = vcvtq_f32_s32(_sum1);

                    if (bias_term)
                    {
                        float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                        float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                        _sumfp32_0 = vmlaq_f32(_bias0, _sumfp32_0, _scale_in0);
                        _sumfp32_1 = vmlaq_f32(_bias1, _sumfp32_1, _scale_in1);
                    }
                    else
                    {
                        _sumfp32_0 = vmulq_f32(_sumfp32_0, _scale_in0);
                        _sumfp32_1 = vmulq_f32(_sumfp32_1, _scale_in1);
                    }

                    _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
                    _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

                    vst1q_f32(outptr, _sumfp32_0);
                    vst1q_f32(outptr + 4, _sumfp32_1);
                    outptr += 8;
                }
            }
        }
#endif // __ARM_NEON

        if (num_output_elempack == 1 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    int sum = 0;

                    int i = 0;
#if __ARM_NEON
                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    for (; i + 7 < num_input; i += 8)
                    {
                        int8x8_t _val = vld1_s8(m);
                        int8x8_t _w = vld1_s8(kptr);

                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        m += 8;
                        kptr += 8;
                    }

                    _sum0 = vaddq_s32(_sum0, _sum1);
#if __aarch64__
                    sum = vaddvq_s32(_sum0);
#else
                    int32x2_t _s2 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                    _s2 = vpadd_s32(_s2, _s2);
                    sum = vget_lane_s32(_s2, 0);
#endif
#endif // __ARM_NEON
                    for (; i < num_input; i++)
                    {
                        sum += *m++ * *kptr++;
                    }

                    // dequantize and relu
                    float sumfp32 = sum * scale_in_data[p];

                    if (bias_term)
                        sumfp32 += bias_data[p];

                    outptr[0] = activation_ss(sumfp32, activation_type, activation_params);
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    Mat bottom_blob_int8_flattened = bottom_blob_int8;
    if (bottom_blob_int8.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;
        flatten->forward(bottom_blob_int8, bottom_blob_int8_flattened, opt_flatten);
    }

    //     int elempack = bottom_blob_int8_flattened.elempack;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);

            int i = 0;
            for (; i + 1 < num_input; i += 2)
            {
                int8x8_t _val0 = vdup_n_s8(sptr[0]);
                int8x8_t _val1 = vdup_n_s8(sptr[1]);

                int8x8_t _w0 = vld1_s8(kptr);
                int8x8_t _w1 = vld1_s8(kptr + 8);

                int16x8_t _s0 = vmull_s8(_val0, _w0);
                _s0 = vmlal_s8(_s0, _val1, _w1);

                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                sptr += 2;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                int8x8_t _val = vdup_n_s8(sptr[0]);

                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _s0 = vmull_s8(_val, _w);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                sptr += 1;
                kptr += 8;
            }

            // dequantize and relu
            float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
            float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);

            float32x4_t _sumfp32_0 = vcvtq_f32_s32(_sum0);
            float32x4_t _sumfp32_1 = vcvtq_f32_s32(_sum1);

            if (bias_term)
            {
                float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                _sumfp32_0 = vmlaq_f32(_bias0, _sumfp32_0, _scale_in0);
                _sumfp32_1 = vmlaq_f32(_bias1, _sumfp32_1, _scale_in1);
            }
            else
            {
                _sumfp32_0 = vmulq_f32(_sumfp32_0, _scale_in0);
                _sumfp32_1 = vmulq_f32(_sumfp32_1, _scale_in1);
            }

            _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
            _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

            float* outptr = (float*)top_blob + p * 8;
            vst1q_f32(outptr, _sumfp32_0);
            vst1q_f32(outptr + 4, _sumfp32_1);
        }
    }
#endif // __ARM_NEON

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int sum = 0;

            int i = 0;
            for (; i < num_input; i++)
            {
                signed char val = sptr[0];

                signed char w = kptr[0];

                sum += val * w;

                sptr += 1;
                kptr += 1;
            }

            // dequantize and relu
            float sumfp32 = sum * scale_in_data[p];

            if (bias_term)
                sumfp32 += bias_data[p];

            sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

            top_blob[p] = sumfp32;
        }
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
