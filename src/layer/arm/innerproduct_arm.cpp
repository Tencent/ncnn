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

namespace ncnn {

InnerProduct_arm::InnerProduct_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    flatten = 0;
    activation = 0;
}

int InnerProduct_arm::create_pipeline(const Option& opt)
{
#if __ARM_NEON
    if (opt.use_packing_layout || opt.use_int8_inference)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }
#endif // __ARM_NEON

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_arm(opt);
    }
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
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

    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int InnerProduct_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_arm(bottom_blob, top_blob, opt);
    }
#endif

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

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data + num_input * p;
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

            if (elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += m[i] * kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

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

    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_fp16.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g0 = weight_data_fp16.row<__fp16>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = (__fp16)(weight_data_r2.row(q + j)[p]);
                }
            }
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int InnerProduct_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == 4 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 4;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

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
                        float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                        float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
                        _sum0 = vfmaq_laneq_f32(_sum0, _val, _k, 0);
                        _sum1 = vfmaq_laneq_f32(_sum1, _val, _k, 1);
                        _sum2 = vfmaq_laneq_f32(_sum2, _val, _k, 2);
                        _sum3 = vfmaq_laneq_f32(_sum3, _val, _k, 3);

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    vst1_f16(outptr, vcvt_f16_f32(_sum0));
                    vst1_f16(outptr + 4, vcvt_f16_f32(_sum1));
                    vst1_f16(outptr + 8, vcvt_f16_f32(_sum2));
                    vst1_f16(outptr + 12, vcvt_f16_f32(_sum3));
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 4;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vdupq_n_f32((float)m[0]);
                        float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
                        _sum = vfmaq_f32(_sum, _val, _k);

                        m += 1;
                        kptr += 4;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, vcvt_f16_f32(_sum));
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vdupq_n_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                        float32x4_t _k = vdupq_n_f32((float)kptr[0]);
                        _sum = vfmaq_f32(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, vcvt_f16_f32(_sum));
                    outptr += 4;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)m[i] * (float)kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
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

    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == 4)
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

            int i = 0;
            for (; i + 3 < num_input; i += 4)
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
            for (; i < num_input; i++)
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

    if (out_elempack == 1)
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
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _m = vcvt_f32_f16(vld1_f16(sptr));
                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));

                _sum = vfmaq_f32(_sum, _m, _w);

                sptr += 4;
                kptr += 4;
            }
            for (; i < num_input; i++)
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
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == 8 && num_output_elempack == 8)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 8;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x8_t _sum0 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum1 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum2 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum3 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum4 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum5 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum6 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum7 = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum0 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 0]);
                        _sum1 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 1]);
                        _sum2 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 2]);
                        _sum3 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 3]);
                        _sum4 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 4]);
                        _sum5 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 5]);
                        _sum6 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 6]);
                        _sum7 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 7]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x8_t _val = vld1q_f16(m);
                        float16x8_t _k = vld1q_f16(kptr);
                        _sum0 = vfmaq_laneq_f16(_sum0, _val, _k, 0);
                        _sum1 = vfmaq_laneq_f16(_sum1, _val, _k, 1);
                        _sum2 = vfmaq_laneq_f16(_sum2, _val, _k, 2);
                        _sum3 = vfmaq_laneq_f16(_sum3, _val, _k, 3);
                        _sum4 = vfmaq_laneq_f16(_sum4, _val, _k, 4);
                        _sum5 = vfmaq_laneq_f16(_sum5, _val, _k, 5);
                        _sum6 = vfmaq_laneq_f16(_sum6, _val, _k, 6);
                        _sum7 = vfmaq_laneq_f16(_sum7, _val, _k, 7);

                        m += 8;
                        kptr += 8;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);
                    _sum4 = activation_ps(_sum4, activation_type, activation_params);
                    _sum5 = activation_ps(_sum5, activation_type, activation_params);
                    _sum6 = activation_ps(_sum6, activation_type, activation_params);
                    _sum7 = activation_ps(_sum7, activation_type, activation_params);

                    vst1q_f16(outptr, _sum0);
                    vst1q_f16(outptr + 8, _sum1);
                    vst1q_f16(outptr + 16, _sum2);
                    vst1q_f16(outptr + 24, _sum3);
                    vst1q_f16(outptr + 32, _sum4);
                    vst1q_f16(outptr + 40, _sum5);
                    vst1q_f16(outptr + 48, _sum6);
                    vst1q_f16(outptr + 56, _sum7);
                    outptr += 64;
                }
            }

            if (elempack == 1 && num_output_elempack == 8)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 8;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x8_t _sum = vdupq_n_f16(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x8_t _val = vdupq_n_f16(m[0]);
                        float16x8_t _k = vld1q_f16(kptr);
                        _sum = vfmaq_f16(_sum, _val, _k);

                        m += 1;
                        kptr += 8;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }

            if (elempack == 4 && num_output_elempack == 8)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 8;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x4_t _sum0 = vdup_n_f16(0.f);
                    float16x4_t _sum1 = vdup_n_f16(0.f);
                    float16x4_t _sum2 = vdup_n_f16(0.f);
                    float16x4_t _sum3 = vdup_n_f16(0.f);
                    float16x4_t _sum4 = vdup_n_f16(0.f);
                    float16x4_t _sum5 = vdup_n_f16(0.f);
                    float16x4_t _sum6 = vdup_n_f16(0.f);
                    float16x4_t _sum7 = vdup_n_f16(0.f);

                    if (bias_term)
                    {
                        _sum0 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 0]);
                        _sum1 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 1]);
                        _sum2 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 2]);
                        _sum3 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 3]);
                        _sum4 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 4]);
                        _sum5 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 5]);
                        _sum6 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 6]);
                        _sum7 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 8 + 7]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x4_t _val = vld1_f16(m);
                        float16x8_t _k = vld1q_f16(kptr);
                        _sum0 = vfma_laneq_f16(_sum0, _val, _k, 0);
                        _sum1 = vfma_laneq_f16(_sum1, _val, _k, 1);
                        _sum2 = vfma_laneq_f16(_sum2, _val, _k, 2);
                        _sum3 = vfma_laneq_f16(_sum3, _val, _k, 3);
                        _sum4 = vfma_laneq_f16(_sum4, _val, _k, 4);
                        _sum5 = vfma_laneq_f16(_sum5, _val, _k, 5);
                        _sum6 = vfma_laneq_f16(_sum6, _val, _k, 6);
                        _sum7 = vfma_laneq_f16(_sum7, _val, _k, 7);

                        m += 4;
                        kptr += 8;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);
                    _sum4 = activation_ps(_sum4, activation_type, activation_params);
                    _sum5 = activation_ps(_sum5, activation_type, activation_params);
                    _sum6 = activation_ps(_sum6, activation_type, activation_params);
                    _sum7 = activation_ps(_sum7, activation_type, activation_params);

                    vst1_f16(outptr, _sum0);
                    vst1_f16(outptr + 4, _sum1);
                    vst1_f16(outptr + 8, _sum2);
                    vst1_f16(outptr + 12, _sum3);
                    vst1_f16(outptr + 16, _sum4);
                    vst1_f16(outptr + 20, _sum5);
                    vst1_f16(outptr + 24, _sum6);
                    vst1_f16(outptr + 28, _sum7);
                    outptr += 32;
                }
            }

            if (elempack == 8 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x8_t _val = vld1q_f16(m);
                        float16x8_t _k = vdupq_n_f16(kptr[0]);
                        _sum = vfmaq_f16(_sum, _val, _k);

                        m += 8;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }

            if (elempack == 8 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 4;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x8_t _sum0 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum1 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum2 = vdupq_n_f16((__fp16)0.f);
                    float16x8_t _sum3 = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum0 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 0]);
                        _sum1 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 1]);
                        _sum2 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 2]);
                        _sum3 = vdupq_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 3]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x8_t _val = vld1q_f16(m);
                        float16x4_t _k = vld1_f16(kptr);
                        _sum0 = vfmaq_lane_f16(_sum0, _val, _k, 0);
                        _sum1 = vfmaq_lane_f16(_sum1, _val, _k, 1);
                        _sum2 = vfmaq_lane_f16(_sum2, _val, _k, 2);
                        _sum3 = vfmaq_lane_f16(_sum3, _val, _k, 3);

                        m += 8;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    vst1q_f16(outptr, _sum0);
                    vst1q_f16(outptr + 8, _sum1);
                    vst1q_f16(outptr + 16, _sum2);
                    vst1q_f16(outptr + 24, _sum3);
                    outptr += 32;
                }
            }

            if (elempack == 4 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 4;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x4_t _sum0 = vdup_n_f16(0.f);
                    float16x4_t _sum1 = vdup_n_f16(0.f);
                    float16x4_t _sum2 = vdup_n_f16(0.f);
                    float16x4_t _sum3 = vdup_n_f16(0.f);

                    if (bias_term)
                    {
                        _sum0 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 0]);
                        _sum1 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 1]);
                        _sum2 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 2]);
                        _sum3 = vdup_n_f16(((const __fp16*)bias_data_fp16)[p * 4 + 3]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x4_t _val = vld1_f16(m);
                        float16x4_t _k = vld1_f16(kptr);
                        _sum0 = vfma_lane_f16(_sum0, _val, _k, 0);
                        _sum1 = vfma_lane_f16(_sum1, _val, _k, 1);
                        _sum2 = vfma_lane_f16(_sum2, _val, _k, 2);
                        _sum3 = vfma_lane_f16(_sum3, _val, _k, 3);

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    vst1_f16(outptr, _sum0);
                    vst1_f16(outptr + 4, _sum1);
                    vst1_f16(outptr + 8, _sum2);
                    vst1_f16(outptr + 12, _sum3);
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * 4;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x4_t _sum = vdup_n_f16(0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x4_t _val = vdup_n_f16(m[0]);
                        float16x4_t _k = vld1_f16(kptr);
                        _sum = vfma_f16(_sum, _val, _k);

                        m += 1;
                        kptr += 4;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float16x4_t _sum = vdup_n_f16(0.f);

                    if (bias_term)
                    {
                        _sum = vdup_n_f16(((const __fp16*)bias_data_fp16)[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float16x4_t _val = vld1_f16(m);
                        float16x4_t _k = vdup_n_f16(kptr[0]);
                        _sum = vfma_f16(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)(m[i] * kptr[i]);
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
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
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x8_t _sum0 = vdupq_n_f16(0.f);
            float16x8_t _sum1 = vdupq_n_f16(0.f);
            float16x8_t _sum2 = vdupq_n_f16(0.f);
            float16x8_t _sum3 = vdupq_n_f16(0.f);
            float16x8_t _sum4 = vdupq_n_f16(0.f);
            float16x8_t _sum5 = vdupq_n_f16(0.f);
            float16x8_t _sum6 = vdupq_n_f16(0.f);
            float16x8_t _sum7 = vdupq_n_f16(0.f);

            if (bias_term)
            {
                _sum0 = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                asm volatile(
                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v0.8h}, [%8], #16          \n" // _val

                    "prfm   pldl1keep, [%9, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%9], #64 \n" // w0123

                    "prfm   pldl1keep, [%9, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%9], #64 \n" // w4567

                    "fmla   %0.8h, v8.8h, v0.h[0]       \n"
                    "fmla   %1.8h, v9.8h, v0.h[1]       \n"
                    "fmla   %2.8h, v10.8h, v0.h[2]      \n"
                    "fmla   %3.8h, v11.8h, v0.h[3]      \n"
                    "fmla   %4.8h, v12.8h, v0.h[4]      \n"
                    "fmla   %5.8h, v13.8h, v0.h[5]      \n"
                    "fmla   %6.8h, v14.8h, v0.h[6]      \n"
                    "fmla   %7.8h, v15.8h, v0.h[7]      \n"

                    : "=w"(_sum0), // %0
                    "=w"(_sum1), // %1
                    "=w"(_sum2), // %2
                    "=w"(_sum3), // %3
                    "=w"(_sum4), // %4
                    "=w"(_sum5), // %5
                    "=w"(_sum6), // %6
                    "=w"(_sum7), // %7
                    "=r"(sptr),  // %8
                    "=r"(kptr)   // %9
                    : "0"(_sum0),
                    "1"(_sum1),
                    "2"(_sum2),
                    "3"(_sum3),
                    "4"(_sum4),
                    "5"(_sum5),
                    "6"(_sum6),
                    "7"(_sum7),
                    "8"(sptr),
                    "9"(kptr)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
            for (; i + 3 < num_input; i += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v0.4h}, [%4], #8           \n" // _val

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%5], #64 \n" // w0123

                    "fmla   %0.8h, v8.8h, v0.h[0]       \n"
                    "fmla   %1.8h, v9.8h, v0.h[1]       \n"
                    "fmla   %2.8h, v10.8h, v0.h[2]      \n"
                    "fmla   %3.8h, v11.8h, v0.h[3]      \n"

                    : "=w"(_sum0), // %0
                    "=w"(_sum1), // %1
                    "=w"(_sum2), // %2
                    "=w"(_sum3), // %3
                    "=r"(sptr),  // %4
                    "=r"(kptr)   // %5
                    : "0"(_sum0),
                    "1"(_sum1),
                    "2"(_sum2),
                    "3"(_sum3),
                    "4"(sptr),
                    "5"(kptr)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11");
            }
            for (; i < num_input; i++)
            {
                float16x8_t _val = vdupq_n_f16(sptr[0]);

                float16x8_t _w = vld1q_f16(kptr);

                _sum0 = vfmaq_f16(_sum0, _val, _w);

                sptr += 1;
                kptr += 8;
            }

            _sum0 = vaddq_f16(_sum0, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _sum4 = vaddq_f16(_sum4, _sum5);
            _sum6 = vaddq_f16(_sum6, _sum7);
            _sum0 = vaddq_f16(_sum0, _sum2);
            _sum4 = vaddq_f16(_sum4, _sum6);
            _sum0 = vaddq_f16(_sum0, _sum4);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1q_f16(outptr + p * 8, _sum0);
        }
    }

    if (out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float16x4_t _sum0 = vdup_n_f16(0.f);
            float16x4_t _sum1 = vdup_n_f16(0.f);
            float16x4_t _sum2 = vdup_n_f16(0.f);
            float16x4_t _sum3 = vdup_n_f16(0.f);
            float16x4_t _sum4 = vdup_n_f16(0.f);
            float16x4_t _sum5 = vdup_n_f16(0.f);
            float16x4_t _sum6 = vdup_n_f16(0.f);
            float16x4_t _sum7 = vdup_n_f16(0.f);

            if (bias_term)
            {
                _sum0 = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                asm volatile(
                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v0.8h}, [%8], #16          \n" // _val

                    "prfm   pldl1keep, [%9, #256]       \n"
                    "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%9], #32 \n" // w0123

                    "prfm   pldl1keep, [%9, #256]       \n"
                    "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%9], #32 \n" // w4567

                    "fmla   %0.4h, v8.4h, v0.h[0]       \n"
                    "fmla   %1.4h, v9.4h, v0.h[1]       \n"
                    "fmla   %2.4h, v10.4h, v0.h[2]      \n"
                    "fmla   %3.4h, v11.4h, v0.h[3]      \n"
                    "fmla   %4.4h, v12.4h, v0.h[4]      \n"
                    "fmla   %5.4h, v13.4h, v0.h[5]      \n"
                    "fmla   %6.4h, v14.4h, v0.h[6]      \n"
                    "fmla   %7.4h, v15.4h, v0.h[7]      \n"

                    : "=w"(_sum0), // %0
                    "=w"(_sum1), // %1
                    "=w"(_sum2), // %2
                    "=w"(_sum3), // %3
                    "=w"(_sum4), // %4
                    "=w"(_sum5), // %5
                    "=w"(_sum6), // %6
                    "=w"(_sum7), // %7
                    "=r"(sptr),  // %8
                    "=r"(kptr)   // %9
                    : "0"(_sum0),
                    "1"(_sum1),
                    "2"(_sum2),
                    "3"(_sum3),
                    "4"(_sum4),
                    "5"(_sum5),
                    "6"(_sum6),
                    "7"(_sum7),
                    "8"(sptr),
                    "9"(kptr)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
            for (; i + 3 < num_input; i += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v0.4h}, [%4], #8           \n" // _val

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%5], #32 \n" // w0123

                    "fmla   %0.4h, v8.4h, v0.h[0]       \n"
                    "fmla   %1.4h, v9.4h, v0.h[1]       \n"
                    "fmla   %2.4h, v10.4h, v0.h[2]      \n"
                    "fmla   %3.4h, v11.4h, v0.h[3]      \n"

                    : "=w"(_sum0), // %0
                    "=w"(_sum1), // %1
                    "=w"(_sum2), // %2
                    "=w"(_sum3), // %3
                    "=r"(sptr),  // %4
                    "=r"(kptr)   // %5
                    : "0"(_sum0),
                    "1"(_sum1),
                    "2"(_sum2),
                    "3"(_sum3),
                    "4"(sptr),
                    "5"(kptr)
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11");
            }
            for (; i < num_input; i++)
            {
                float16x4_t _val = vdup_n_f16(sptr[0]);

                float16x4_t _w = vld1_f16(kptr);

                _sum0 = vfma_f16(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = vadd_f16(_sum0, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _sum4 = vadd_f16(_sum4, _sum5);
            _sum6 = vadd_f16(_sum6, _sum7);
            _sum0 = vadd_f16(_sum0, _sum2);
            _sum4 = vadd_f16(_sum4, _sum6);
            _sum0 = vadd_f16(_sum0, _sum4);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            vst1_f16(outptr + p * 4, _sum0);
        }
    }

    if (out_elempack == 1)
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
            for (; i + 7 < num_input; i += 8)
            {
                float16x8_t _m = vld1q_f16(sptr);
                float16x8_t _w = vld1q_f16(kptr);

                _sum = vfmaq_f16(_sum, _m, _w);

                sptr += 8;
                kptr += 8;
            }
            for (; i < num_input; i++)
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

#if NCNN_BF16
int InnerProduct_arm::create_pipeline_bf16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_bf16.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g0 = weight_data_bf16.row<unsigned short>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = float32_to_bfloat16(weight_data_r2.row(q + j)[p]);
                }
            }
        }
    }

    return 0;
}

int InnerProduct_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
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
                    const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + num_input * p * 4;
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
                        float32x4_t _val = vcvt_f32_bf16(vld1_u16(m));
                        float32x4_t _k = vcvt_f32_bf16(vld1_u16(kptr));
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

                    vst1_u16(outptr, vcvt_bf16_f32(_sum0));
                    vst1_u16(outptr + 4, vcvt_bf16_f32(_sum1));
                    vst1_u16(outptr + 8, vcvt_bf16_f32(_sum2));
                    vst1_u16(outptr + 12, vcvt_bf16_f32(_sum3));
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + num_input * p * 4;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(m[0]));
                        float32x4_t _k = vcvt_f32_bf16(vld1_u16(kptr));
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 1;
                        kptr += 4;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_u16(outptr, vcvt_bf16_f32(_sum));
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + num_input * p;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vdupq_n_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        float32x4_t _val = vcvt_f32_bf16(vld1_u16(m));
                        float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(kptr[0]));
                        _sum = vmlaq_f32(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_u16(outptr, vcvt_bf16_f32(_sum));
                    outptr += 4;
                }
            }
#endif // __ARM_NEON

            if (elempack == 1 && num_output_elempack == 1)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + num_input * p;
                    const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += bfloat16_to_float32(m[i]) * bfloat16_to_float32(kptr[i]);
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

    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
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

            const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr));
                float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr + 4));
                float32x4_t _w2 = vcvt_f32_bf16(vld1_u16(kptr + 8));
                float32x4_t _w3 = vcvt_f32_bf16(vld1_u16(kptr + 12));

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

                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));

                _sum0 = vmlaq_f32(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = vaddq_f32(_sum0, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _sum0 = vaddq_f32(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            vst1_u16(outptr + p * 4, vcvt_bf16_f32(_sum0));
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

            const unsigned short* kptr = weight_data_bf16.row<unsigned short>(p);

            const unsigned short* sptr = bottom_blob_flattened;

            int i = 0;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
            for (; i + 3 < num_input; i += 4)
            {
                float32x4_t _m = vcvt_f32_bf16(vld1_u16(sptr));
                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));

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
    if (activation_type == 1)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 2)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]); // slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]); // min
        pd.set(1, activation_params[1]); // max
        activation->load_param(pd);
    }
    else if (activation_type == 4)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Sigmoid);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 5)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Mish);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }

    if (activation)
    {
        activation->create_pipeline(opt);
    }

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_int8.create(num_input, num_output / out_elempack, (size_t)out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g0 = weight_data_int8.row<signed char>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<signed char>(q + j)[p];
                }
            }
        }
    }

    return 0;
}

int InnerProduct_arm::forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        Mat bottom_blob_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_unpack);

        return forward_int8(bottom_blob_unpacked, top_blob, opt);
    }

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
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
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat top_blob_int32;
    top_blob_int32.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

#if __ARM_NEON
    if (out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_int8.row<const signed char>(p);
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

            int* outptr = (int*)top_blob_int32;
            vst1q_s32(outptr + p * 8, _sum0);
            vst1q_s32(outptr + p * 8 + 4, _sum1);
        }
    }
#endif // __ARM_NEON

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_int8.row<const signed char>(p);
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

            int* outptr = (int*)top_blob_int32;
            outptr[p] = sum;
        }
    }

    Mat scale_data(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // dequantize
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_data[p] = scale_in;
    }

    dequantize_from_int32(top_blob_int32, top_blob, scale_data, bias_data, opt);

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
