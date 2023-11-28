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

#include "innerproduct_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "innerproduct_fp16s.h"
#include "innerproduct_gemm_fp16s.h"

void innerproduct_pack4_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_pack4_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_transform_kernel_fp16s_neon_asimdhp(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    innerproduct_transform_kernel_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, opt);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int InnerProduct_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 8;
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

                    _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps_f16(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps_f16(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps_f16(_sum3, activation_type, activation_params);
                    _sum4 = activation_ps_f16(_sum4, activation_type, activation_params);
                    _sum5 = activation_ps_f16(_sum5, activation_type, activation_params);
                    _sum6 = activation_ps_f16(_sum6, activation_type, activation_params);
                    _sum7 = activation_ps_f16(_sum7, activation_type, activation_params);

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 8;
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

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }

            if (elempack == 4 && num_output_elempack == 8)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 8;
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

                    _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps_f16(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps_f16(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps_f16(_sum3, activation_type, activation_params);
                    _sum4 = activation_ps_f16(_sum4, activation_type, activation_params);
                    _sum5 = activation_ps_f16(_sum5, activation_type, activation_params);
                    _sum6 = activation_ps_f16(_sum6, activation_type, activation_params);
                    _sum7 = activation_ps_f16(_sum7, activation_type, activation_params);

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
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

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }

            if (elempack == 8 && num_output_elempack == 4)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 4;
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

                    _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps_f16(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps_f16(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps_f16(_sum3, activation_type, activation_params);

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 4;
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

                    _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps_f16(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps_f16(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps_f16(_sum3, activation_type, activation_params);

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * 4;
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

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
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

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)(*m * *kptr);

                        m += 1;
                        kptr += 1;
                    }

                    sum = activation_ss_f16(sum, activation_type, activation_params);

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

            const __fp16* kptr = weight_data_tm.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
#if NCNN_GNU_INLINE_ASM
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
#endif // NCNN_GNU_INLINE_ASM
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

            _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);

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

            const __fp16* kptr = weight_data_tm.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
#if NCNN_GNU_INLINE_ASM
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
#endif // NCNN_GNU_INLINE_ASM
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

            _sum0 = activation_ps_f16(_sum0, activation_type, activation_params);

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

            const __fp16* kptr = weight_data_tm.row<__fp16>(p);

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

            sum = activation_ss_f16(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
