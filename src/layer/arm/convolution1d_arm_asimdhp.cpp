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

#include "convolution1d_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Convolution1D_arm::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / kernel_w / num_output;

    int elempack = 1;
    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        elempack = opt.use_fp16_arithmetic && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }

    // src = kw-inch-outch
    // dst = pb-pa-kw-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(kernel_w, num_input, num_output);

        weight_data_fp16.create(kernel_w, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g00 = weight_data_fp16.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < kernel_w; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = (__fp16)k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int Convolution1D_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int out_elempack = (opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 4 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 4;
                            kptr += 16;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, vcvt_f16_f32(_sum));
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32((const float*)bias_data + p * 4);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vdup_n_f16(sptr[0]));
                            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
                            _sum = vfmaq_f32(_sum, _val, _w);

                            sptr += dilation_w;
                            kptr += 4;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, vcvt_f16_f32(_sum));
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));
                            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
                            float32x4_t _s4 = vmulq_f32(_val, _w);

                            sum += vaddvq_f32(_s4); // dot

                            sptr += dilation_w * 4;
                            kptr += 4;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = (__fp16)sum;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float val = (float)sptr[0];
                            float w = (float)kptr[0];
                            sum += val * w;

                            sptr += dilation_w;
                            kptr += 1;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = (__fp16)sum;
                }
            }
        }
    }

    return 0;
}

int Convolution1D_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 8 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float16x8_t _val = vld1q_f16(sptr);

                            float16x8_t _w0 = vld1q_f16(kptr);
                            float16x8_t _w1 = vld1q_f16(kptr + 8);
                            float16x8_t _w2 = vld1q_f16(kptr + 16);
                            float16x8_t _w3 = vld1q_f16(kptr + 24);
                            float16x8_t _w4 = vld1q_f16(kptr + 32);
                            float16x8_t _w5 = vld1q_f16(kptr + 40);
                            float16x8_t _w6 = vld1q_f16(kptr + 48);
                            float16x8_t _w7 = vld1q_f16(kptr + 56);

                            _sum = vfmaq_laneq_f16(_sum, _w0, _val, 0);
                            _sum = vfmaq_laneq_f16(_sum, _w1, _val, 1);
                            _sum = vfmaq_laneq_f16(_sum, _w2, _val, 2);
                            _sum = vfmaq_laneq_f16(_sum, _w3, _val, 3);
                            _sum = vfmaq_laneq_f16(_sum, _w4, _val, 4);
                            _sum = vfmaq_laneq_f16(_sum, _w5, _val, 5);
                            _sum = vfmaq_laneq_f16(_sum, _w6, _val, 6);
                            _sum = vfmaq_laneq_f16(_sum, _w7, _val, 7);

                            sptr += dilation_w * 8;
                            kptr += 64;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float16x8_t _val = vdupq_n_f16(sptr[0]);
                            float16x8_t _w = vld1q_f16(kptr);
                            _sum = vfmaq_f16(_sum, _val, _w);

                            sptr += dilation_w;
                            kptr += 8;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 4;
                            kptr += 32;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f16(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = ((const __fp16*)bias_data_fp16)[p];
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float16x8_t _val = vld1q_f16(sptr);
                            float16x8_t _w = vld1q_f16(kptr);
                            float16x8_t _s8 = vmulq_f16(_val, _w);

                            float16x4_t _s4 = vadd_f16(vget_low_f16(_s8), vget_high_f16(_s8));
                            sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

                            sptr += dilation_w * 8;
                            kptr += 8;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 8;
                            kptr += 32;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 4;
                            kptr += 16;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float16x4_t _val = vdup_n_f16(sptr[0]);
                            float16x4_t _w = vld1_f16(kptr);
                            _sum = vfma_f16(_sum, _val, _w);

                            sptr += dilation_w;
                            kptr += 4;
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1_f16(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = ((const __fp16*)bias_data_fp16)[p];
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float16x4_t _val = vld1_f16(sptr);
                            float16x4_t _w = vld1_f16(kptr);
                            float16x4_t _s4 = vmul_f16(_val, _w);

                            sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

                            sptr += dilation_w * 4;
                            kptr += 4;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                __fp16* outptr = top_blob.row<__fp16>(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const __fp16* kptr = weight_data_fp16.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const __fp16* sptr = bottom_blob_bordered.row<const __fp16>(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float val = (float)sptr[0];
                            float w = (float)kptr[0];
                            sum += val * w;

                            sptr += dilation_w;
                            kptr += 1;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = (__fp16)sum;
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
