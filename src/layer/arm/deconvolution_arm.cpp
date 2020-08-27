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

#include "deconvolution_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

#include "neon_activation.h"

namespace ncnn {

#include "deconvolution_3x3.h"
#include "deconvolution_4x4.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "deconvolution_4x4_fp16s.h"
#endif

Deconvolution_arm::Deconvolution_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;

    activation = 0;
}

int Deconvolution_arm::create_pipeline(const Option& opt)
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

    if (activation)
    {
        activation->create_pipeline(opt);
    }

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

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    int elempack = (support_packing && opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;

#if __ARM_NEON
    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-4a-kw-kh-inch/4a-outch/4b
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack4.create(maxk, num_input / 4, num_output / 4, (size_t)4 * 16, 16);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);

                Mat g0 = weight_data_pack4.channel(q / 4);

                for (int p = 0; p + 3 < num_input; p += 4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p + 1);
                    const float* k02 = k0.row(p + 2);
                    const float* k03 = k0.row(p + 3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p + 1);
                    const float* k12 = k1.row(p + 2);
                    const float* k13 = k1.row(p + 3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p + 1);
                    const float* k22 = k2.row(p + 2);
                    const float* k23 = k2.row(p + 3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p + 1);
                    const float* k32 = k3.row(p + 2);
                    const float* k33 = k3.row(p + 3);

                    float* g00 = g0.row(p / 4);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00[4] = k01[k];
                        g00[5] = k11[k];
                        g00[6] = k21[k];
                        g00[7] = k31[k];

                        g00[8] = k02[k];
                        g00[9] = k12[k];
                        g00[10] = k22[k];
                        g00[11] = k32[k];

                        g00[12] = k03[k];
                        g00[13] = k13[k];
                        g00[14] = k23[k];
                        g00[15] = k33[k];

                        g00 += 16;
                    }
                }
            }
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack1to4.create(maxk, num_input, num_output / 4, (size_t)4 * 4, 4);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);

                Mat g0 = weight_data_pack1to4.channel(q / 4);

                for (int p = 0; p < num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    float* g00 = g0.row(p);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00 += 4;
                    }
                }
            }
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_pack4to1.create(maxk, num_input / 4, num_output, (size_t)4 * 4, 4);

            for (int q = 0; q < num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1.channel(q);

                for (int p = 0; p + 3 < num_input; p += 4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p + 1);
                    const float* k02 = k0.row(p + 2);
                    const float* k03 = k0.row(p + 3);

                    float* g00 = g0.row(p / 4);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00 += 4;
                    }
                }
            }
        }
    }
#endif // __ARM_NEON

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        weight_data_pack1 = weight_data_transposed;
    }

    return 0;
}

int Deconvolution_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int Deconvolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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

    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                    }

                    const float* kptr = (const float*)weight_data_pack4 + maxk * channels * p * 16;

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 4;

                                float32x4_t _val = vld1q_f32(sptr);

                                int k = y * kernel_w + x;

                                float32x4_t _w0 = vld1q_f32(kptr + k * 16);
                                float32x4_t _w1 = vld1q_f32(kptr + k * 16 + 4);
                                float32x4_t _w2 = vld1q_f32(kptr + k * 16 + 8);
                                float32x4_t _w3 = vld1q_f32(kptr + k * 16 + 12);

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
                            }
                        }

                        kptr += maxk * 16;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f32(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                    }

                    const float* kptr = (const float*)weight_data_pack1to4 + maxk * channels * p * 4;

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const float* sptr = m.row(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float32x4_t _val = vdupq_n_f32(sptr[sx]);

                                int k = y * kernel_w + x;

                                float32x4_t _w = vld1q_f32(kptr + k * 4);

                                _sum = vmlaq_f32(_sum, _val, _w);
                            }
                        }

                        kptr += maxk * 4;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f32(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const float* kptr = (const float*)weight_data_pack4to1 + maxk * channels * p * 4;

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 4;

                                float32x4_t _val = vld1q_f32(sptr);

                                int k = y * kernel_w + x;

                                float32x4_t _w = vld1q_f32(kptr + k * 4);

                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif
                            }
                        }

                        kptr += maxk * 4;
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            deconv3x3s1_neon(bottom_blob, top_blob_bordered, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob_bordered, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            deconv3x3s2_neon(bottom_blob, top_blob_bordered, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob_bordered, opt);
            }
        }
        else if (kernel_w == 4 && kernel_h == 4 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            deconv4x4s1_neon(bottom_blob, top_blob_bordered, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob_bordered, opt);
            }
        }
        else if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            deconv4x4s2_neon(bottom_blob, top_blob_bordered, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob_bordered, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_pack1 + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const float* sptr = m.row(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = sptr[sx];

                                    int k = y * kernel_w + x;

                                    float w = kptr[k];

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        if (activation_type == 1)
                        {
                            sum = std::max(sum, 0.f);
                        }
                        else if (activation_type == 2)
                        {
                            float slope = activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (activation_type == 3)
                        {
                            float min = activation_params[0];
                            float max = activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (activation_type == 4)
                        {
                            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Deconvolution_arm::create_pipeline_fp16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        elempack = opt.use_fp16_arithmetic && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_fp16.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            Mat g0 = weight_data_fp16.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                __fp16* g00 = g0.row<__fp16>(p / elempack);

                for (int k = 0; k < maxk; k++)
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

    if (elempack == 1 && out_elempack == 1 && opt.use_fp16_arithmetic)
    {
        if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int Deconvolution_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr + k * 16));
                                    float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 4));
                                    float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 8));
                                    float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 12));

                                    _sum = vfmaq_laneq_f32(_sum, _w0, _val, 0);
                                    _sum = vfmaq_laneq_f32(_sum, _w1, _val, 1);
                                    _sum = vfmaq_laneq_f32(_sum, _w2, _val, 2);
                                    _sum = vfmaq_laneq_f32(_sum, _w3, _val, 3);
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float32x4_t _val = vdupq_n_f32((float)sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                    _sum = vfmaq_f32(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                    float32x4_t _s4 = vmulq_f32(_val, _w);

                                    sum += vaddvq_f32(_s4); // dot
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = (float)sptr[sx];

                                    int k = y * kernel_w + x;

                                    float w = (float)kptr[k];

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Deconvolution_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    if (elempack == 8 && out_elempack == 8)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                    float16x8_t _val = vld1q_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x8_t _w0 = vld1q_f16(kptr + k * 64);
                                    float16x8_t _w1 = vld1q_f16(kptr + k * 64 + 8);
                                    float16x8_t _w2 = vld1q_f16(kptr + k * 64 + 16);
                                    float16x8_t _w3 = vld1q_f16(kptr + k * 64 + 24);
                                    float16x8_t _w4 = vld1q_f16(kptr + k * 64 + 32);
                                    float16x8_t _w5 = vld1q_f16(kptr + k * 64 + 40);
                                    float16x8_t _w6 = vld1q_f16(kptr + k * 64 + 48);
                                    float16x8_t _w7 = vld1q_f16(kptr + k * 64 + 56);

                                    _sum = vfmaq_laneq_f16(_sum, _w0, _val, 0);
                                    _sum = vfmaq_laneq_f16(_sum, _w1, _val, 1);
                                    _sum = vfmaq_laneq_f16(_sum, _w2, _val, 2);
                                    _sum = vfmaq_laneq_f16(_sum, _w3, _val, 3);
                                    _sum = vfmaq_laneq_f16(_sum, _w4, _val, 4);
                                    _sum = vfmaq_laneq_f16(_sum, _w5, _val, 5);
                                    _sum = vfmaq_laneq_f16(_sum, _w6, _val, 6);
                                    _sum = vfmaq_laneq_f16(_sum, _w7, _val, 7);
                                }
                            }

                            kptr += maxk * 64;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f16(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float16x8_t _val = vdupq_n_f16(sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float16x8_t _w = vld1q_f16(kptr + k * 8);

                                    _sum = vfmaq_f16(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 8;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f16(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float16x4_t _val = vld1_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x8_t _w0 = vld1q_f16(kptr + k * 32);
                                    float16x8_t _w1 = vld1q_f16(kptr + k * 32 + 8);
                                    float16x8_t _w2 = vld1q_f16(kptr + k * 32 + 16);
                                    float16x8_t _w3 = vld1q_f16(kptr + k * 32 + 24);

                                    _sum = vfmaq_lane_f16(_sum, _w0, _val, 0);
                                    _sum = vfmaq_lane_f16(_sum, _w1, _val, 1);
                                    _sum = vfmaq_lane_f16(_sum, _w2, _val, 2);
                                    _sum = vfmaq_lane_f16(_sum, _w3, _val, 3);
                                }
                            }

                            kptr += maxk * 32;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f16(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                    float16x8_t _val = vld1q_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x8_t _w = vld1q_f16(kptr + k * 8);

                                    float16x8_t _s8 = vmulq_f16(_val, _w);

                                    float16x4_t _s4 = vadd_f16(vget_low_f16(_s8), vget_high_f16(_s8));
                                    sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot
                                }
                            }

                            kptr += maxk * 8;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                    float16x8_t _val = vld1q_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x4_t _w0 = vld1_f16(kptr + k * 32);
                                    float16x4_t _w1 = vld1_f16(kptr + k * 32 + 4);
                                    float16x4_t _w2 = vld1_f16(kptr + k * 32 + 8);
                                    float16x4_t _w3 = vld1_f16(kptr + k * 32 + 12);
                                    float16x4_t _w4 = vld1_f16(kptr + k * 32 + 16);
                                    float16x4_t _w5 = vld1_f16(kptr + k * 32 + 20);
                                    float16x4_t _w6 = vld1_f16(kptr + k * 32 + 24);
                                    float16x4_t _w7 = vld1_f16(kptr + k * 32 + 28);

                                    _sum = vfma_laneq_f16(_sum, _w0, _val, 0);
                                    _sum = vfma_laneq_f16(_sum, _w1, _val, 1);
                                    _sum = vfma_laneq_f16(_sum, _w2, _val, 2);
                                    _sum = vfma_laneq_f16(_sum, _w3, _val, 3);
                                    _sum = vfma_laneq_f16(_sum, _w4, _val, 4);
                                    _sum = vfma_laneq_f16(_sum, _w5, _val, 5);
                                    _sum = vfma_laneq_f16(_sum, _w6, _val, 6);
                                    _sum = vfma_laneq_f16(_sum, _w7, _val, 7);
                                }
                            }

                            kptr += maxk * 32;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float16x4_t _val = vld1_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x4_t _w0 = vld1_f16(kptr + k * 16);
                                    float16x4_t _w1 = vld1_f16(kptr + k * 16 + 4);
                                    float16x4_t _w2 = vld1_f16(kptr + k * 16 + 8);
                                    float16x4_t _w3 = vld1_f16(kptr + k * 16 + 12);

                                    _sum = vfma_lane_f16(_sum, _w0, _val, 0);
                                    _sum = vfma_lane_f16(_sum, _w1, _val, 1);
                                    _sum = vfma_lane_f16(_sum, _w2, _val, 2);
                                    _sum = vfma_lane_f16(_sum, _w3, _val, 3);
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float16x4_t _val = vdup_n_f16(sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float16x4_t _w = vld1_f16(kptr + k * 4);

                                    _sum = vfma_f16(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float16x4_t _val = vld1_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x4_t _w = vld1_f16(kptr + k * 4);

                                    float16x4_t _s4 = vmul_f16(_val, _w);

                                    sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            deconv4x4s2_fp16sa_neon(bottom_blob, top_blob_bordered, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob_bordered, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    __fp16 val = sptr[sx];

                                    int k = y * kernel_w + x;

                                    __fp16 w = kptr[k];

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int Deconvolution_arm::create_pipeline_bf16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = opt.use_packing_layout && num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_bf16.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            Mat g0 = weight_data_bf16.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                unsigned short* g00 = g0.row<unsigned short>(p / elempack);

                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = float32_to_bfloat16(k00[k]);

                            g00++;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int Deconvolution_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const unsigned short* sptr = m.row<const unsigned short>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr + k * 16));
                                    float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr + k * 16 + 4));
                                    float32x4_t _w2 = vcvt_f32_bf16(vld1_u16(kptr + k * 16 + 8));
                                    float32x4_t _w3 = vcvt_f32_bf16(vld1_u16(kptr + k * 16 + 12));

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
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_u16(outptr + j * 4, vcvt_bf16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const unsigned short* sptr = m.row<const unsigned short>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[sx]));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr + k * 4));

                                    _sum = vmlaq_f32(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_u16(outptr + j * 4, vcvt_bf16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const unsigned short* sptr = m.row<const unsigned short>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr + k * 4));

                                    float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                    sum += vaddvq_f32(_s4); // dot
#else
                                    float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                    _ss = vpadd_f32(_ss, _ss);
                                    sum += vget_lane_f32(_ss, 0);
#endif
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const unsigned short* sptr = m.row<const unsigned short>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = bfloat16_to_float32(sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float w = bfloat16_to_float32(kptr[k]);

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        if (activation_type == 1)
                        {
                            sum = std::max(sum, 0.f);
                        }
                        else if (activation_type == 2)
                        {
                            float slope = activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (activation_type == 3)
                        {
                            float min = activation_params[0];
                            float max = activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (activation_type == 4)
                        {
                            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                        }

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
