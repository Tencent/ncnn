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

#include "convolution_arm.h"

#include "benchmark.h"
#include "cpu.h"
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

#include "convolution_sgemm.h"
#include "convolution_sgemm_int8.h"

#include "convolution_1x1.h"
#include "convolution_1x1_bf16s.h"
#include "convolution_1x1_int8.h"
#include "convolution_2x2.h"
#include "convolution_3x3.h"
#include "convolution_3x3_int8.h"
#include "convolution_4x4.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"

#if __ARM_NEON
#include "convolution_1x1_pack4.h"
#include "convolution_1x1_pack4_bf16s.h"
#include "convolution_1x1_pack4to1.h"
#include "convolution_1x1_pack4to1_bf16s.h"
#include "convolution_3x3_pack1to4.h"
#include "convolution_3x3_pack1to4_bf16s.h"
#include "convolution_3x3_pack4.h"
#include "convolution_3x3_pack4_bf16s.h"
#include "convolution_3x3_pack4to1.h"
#include "convolution_3x3_pack4to1_bf16s.h"
#include "convolution_5x5_pack4.h"
#include "convolution_5x5_pack4_bf16s.h"
#include "convolution_7x7_pack1to4.h"
#include "convolution_7x7_pack1to4_bf16s.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "convolution_1x1_fp16s.h"
#include "convolution_1x1_pack4_fp16s.h"
#include "convolution_1x1_pack8_fp16s.h"
#include "convolution_1x1_pack4to8_fp16s.h"
#include "convolution_1x1_pack8to1_fp16s.h"
#include "convolution_1x1_pack8to4_fp16s.h"
#include "convolution_3x3_pack4_fp16s.h"
#include "convolution_3x3_pack1to8_fp16s.h"
#include "convolution_3x3_pack8_fp16s.h"
#include "convolution_3x3_pack8to1_fp16s.h"
#include "convolution_3x3_pack8to4_fp16s.h"
#include "convolution_5x5_pack8_fp16s.h"
#include "convolution_7x7_pack1to8_fp16s.h"
#endif
#endif // __ARM_NEON

Convolution_arm::Convolution_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;

    activation = 0;
    convolution_dilation1 = 0;
}

int Convolution_arm::create_pipeline(const Option& opt)
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

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        support_packing = false;

        return create_pipeline_int8_arm(opt);
    }

    if ((!support_packing || !opt.use_packing_layout) && !opt.use_bf16_storage && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        convolution_dilation1 = ncnn::create_layer(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, 1);  // stride_w
        pd.set(13, 1); // stride_h
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, weight_data_size);

        convolution_dilation1->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = weight_data;
            weights[1] = bias_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = weight_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }

        convolution_dilation1->create_pipeline(opt);

        return 0;
    }

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = (support_packing && opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;

#if __ARM_NEON
    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4b-4a-kw-kh-inch/4a-outch/4b
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4a-kw-kh-inch/4a-outch
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
        use_winograd3x3 = false;
        use_sgemm1x1 = false;

        if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // winograd is slow on small channel count
            if (num_input >= 16 && num_output >= 16)
                use_winograd3x3 = true;

            if (use_winograd3x3)
            {
                //                 conv3x3s1_winograd64_transform_kernel_neon(weight_data, weight_3x3_winograd64_data, num_input, num_output);
                conv3x3s1_winograd64_transform_kernel_neon5(weight_data, weight_3x3_winograd64_data, num_input, num_output);
            }
        }

        // TODO assume more proper condition
        if (opt.use_sgemm_convolution && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (num_input >= 64 && num_output >= 64)
                use_sgemm1x1 = true;

            if (use_sgemm1x1)
            {
                conv1x1s1_sgemm_transform_kernel_neon(weight_data, weight_1x1_sgemm_data, num_input, num_output);
            }
        }

        if (impl_type > 0 && impl_type < 6 && impl_type != 4)
        {
            switch (impl_type)
            {
            case 1:
                // winograd
                conv3x3s1_winograd64_transform_kernel_neon5(weight_data, weight_3x3_winograd64_data, num_input, num_output);
                break;
            case 2:
                // pointwise
                conv1x1s1_sgemm_transform_kernel_neon(weight_data, weight_1x1_sgemm_data, num_input, num_output);
                break;
            case 3:
                // im2col
                conv_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, maxk);
                break;
            //                 case 4:
            //                     // direct
            //                     break;
            case 5:
                // conv3x3s2
                conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, num_input, num_output);
                break;
            }
        }

        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, num_input, num_output);
        }

        if (opt.use_sgemm_convolution && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, maxk);
        }

        if (opt.use_sgemm_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, maxk);
        }
    }

    return 0;
}

int Convolution_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    if (convolution_dilation1)
    {
        convolution_dilation1->destroy_pipeline(opt);
        delete convolution_dilation1;
        convolution_dilation1 = 0;
    }

    return 0;
}

int Convolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_arm(bottom_blob, top_blob, opt);
    }

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
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

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        if (outw >= dilation_w && outh >= dilation_h)
        {
            return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
        }
    }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);

                                float32x4_t _w0 = vld1q_f32(kptr);
                                float32x4_t _w1 = vld1q_f32(kptr + 4);
                                float32x4_t _w2 = vld1q_f32(kptr + 8);
                                float32x4_t _w3 = vld1q_f32(kptr + 12);

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

                                kptr += 16;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vdupq_n_f32(sptr[space_ofs[k]]);
                                float32x4_t _w = vld1q_f32(kptr);
                                _sum = vmlaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            //             conv3x3s1_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);
                                float32x4_t _w = vld1q_f32(kptr);
                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif

                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (impl_type > 0 && impl_type < 6 && impl_type != 4)
        {
            // engineering is magic.
            switch (impl_type)
            {
            case 1:
                conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
                break;
            case 2:
                conv1x1s1_sgemm_neon(bottom_blob_bordered, top_blob, weight_1x1_sgemm_data, bias_data, opt);
                break;
            case 3:
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
                break;
            //                 case 4: FIXME fallback to auto path
            //                     conv(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            //                     break;
            case 5:
                conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data, bias_data, opt);
                break;
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (use_sgemm1x1)
            {
                conv1x1s1_sgemm_neon(bottom_blob_bordered, top_blob, weight_1x1_sgemm_data, bias_data, opt);
            }
            else
            {
                conv1x1s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            if (opt.use_sgemm_convolution)
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
            else
                conv1x1s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (use_winograd3x3 && w <= 120 && h <= 120)
            {
                //                 conv3x3s1_winograd64_neon4(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
                conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
            }
            else
            {
                conv3x3s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            if (opt.use_sgemm_convolution && !(outw >= 8 && outh >= 8))
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
            else
                conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 4 && kernel_h == 4 && dilation_w == 1 && dilation_h == 1 && stride_w == 4 && stride_h == 4)
        {
            conv4x4s4_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv7x7s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
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
                        else if (activation_type == 5)
                        {
                            sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Convolution_arm::create_pipeline_fp16s(const Option& opt)
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

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack8_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack8_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack8_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to8_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to8_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack8to1_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack8to1_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack8to1_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack8to4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack8to4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack8to4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    // pack4
    if (elempack == 4 && out_elempack == 4 && opt.use_fp16_arithmetic)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    // pack1
    if (elempack == 1 && out_elempack == 1 && opt.use_fp16_arithmetic)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_fp16sa_neon(weight_data, weight_data_fp16, num_input, num_output);
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int Convolution_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // TODO dilated conv for bf16s
    //     if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    //     {
    //         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
    //     }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32((const float*)bias_data + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));

                                float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr));
                                float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr + 4));
                                float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr + 8));
                                float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr + 12));

                                _sum = vfmaq_laneq_f32(_sum, _w0, _val, 0);
                                _sum = vfmaq_laneq_f32(_sum, _w1, _val, 1);
                                _sum = vfmaq_laneq_f32(_sum, _w2, _val, 2);
                                _sum = vfmaq_laneq_f32(_sum, _w3, _val, 3);

                                kptr += 16;
                            }
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
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32((const float*)bias_data + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_f16(vdup_n_f16(sptr[space_ofs[k]]));
                                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
                                _sum = vfmaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
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
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));
                                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
                                float32x4_t _s4 = vmulq_f32(_val, _w);

                                sum += vaddvq_f32(_s4); // dot

                                kptr += 4;
                            }
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
                __fp16* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<__fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
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

    return 0;
}

int Convolution_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // TODO dilated conv for bf16s
    //     if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    //     {
    //         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
    //     }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16(((const __fp16*)bias_data_fp16) + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);

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

                                kptr += 64;
                            }
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
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16(((const __fp16*)bias_data_fp16) + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vdupq_n_f16(sptr[space_ofs[k]]);
                                float16x8_t _w = vld1q_f16(kptr);
                                _sum = vfmaq_f16(_sum, _val, _w);

                                kptr += 8;
                            }
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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16(((const __fp16*)bias_data_fp16) + p * 8);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);

                                float16x8_t _w0 = vld1q_f16(kptr);
                                float16x8_t _w1 = vld1q_f16(kptr + 8);
                                float16x8_t _w2 = vld1q_f16(kptr + 16);
                                float16x8_t _w3 = vld1q_f16(kptr + 24);

                                _sum = vfmaq_lane_f16(_sum, _w0, _val, 0);
                                _sum = vfmaq_lane_f16(_sum, _w1, _val, 1);
                                _sum = vfmaq_lane_f16(_sum, _w2, _val, 2);
                                _sum = vfmaq_lane_f16(_sum, _w3, _val, 3);

                                kptr += 32;
                            }
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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            //             conv3x3s1_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);
                                float16x8_t _w = vld1q_f16(kptr);
                                float16x8_t _s8 = vmulq_f16(_val, _w);

                                float16x4_t _s4 = vadd_f16(vget_low_f16(_s8), vget_high_f16(_s8));
                                sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

                                kptr += 8;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            //             conv3x3s1_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16(((const __fp16*)bias_data_fp16) + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);

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

                                kptr += 32;
                            }
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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            //             conv3x3s1_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16(((const __fp16*)bias_data_fp16) + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);

                                float16x4_t _w0 = vld1_f16(kptr);
                                float16x4_t _w1 = vld1_f16(kptr + 4);
                                float16x4_t _w2 = vld1_f16(kptr + 8);
                                float16x4_t _w3 = vld1_f16(kptr + 12);

                                _sum = vfma_lane_f16(_sum, _w0, _val, 0);
                                _sum = vfma_lane_f16(_sum, _w1, _val, 1);
                                _sum = vfma_lane_f16(_sum, _w2, _val, 2);
                                _sum = vfma_lane_f16(_sum, _w3, _val, 3);

                                kptr += 16;
                            }
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
                __fp16* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16(((const __fp16*)bias_data_fp16) + p * 4);
                        }

                        const __fp16* kptr = weight_data_fp16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vdup_n_f16(sptr[space_ofs[k]]);
                                float16x4_t _w = vld1_f16(kptr);
                                _sum = vfma_f16(_sum, _val, _w);

                                kptr += 4;
                            }
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
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);
                                float16x4_t _w = vld1_f16(kptr);
                                float16x4_t _s4 = vmul_f16(_val, _w);

                                sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot

                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const __fp16* sptr = m.row<__fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                __fp16 val = sptr[space_ofs[k]];
                                __fp16 w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int Convolution_arm::create_pipeline_bf16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = (support_packing && opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;

#if __ARM_NEON
    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_bf16s_neon(weight_data, weight_data_pack4_bf16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_bf16s_neon(weight_data, weight_data_pack4_bf16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_neon(weight_data, weight_data_pack4_bf16, num_input, num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4b-4a-kw-kh-inch/4a-outch/4b
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4_bf16.create(maxk, num_input / 4, num_output / 4, (size_t)2 * 16, 16);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);

                Mat g0 = weight_data_pack4_bf16.channel(q / 4);

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

                    unsigned short* g00 = g0.row<unsigned short>(p / 4);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k10[k]);
                        g00[2] = float32_to_bfloat16(k20[k]);
                        g00[3] = float32_to_bfloat16(k30[k]);

                        g00[4] = float32_to_bfloat16(k01[k]);
                        g00[5] = float32_to_bfloat16(k11[k]);
                        g00[6] = float32_to_bfloat16(k21[k]);
                        g00[7] = float32_to_bfloat16(k31[k]);

                        g00[8] = float32_to_bfloat16(k02[k]);
                        g00[9] = float32_to_bfloat16(k12[k]);
                        g00[10] = float32_to_bfloat16(k22[k]);
                        g00[11] = float32_to_bfloat16(k32[k]);

                        g00[12] = float32_to_bfloat16(k03[k]);
                        g00[13] = float32_to_bfloat16(k13[k]);
                        g00[14] = float32_to_bfloat16(k23[k]);
                        g00[15] = float32_to_bfloat16(k33[k]);

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
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack1to4_bf16.create(maxk, num_input, num_output / 4, (size_t)2 * 4, 4);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);

                Mat g0 = weight_data_pack1to4_bf16.channel(q / 4);

                for (int p = 0; p < num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    unsigned short* g00 = g0.row<unsigned short>(p);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k10[k]);
                        g00[2] = float32_to_bfloat16(k20[k]);
                        g00[3] = float32_to_bfloat16(k30[k]);

                        g00 += 4;
                    }
                }
            }
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_bf16s_neon(weight_data, weight_data_pack4to1_bf16, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_bf16s_neon(weight_data, weight_data_pack4to1_bf16, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1_bf16, num_input, num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 4a-kw-kh-inch/4a-outch
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4to1_bf16.create(maxk, num_input / 4, num_output, (size_t)2 * 4, 4);

            for (int q = 0; q < num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1_bf16.channel(q);

                for (int p = 0; p + 3 < num_input; p += 4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p + 1);
                    const float* k02 = k0.row(p + 2);
                    const float* k03 = k0.row(p + 3);

                    unsigned short* g00 = g0.row<unsigned short>(p / 4);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00[1] = float32_to_bfloat16(k01[k]);
                        g00[2] = float32_to_bfloat16(k02[k]);
                        g00[3] = float32_to_bfloat16(k03[k]);

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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_bf16s_neon(weight_data, weight_data_bf16, num_input, num_output);
        }
        else
        {
            ncnn::cast_float32_to_bfloat16(weight_data, weight_data_bf16, opt);
        }
    }

    return 0;
}

int Convolution_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // TODO dilated conv for bf16s
    //     if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    //     {
    //         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
    //     }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_pack4_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr + space_ofs[k] * 4));

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

                                kptr += 16;
                            }
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
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_pack1to4_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[space_ofs[k]]));
                                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));
                                _sum = vmlaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, bias_data, opt);

            //             conv3x3s1_pack4to1_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_pack4to1_bf16.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr + space_ofs[k] * 4));
                                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr));
                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif

                                kptr += 4;
                            }
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
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_bf16, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                unsigned short* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const unsigned short* sptr = m.row<unsigned short>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = bfloat16_to_float32(sptr[space_ofs[k]]);
                                float w = bfloat16_to_float32(kptr[k]);
                                sum += val * w;
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
                        else if (activation_type == 5)
                        {
                            sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
                        }

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_arm::create_pipeline_int8_arm(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    use_winograd3x3_int8 = false;
    use_sgemm1x1_int8 = false;

    if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        use_winograd3x3_int8 = true;
        //         conv3x3s1_winograd23_transform_kernel_int8_neon(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
        conv3x3s1_winograd43_transform_kernel_int8_neon(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        conv3x3s2_transform_kernel_int8_neon(weight_data, weight_3x3s2_data_int8, num_input, num_output);
    }
    else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        use_sgemm1x1_int8 = true;
        conv1x1s1_sgemm_transform_kernel_int8_neon(weight_data, weight_1x1s1_sgemm_data_int8, num_input, num_output);
    }
    else
    {
        conv_im2col_sgemm_transform_kernel_int8_neon(weight_data, weight_sgemm_data_int8, num_input, num_output, maxk);
    }

    return 0;
}

int Convolution_arm::forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dilation_w > 1 || dilation_h > 1)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    // int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Convolution_arm input %d x %d  ksize=%d %d  stride=%d %d", w, h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scale, opt_g);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    // int8
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // int8
    if (use_int8_requantize == true)
    {
        Mat top_blob_tm;
        top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
        if (top_blob_tm.empty())
            return -100;

        if (use_sgemm1x1_int8)
        {
            std::vector<float> requantize_scales;
            for (int p = 0; p < num_output; p++)
            {
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                float scale_out = top_blob_int8_scale;

                requantize_scales.push_back(scale_in);
                requantize_scales.push_back(scale_out);
            }

            conv1x1s1_sgemm_int8_requant_neon(bottom_blob_bordered, top_blob, weight_1x1s1_sgemm_data_int8, bias_data, requantize_scales, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }
        else if (use_winograd3x3_int8)
        {
            //             conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);
            conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3s2_data_int8, opt);
        }
        else
        {
            conv_im2col_sgemm_int8_neon(bottom_blob_bordered, top_blob_tm, weight_sgemm_data_int8, kernel_w, kernel_h, stride_w, stride_h, opt);
        }

        // requantize, reverse scale inplace
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob.allocator;

            Mat top_blob_tm_g = top_blob_tm.channel_range(p, 1);
            Mat top_blob_g = top_blob.channel_range(p, 1);

            // requantize and relu
            float scale_in;
            if (weight_data_int8_scales[p] == 0)
                scale_in = 0;
            else
                scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

            float scale_out = top_blob_int8_scale; //FIXME load param

            requantize_int8_to_int8(top_blob_tm_g, top_blob_g, scale_in, scale_out, bias_term ? (const float*)bias_data + p : 0, bias_term ? 1 : 0, 0, opt_g);
        }
    }
    else
    {
        if (use_sgemm1x1_int8)
        {
            conv1x1s1_sgemm_int8_neon(bottom_blob_bordered, top_blob, weight_1x1s1_sgemm_data_int8, opt);
        }
        else if (use_winograd3x3_int8)
        {
            //             conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);
            conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data_int8, opt);
        }
        else
        {
            conv_im2col_sgemm_int8_neon(bottom_blob_bordered, top_blob, weight_sgemm_data_int8, kernel_w, kernel_h, stride_w, stride_h, opt);
        }

        // dequantize, reverse scale inplace
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob.allocator;

            Mat top_blob_g = top_blob.channel_range(p, 1);

            // dequantize
            float scale_in;
            if (weight_data_int8_scales[p] == 0)
                scale_in = 0;
            else
                scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

            dequantize_int32_to_float32(top_blob_g, scale_in, bias_term ? (const float*)bias_data + p : 0, bias_term ? 1 : 0, opt_g);
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

int Convolution_arm::forwardDilation_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Make (dilation * dilation) batches
    Mat inner_bottom_blob;
    Mat inner_top_blob;
    for (int x = 0; x < dilation; x++)
    {
        for (int y = 0; y < dilation; y++)
        {
            int inner_w = (w - y + dilation - 1) / dilation;
            int inner_h = (h - x + dilation - 1) / dilation;

            int inner_outw = (inner_w - kernel_size) / stride + 1;
            int inner_outh = (inner_h - kernel_size) / stride + 1;

            inner_bottom_blob.create(inner_w, inner_h, bottom_blob.c, elemsize, opt.workspace_allocator);
            if (inner_bottom_blob.empty())
                return -100;

            inner_top_blob.create(inner_outw, inner_outh, num_output, elemsize, opt.workspace_allocator);
            if (inner_top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < bottom_blob.c; c++)
            {
                float* outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i++)
                {
                    const float* ptr = (const float*)bottom_blob.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j++)
                    {
                        outptr[j] = ptr[j * dilation];
                    }
                    outptr += inner_w;
                }
            }

            Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            convolution_dilation1->forward(inner_bottom_blob, inner_top_blob, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < num_output; c++)
            {
                float* outptr = (float*)top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i++)
                {
                    const float* ptr = (const float*)inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j++)
                    {
                        outptr[j * dilation] = ptr[j];
                    }
                    outptr += dilation * outw;
                }
            }
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
