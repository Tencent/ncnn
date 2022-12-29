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

#include "convolution_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "convolution_fp16s.h"
#include "convolution_pack8_fp16s.h"
#include "convolution_pack1to8_fp16s.h"
#include "convolution_pack4to8_fp16s.h"
#include "convolution_pack8to1_fp16s.h"
#include "convolution_pack8to4_fp16s.h"
#include "convolution_pack4_fp16s.h"
#include "convolution_pack1to4_fp16s.h"
#include "convolution_pack4to1_fp16s.h"
#include "convolution_winograd_transform_fp16s.h"
#include "convolution_winograd_transform_pack4_fp16s.h"
#include "convolution_winograd_transform_pack8_fp16s.h"
#include "convolution_winograd_dot_pack4_fp16s.h"
#include "convolution_winograd_dot_pack8_fp16s.h"
#include "convolution_3x3_pack4_fp16s.h"
#include "convolution_3x3_pack1to8_fp16s.h"
#include "convolution_3x3_pack1to4_fp16s.h"
#include "convolution_3x3_pack8_fp16s.h"
#include "convolution_3x3_pack8to1_fp16s.h"
#include "convolution_3x3_pack8to4_fp16s.h"
#include "convolution_5x5_pack8_fp16s.h"
#include "convolution_7x7_pack1to8_fp16s.h"
#endif
#endif // __ARM_NEON

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void convolution_transform_kernel_packed_fp16s_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
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
}

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

    if (opt.use_fp16_arithmetic)
    {
        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);
    }

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input > 8 || num_output > 8))
        {
            if ((opt.use_winograd63_convolution && num_input >= 16 && num_output >= 16 && num_input <= 128 && num_output <= 128) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_pack8_fp16sa_neon(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_pack8_fp16sa_neon(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_pack8_fp16sa_neon(weight_data, weight_winograd23_data, num_input, num_output, opt);
            return 0;
        }
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            return 0;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            return 0;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            return 0;
        }
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        if (opt.use_winograd_convolution && opt.use_winograd63_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd63_transform_kernel_pack8to1_fp16sa_neon(weight_data, weight_winograd63_data, num_input, num_output, opt);
            return 0;
        }
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        if (opt.use_winograd_convolution && opt.use_winograd63_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd63_transform_kernel_pack8to4_fp16sa_neon(weight_data, weight_winograd63_data, num_input, num_output, opt);
            return 0;
        }
    }

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (opt.use_fp16_arithmetic && opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input > 4 || num_output > 4))
        {
            if ((opt.use_winograd63_convolution && num_input >= 8 && num_output >= 8 && num_input <= 64 && num_output <= 64) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_pack4_fp16sa_neon(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= 8 && num_output >= 8) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_pack4_fp16sa_neon(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_pack4_fp16sa_neon(weight_data, weight_winograd23_data, num_input, num_output, opt);
            return 0;
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        if (opt.use_fp16_arithmetic && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            return 0;
        }
        else if (opt.use_fp16_arithmetic && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            return 0;
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
    }

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(__fp16) * 2 > l2_cache_size || (num_input >= 16 || num_output >= 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        const int maxk = kernel_w * kernel_h;

        gemm = ncnn::create_layer(ncnn::LayerType::Gemm);

        ncnn::ParamDict pd;
        pd.set(2, 0);                   // transA
        pd.set(3, 0);                   // transB
        pd.set(4, 1);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, 1);                   // constantC
        pd.set(7, num_output);          // M = outch
        pd.set(8, 0);                   // N = size
        pd.set(9, maxk * num_input);    // K = maxk*inch
        pd.set(10, bias_term ? 1 : -1); // constant_broadcast_type_C = (M)
        pd.set(11, 1);                  // output_N1M

        gemm->load_param(pd);

        // maxk-inch-outch to pa-maxk-inch/pa-outch
        Mat tmp;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            tmp.create(maxk * num_input, num_output);

            for (int q = 0; q < num_output; q += 1)
            {
                float* g00 = tmp.row(q);

                for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < elempack; i++)
                        {
                            const float* k00 = weight_data_r2.channel(q).row(p + i);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }

        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = tmp;
            weights[1] = bias_data;

            gemm->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = tmp;

            gemm->load_model(ModelBinFromMatArray(weights));
        }

        gemm->create_pipeline(opt);
    }
    else
    {
        convolution_transform_kernel_packed_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

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
    int out_elempack = (opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    // TODO dilated conv for bf16s
    //     if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    //     {
    //         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
    //     }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(__fp16) * 2 > l2_cache_size || (num_input >= 16 || num_output >= 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        // im2col
        Mat bottom_im2col;
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            bottom_im2col = bottom_blob_bordered;
            bottom_im2col.w = w * h;
            bottom_im2col.h = 1;
        }
        else if (kernel_w == 1 && kernel_h == 1)
        {
            const int size = outw * outh;

            bottom_im2col.create(size, channels, elemsize, elempack, opt.workspace_allocator);
            if (bottom_im2col.empty())
                return -100;

            const int gap = (w * stride_h - outw * stride_w) * elempack;

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const __fp16* sptr = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x4_t _val = vld1_f16(sptr);
                            vst1_f16(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const __fp16* sptr = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
        else
        {
            const int size = outw * outh;
            const int maxk = kernel_w * kernel_h;

            bottom_im2col.create(size, maxk * channels, elemsize, elempack, opt.workspace_allocator);
            if (bottom_im2col.empty())
                return -100;

            const int gap = (w * stride_h - outw * stride_w) * elempack;

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 4;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    float16x4_t _val = vld1_f16(sptr);
                                    vst1_f16(ptr, _val);

                                    sptr += stride_w * 4;
                                    ptr += 4;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    ptr[0] = sptr[0];

                                    sptr += stride_w;
                                    ptr += 1;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }
        }

        // sgemm
        {
            top_blob.w = outw * outh;
            top_blob.h = 1;
        }
        Option opt_b = opt;
        opt_b.blob_allocator = top_blob.allocator;
        gemm->forward(bottom_im2col, top_blob, opt_b);
        {
            top_blob.w = outw;
            top_blob.h = outh;
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }
    else
    {
#if __ARM_NEON
        if (elempack == 4 && out_elempack == 4)
        {
            convolution_pack4_fp16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 1 && out_elempack == 4)
        {
            convolution_pack1to4_fp16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 4 && out_elempack == 1)
        {
            convolution_pack4to1_fp16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
#endif // __ARM_NEON

        if (elempack == 1 && out_elempack == 1)
        {
            convolution_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
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
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
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

    const int num_input = channels * elempack;

    if (elempack == 8 && out_elempack == 8)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input > 8 || num_output > 8))
        {
            if ((opt.use_winograd63_convolution && num_input >= 16 && num_output >= 16 && num_input <= 128 && num_output <= 128) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, opt);
            else if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data_fp16, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
    }

    if (elempack == 8 && out_elempack == 1)
    {
        if (opt.use_winograd_convolution && opt.use_winograd63_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd63_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, opt);

            //             conv3x3s1_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        if (opt.use_winograd_convolution && opt.use_winograd63_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd63_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, opt);

            //             conv3x3s1_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input > 4 || num_output > 4))
        {
            if ((opt.use_winograd63_convolution && num_input >= 8 && num_output >= 8 && num_input <= 64 && num_output <= 64) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, opt);
            else if ((opt.use_winograd43_convolution && num_input >= 8 && num_output >= 8) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data_fp16, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
    }

    if (elempack == 1 && out_elempack == 1)
    {
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(__fp16) * 2 > l2_cache_size || (num_input >= 16 || num_output >= 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        // im2col
        Mat bottom_im2col;
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            bottom_im2col = bottom_blob_bordered;
            bottom_im2col.w = w * h;
            bottom_im2col.h = 1;
        }
        else if (kernel_w == 1 && kernel_h == 1)
        {
            const int size = outw * outh;

            bottom_im2col.create(size, channels, elemsize, elempack, opt.workspace_allocator);
            if (bottom_im2col.empty())
                return -100;

            const int gap = (w * stride_h - outw * stride_w) * elempack;

            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const __fp16* sptr = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x8_t _val = vld1q_f16(sptr);
                            vst1q_f16(ptr, _val);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }

                        sptr += gap;
                    }
                }
            }

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const __fp16* sptr = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x4_t _val = vld1_f16(sptr);
                            vst1_f16(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const __fp16* sptr = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
        else
        {
            const int size = outw * outh;
            const int maxk = kernel_w * kernel_h;

            bottom_im2col.create(size, maxk * channels, elemsize, elempack, opt.workspace_allocator);
            if (bottom_im2col.empty())
                return -100;

            const int gap = (w * stride_h - outw * stride_w) * elempack;

            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 8;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    float16x8_t _val = vld1q_f16(sptr);
                                    vst1q_f16(ptr, _val);

                                    sptr += stride_w * 8;
                                    ptr += 8;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 4;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    float16x4_t _val = vld1_f16(sptr);
                                    vst1_f16(ptr, _val);

                                    sptr += stride_w * 4;
                                    ptr += 4;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    __fp16* ptr = bottom_im2col.row<__fp16>(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    ptr[0] = sptr[0];

                                    sptr += stride_w;
                                    ptr += 1;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }
        }

        // sgemm
        {
            top_blob.w = outw * outh;
            top_blob.h = 1;
        }
        Option opt_b = opt;
        opt_b.blob_allocator = top_blob.allocator;
        gemm->forward(bottom_im2col, top_blob, opt_b);
        {
            top_blob.w = outw;
            top_blob.h = outh;
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }
    else
    {
        if (elempack == 8 && out_elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv3x3s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv5x5s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv5x5s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                convolution_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
            }
        }

        if (elempack == 1 && out_elempack == 8)
        {
            convolution_pack1to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 4 && out_elempack == 8)
        {
            convolution_pack4to8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 8 && out_elempack == 1)
        {
            convolution_pack8to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 8 && out_elempack == 4)
        {
            convolution_pack8to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 4 && out_elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                convolution_pack4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
            }
        }

        if (elempack == 1 && out_elempack == 4)
        {
            convolution_pack1to4_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 4 && out_elempack == 1)
        {
            convolution_pack4to1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }

        if (elempack == 1 && out_elempack == 1)
        {
            convolution_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
