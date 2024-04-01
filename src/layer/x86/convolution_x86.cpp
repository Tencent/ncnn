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

#include "convolution_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __SSSE3__
#include <tmmintrin.h>
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE4_1__
#endif // __SSSE3__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "convolution_3x3.h"
#include "convolution_5x5.h"

#include "convolution_3x3_winograd.h"
#include "convolution_packed.h"

#if NCNN_INT8
#include "convolution_3x3_int8.h"

#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"

#include "convolution_3x3_winograd_int8.h"
#endif // NCNN_INT8

#if __SSE2__
#include "convolution_3x3_pack1to4.h"

#if __AVX__
#include "convolution_3x3_pack1to8.h"
#include "convolution_3x3_pack8to1.h"
#include "convolution_3x3_pack8.h"
#include "convolution_2x2_pack8.h"

#if __AVX512F__
#include "convolution_3x3_pack16to1.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

Convolution_x86::Convolution_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    activation = 0;
    nT = 0;
    convolution_dilation1 = 0;
    gemm = 0;
}

static void convolution_transform_kernel_packed_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }
}

static bool test_prefer_winograd63(int num_input, int num_output, int w, int h)
{
    // winograd selection strategy (profiled on i7-7700 single thread)

    int minwh = std::min(w, h);

    if (num_input >= 64)
    {
        return false;
    }
    if (num_input >= 32)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 56)
                                         || (minwh >= 63 && minwh <= 130);
        if (num_output >= 16) return (minwh >= 13 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 140);
        if (num_output >= 8) return (minwh >= 11 && minwh <= 14)
                                        || (minwh >= 19 && minwh <= 20)
                                        || (minwh >= 31 && minwh <= 38)
                                        || (minwh >= 43 && minwh <= 44)
                                        || (minwh >= 55 && minwh <= 162);
        return false;
    }
    if (num_input >= 16)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 92)
                                         || (minwh >= 95 && minwh <= 188);
        if (num_output >= 16) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 27 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 74)
                                         || (minwh >= 81 && minwh <= 110)
                                         || (minwh >= 117 && minwh <= 170)
                                         || (minwh >= 177 && minwh <= 182);
        if (num_output >= 8) return (minwh >= 19 && minwh <= 20)
                                        || (minwh >= 33 && minwh <= 38)
                                        || (minwh >= 43 && minwh <= 44)
                                        || (minwh >= 47 && minwh <= 128)
                                        || (minwh >= 155 && minwh <= 210);
        return false;
    }
    if (num_input >= 8)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 7 && minwh <= 14)
                                         || (minwh >= 17 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 26)
                                         || (minwh >= 31 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 162);
        if (num_output >= 16) return minwh == 31 || minwh == 32
                                         || (minwh >= 39 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 212);
        if (num_output >= 8) return false;
        return false;
    }

    return false;
}

static bool test_prefer_winograd23(int num_input, int num_output, int w, int h)
{
    int minwh = std::min(w, h);

    if (num_input >= 512)
    {
        if (num_output >= 512) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 256) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 8) || (minwh >= 11 && minwh <= 12);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 6);
        return false;
    }
    if (num_input >= 256)
    {
        if (num_output >= 512) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 256) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 12);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 4);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 6);
        return false;
    }
    if (num_input >= 128)
    {
        if (num_output >= 512) return (minwh >= 3 && minwh <= 14);
        if (num_output >= 256) return (minwh >= 3 && minwh <= 8) || (minwh >= 11 && minwh <= 12);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 10);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 10);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 6);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 6);
        return false;
    }
    if (num_input >= 64)
    {
        if (num_output >= 512) return (minwh >= 3 && minwh <= 8) || (minwh >= 11 && minwh <= 12) || (minwh >= 15 && minwh <= 20);
        if (num_output >= 256) return (minwh >= 7 && minwh <= 8);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 8) || (minwh >= 19 && minwh <= 22);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 12);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 12);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 12);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 12);
        return false;
    }
    if (num_input >= 32)
    {
        if (num_output >= 512) return (minwh >= 3 && minwh <= 6) || (minwh >= 11 && minwh <= 12);
        if (num_output >= 256) return (minwh >= 3 && minwh <= 6) || (minwh >= 11 && minwh <= 12);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 4) || (minwh >= 7 && minwh <= 16);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 8);
        if (num_output >= 32) return (minwh >= 7 && minwh <= 8);
        if (num_output >= 16) return (minwh >= 7 && minwh <= 8);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 10);
        return false;
    }
    if (num_input >= 16)
    {
        if (num_output >= 512) return (minwh >= 11 && minwh <= 12);
        if (num_output >= 256) return (minwh >= 3 && minwh <= 12);
        if (num_output >= 128) return (minwh >= 3 && minwh <= 6)
                                          || (minwh >= 9 && minwh <= 18);
        if (num_output >= 64) return (minwh >= 3 && minwh <= 4)
                                         || (minwh >= 7 && minwh <= 8)
                                         || (minwh >= 11 && minwh <= 12)
                                         || (minwh >= 15 && minwh <= 18);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 4)
                                         || (minwh >= 9 && minwh <= 10);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 10);
        if (num_output >= 8) return (minwh >= 3 && minwh <= 8)
                                        || (minwh >= 11 && minwh <= 12);
        return false;
    }
    if (num_input >= 8)
    {
        if (num_output >= 128) return false;
        if (num_output >= 64) return (minwh >= 3 && minwh <= 4)
                                         || (minwh >= 7 && minwh <= 14)
                                         || (minwh >= 47 && minwh <= 48);
        if (num_output >= 32) return (minwh >= 3 && minwh <= 6)
                                         || (minwh >= 15 && minwh <= 16);
        if (num_output >= 16) return (minwh >= 3 && minwh <= 6)
                                         || (minwh >= 9 && minwh <= 14)
                                         || (minwh >= 47 && minwh <= 212);
        if (num_output >= 8) return true;
        return false;
    }

    return false;
}

int Convolution_x86::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);
    nT = opt.num_threads;

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_x86(opt);
    }
#endif

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    if (!opt.use_packing_layout && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        convolution_dilation1 = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

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

        weight_data.release();

        return 0;
    }

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        elempack = num_input % 16 == 0 ? 16 : num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if ((bottom_shapes.empty() || bottom_shapes[0].w == 0 || bottom_shapes[0].h == 0) && (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0))
        {
            // dynamic shape
            if ((opt.use_winograd63_convolution) && (num_input <= 32 && num_output <= 32))
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if (opt.use_winograd43_convolution)
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            int w;
            int h;
            if (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0)
            {
                w = bottom_shapes[0].w;
                h = bottom_shapes[0].h;

                // make padding
                if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
                {
                    w += pad_left + pad_right;
                    h += pad_top + pad_bottom;
                }
                else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                         || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
                {
                    // tensorflow padding=SAME or onnx padding=SAME_UPPER/SAME_LOWER
                    w += 2;
                    h += 2;
                }
            }
            else
            {
                w = top_shapes[0].w + 2;
                h = top_shapes[0].h + 2;
            }

            bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
            bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
            bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

            if (prefer_winograd23 && !opt.use_winograd23_convolution)
            {
                // f23 fallback to f43
                prefer_winograd23 = false;
                prefer_winograd43 = true;
            }

            if (prefer_winograd63 && !opt.use_winograd63_convolution)
            {
                // f63 fallback to f43
                prefer_winograd63 = false;
                prefer_winograd43 = true;
            }

            if (prefer_winograd43 && !opt.use_winograd43_convolution)
            {
                // f43 fallback to f63 or f23
                prefer_winograd43 = false;
                if (opt.use_winograd63_convolution)
                {
                    prefer_winograd63 = true;
                }
                else
                {
                    prefer_winograd23 = true;
                }
            }

            if (prefer_winograd23)
            {
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
            }
            else if (prefer_winograd43)
            {
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            }
            else if (prefer_winograd63)
            {
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
            }
            else
            {
                // should never reach here
            }
        }

        weight_data.release();

        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(float) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        const int maxk = kernel_w * kernel_h;

        gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);

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
        if ((elempack == 16 && out_elempack == 1 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 8 && out_elempack == 8 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 8 && out_elempack == 8 && kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 1 && out_elempack == 8 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 1 && out_elempack == 8 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (elempack == 8 && out_elempack == 1 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else
        {
            convolution_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
    }

    weight_data.release();

    return 0;
}

int Convolution_x86::destroy_pipeline(const Option& opt)
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

    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int Convolution_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        Mat bottom_blob_3d;
        if (bottom_blob.elemsize % 16 == 0)
        {
            bottom_blob_3d = bottom_blob;
            bottom_blob_3d.dims = 3;
            bottom_blob_3d.w = 1;
            bottom_blob_3d.h = 1;
            bottom_blob_3d.c = bottom_blob.w;
            bottom_blob_3d.cstep = 1;
        }
        else
        {
            bottom_blob_3d = bottom_blob.reshape(1, 1, bottom_blob.w, opt.workspace_allocator);
        }

        Mat top_blob_3d;
        int ret = forward(bottom_blob_3d, top_blob_3d, opt);
        if (ret != 0)
            return ret;

        if (top_blob_3d.elemsize % 16 == 0)
        {
            top_blob = top_blob_3d;
            top_blob.dims = 1;
            top_blob.w = top_blob_3d.c;
            top_blob.h = 1;
            top_blob.c = 1;
            bottom_blob_3d.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
        }

        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

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
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (!opt.use_packing_layout && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        if (outw >= dilation_w && outh >= dilation_h)
        {
            return forwardDilation_x86(bottom_blob_bordered, top_blob, opt);
        }
    }

    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
        bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
        bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

        if (prefer_winograd23 && (!opt.use_winograd23_convolution || weight_winograd23_data.empty()))
        {
            // f23 fallback to f43
            prefer_winograd23 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd63 && (!opt.use_winograd63_convolution || weight_winograd63_data.empty()))
        {
            // f63 fallback to f43
            prefer_winograd63 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd43 && (!opt.use_winograd43_convolution || weight_winograd43_data.empty()))
        {
            // f43 fallback to f63 or f23
            prefer_winograd43 = false;
            if (opt.use_winograd63_convolution && !weight_winograd63_data.empty())
            {
                prefer_winograd63 = true;
            }
            else
            {
                prefer_winograd23 = true;
            }
        }

        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution winograd will use load-time value %d", opt.num_threads, nT);
        }

        if (prefer_winograd23)
        {
            conv3x3s1_winograd23(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd43)
        {
            conv3x3s1_winograd43(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd63)
        {
            conv3x3s1_winograd63(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, _nT, opt);
        }
        else
        {
            // should never reach here
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(float) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const float* sptr = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m512 _val = _mm512_load_ps(sptr);
                            _mm512_store_ps(ptr, _val);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }

                        sptr += gap;
                    }
                }
            }
#endif // __AVX512F__

            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const float* sptr = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m256 _val = _mm256_load_ps(sptr);
                            _mm256_store_ps(ptr, _val);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }

                        sptr += gap;
                    }
                }
            }
#endif // __AVX__

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const float* sptr = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m128 _val = _mm_load_ps(sptr);
                            _mm_store_ps(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
#endif // __SSE2__

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const float* sptr = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p);

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const float* sptr = img.row(dilation_h * u) + dilation_w * v * 16;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    __m512 _val = _mm512_load_ps(sptr);
                                    _mm512_store_ps(ptr, _val);

                                    sptr += stride_w * 16;
                                    ptr += 16;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }
#endif // __AVX512F__

            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const float* sptr = img.row(dilation_h * u) + dilation_w * v * 8;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    __m256 _val = _mm256_load_ps(sptr);
                                    _mm256_store_ps(ptr, _val);

                                    sptr += stride_w * 8;
                                    ptr += 8;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }
#endif // __AVX__

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const float* sptr = img.row(dilation_h * u) + dilation_w * v * 4;

                            for (int i = 0; i < outh; i++)
                            {
                                for (int j = 0; j < outw; j++)
                                {
                                    __m128 _val = _mm_load_ps(sptr);
                                    _mm_store_ps(ptr, _val);

                                    sptr += stride_w * 4;
                                    ptr += 4;
                                }

                                sptr += gap;
                            }
                        }
                    }
                }
            }
#endif // __SSE2__

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < channels; p++)
                {
                    const Mat img = bottom_blob_bordered.channel(p);
                    float* ptr = bottom_im2col.row(p * maxk);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            const float* sptr = img.row(dilation_h * u) + dilation_w * v;

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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16 && out_elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack16to1_avx512(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
        }
#endif // __AVX512F__

        if (elempack == 8 && out_elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
            if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv2x2s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

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
                conv3x3s1_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv3x3s2_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
        }

        if (elempack == 8 && out_elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
        }
#endif // __AVX__

        if (elempack == 1 && out_elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                conv3x3s1_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv3x3s2_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
                return 0;
            }
        }
#endif // __SSE2__

        convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    }

    return 0;
}

int Convolution_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_INT8
int Convolution_x86::create_pipeline_int8_x86(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if (opt.use_winograd43_convolution)
            conv3x3s1_winograd43_transform_kernel_int8(weight_data, weight_winograd43_data, num_input, num_output, opt);
        else
            conv3x3s1_winograd23_transform_kernel_int8(weight_data, weight_winograd23_data, num_input, num_output, opt);
    }
    else if (opt.use_sgemm_convolution)
    {
        convolution_im2col_gemm_transform_kernel_int8(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
    }
    else
    {
        convolution_transform_kernel_packed_int8(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
    }

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // requantize and relu
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

int Convolution_x86::forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    //     NCNN_LOGE("Convolution_x86 input %d x %d  ksize=%d %d  stride=%d %d", w, h, kernel_w, kernel_h, stride_w, stride_h);

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __SSE2__
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

    //     NCNN_LOGE("forward_int8_x86 %d %d %d    %d %d", w, h, bottom_blob_bordered.c, elempack, out_elempack);

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    int out_elempack_int32 = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        out_elempack_int32 = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __SSE2__

    Mat top_blob_int32;
    top_blob_int32.create(outw, outh, num_output / out_elempack_int32, (size_t)(4u * out_elempack_int32), out_elempack_int32, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input > 8 || num_output > 8);

    int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        // force num_threads the same as in create_pipeline
        // so we could use pre-packed A/B from the same tile config
        NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
    }

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if (opt.use_winograd43_convolution && !weight_winograd43_data.empty())
            conv3x3s1_winograd43_int8(bottom_blob_bordered, top_blob_int32, weight_winograd43_data, _nT, opt);
        else
            conv3x3s1_winograd23_int8(bottom_blob_bordered, top_blob_int32, weight_winograd23_data, _nT, opt);
    }
    else if (opt.use_sgemm_convolution)
    {
        convolution_im2col_gemm_int8(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);
    }
    else
    {
        convolution_packed_int8(bottom_blob_bordered, top_blob_int32, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    }

    if (use_int8_requantize)
    {
        requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
    }
    else
    {
        dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }

    return 0;
}
#endif // NCNN_INT8

int Convolution_x86::forwardDilation_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
