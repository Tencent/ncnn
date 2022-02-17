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
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE4_1__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_1x1.h"
#include "convolution_3x3.h"
#include "convolution_5x5.h"

#if NCNN_INT8
#include "convolution_sgemm_int8.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3_int8.h"
#include "convolution_int8.h"
#endif // NCNN_INT8

#if __SSE2__
#include "convolution_pack4.h"
#include "convolution_pack1to4.h"
#include "convolution_pack4to1.h"

#include "convolution_sgemm_pack4.h"
#include "convolution_sgemm_pack1to4.h"
#include "convolution_sgemm_pack4to1.h"
#include "convolution_1x1_pack4.h"
#include "convolution_1x1_pack1to4.h"
#include "convolution_1x1_pack4to1.h"
#include "convolution_3x3_pack4.h"
#include "convolution_3x3_pack4to1.h"
#include "convolution_3x3_pack1to4.h"

#if NCNN_INT8
#include "convolution_pack8to4_int8.h"
#include "convolution_pack1to4_int8.h"
#include "convolution_pack8to1_int8.h"
#include "convolution_sgemm_pack8to4_int8.h"
#include "convolution_sgemm_pack1to4_int8.h"
#include "convolution_sgemm_pack8to1_int8.h"
#include "convolution_1x1_pack8to4_int8.h"
#include "convolution_1x1_pack1to4_int8.h"
#include "convolution_1x1_pack8to1_int8.h"
#include "convolution_3x3_pack8to4_int8.h"
#include "convolution_3x3_pack1to4_int8.h"
#include "convolution_3x3_pack8to1_int8.h"
#include "convolution_7x7_pack1to4_int8.h"
#endif // NCNN_INT8

#if __AVX__
#include "convolution_pack8.h"
#include "convolution_pack4to8.h"
#include "convolution_pack1to8.h"
#include "convolution_pack8to4.h"
#include "convolution_pack8to1.h"

#include "convolution_sgemm_pack8.h"
#include "convolution_sgemm_pack4to8.h"
#include "convolution_sgemm_pack1to8.h"
#include "convolution_sgemm_pack8to4.h"
#include "convolution_sgemm_pack8to1.h"
#include "convolution_1x1_pack8.h"
#include "convolution_1x1_pack4to8.h"
#include "convolution_1x1_pack1to8.h"
#include "convolution_1x1_pack8to4.h"
#include "convolution_1x1_pack8to1.h"
#include "convolution_3x3_pack1to8.h"
#include "convolution_3x3_pack8to1.h"
#include "convolution_3x3_pack8.h"
#include "convolution_2x2_pack8.h"
#endif
#endif // __SSE2__

Convolution_x86::Convolution_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    activation = 0;
    convolution_dilation1 = 0;
}

static void convolution_transform_kernel_packed_sse(const Mat& weight_data, Mat& weight_data_packed, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

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

int Convolution_x86::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

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

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

#if __SSE2__
#if __AVX__
    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (num_input >= 16 && num_output >= 16)
            {
                conv3x3s1_winograd64_transform_kernel_pack8_avx(weight_data, weight_3x3_winograd64_data, num_input, num_output, opt);
                conv3x3s1_winograd42_transform_kernel_pack8_avx(weight_data, weight_3x3_winograd42_data, num_input, num_output, opt);
            }
            else
            {
                convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            }
        }
        else if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_winograd_convolution)
            {
                conv3x3s1_winograd64_transform_kernel_pack8to1_avx(weight_data, weight_3x3_winograd64_data, num_input, num_output, opt);
            }
            else
            {
                convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_sse(weight_data, weight_3x3_winograd64_data, num_input, num_output, opt);
            conv3x3s1_winograd42_transform_kernel_pack4_sse(weight_data, weight_3x3_winograd42_data, num_input, num_output, opt);
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
            }
            else
            {
                convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            }
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
            }
            else
            {
                convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            }
        }
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4to1_sse(weight_data, weight_3x3_winograd64_data, num_input, num_output, opt);
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
            }
            else
            {
                convolution_transform_kernel_packed_sse(weight_data, weight_data_packed, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
            }
        }
    }
#endif

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_winograd_convolution && num_input >= 16 && num_output >= 16)
            {
                conv3x3s1_winograd23_transform_kernel_sse(weight_data, weight_data_3x3_winograd23, num_input, num_output, opt);
                // conv3x3s1_winograd43_transform_kernel_sse(weight_data, weight_3x3_winograd43_data, num_input, num_output, opt);
            }

            convolution_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
    }

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

    return 0;
}

int Convolution_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }
#endif

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
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
#if __AVX__
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

#if __SSE2__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (num_input >= 16 && num_output >= 16)
            {
                // we need more proper conditions
                if ((w <= 10 || (w >= 15 && w <= 18) || w == 21 || w == 22) && (h <= 10 || (h >= 15 && h <= 18) || h == 21 || h == 22))
                {
                    conv3x3s1_winograd42_pack8_avx(bottom_blob_bordered, top_blob, weight_3x3_winograd42_data, bias_data, opt);
                }
                else
                {
                    conv3x3s1_winograd64_pack8_avx(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
                }
            }
            else
            {
                conv3x3s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv2x2s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack1to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack1to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack4to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack4to8_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack4to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to1_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack8to1_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_winograd_convolution)
            {
                // TODO more proper condition
                conv3x3s1_winograd64_pack8to1_avx(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
            }
            else
            {
                conv3x3s1_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack8to1_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to4_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack8to4_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack8to4_avx(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack8to4_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif

    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            // we need more proper conditions
            if ((w <= 10 || (w >= 15 && w <= 18) || w == 21 || w == 22) && (h <= 10 || (h >= 15 && h <= 18) || h == 21 || h == 22))
            {
                conv3x3s1_winograd42_pack4_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd42_data, bias_data, opt);
            }
            else
            {
                conv3x3s1_winograd64_pack4_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_pack4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                convolution_pack4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1to4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack1to4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_pack1to4_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                convolution_pack1to4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack4to1_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            // TODO more proper condition
            conv3x3s1_winograd64_pack4to1_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);

            // conv3x3s1_pack4to1_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (num_input >= 12 || num_output >= 12))
                                || (dilation_w == 1 && dilation_h == 1 && (stride_w >= 2 || stride_h >= 2) && (num_input >= 16 || num_output >= 16))
                                || ((dilation_w >= 2 || dilation_h >= 2) && (num_input >= 16 || num_output >= 16));

            if (opt.use_sgemm_convolution && prefer_sgemm)
            {
                convolution_im2col_sgemm_pack4to1_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
                convolution_pack4to1_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_winograd_convolution && num_input >= 16 && num_output >= 16 && outw >= 8 && outh >= 8)
            {
                conv3x3s1_winograd23_sse(bottom_blob_bordered, top_blob, weight_data_3x3_winograd23, bias_data, opt);
                //             conv3x3s1_winograd43_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd43_data, bias_data, opt);
            }
            else
            {
                convolution_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
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
                                float wt = kptr[k];
                                sum += val * wt;
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

    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(21, dilation_h);
    pd.set(3, stride_w);
    pd.set(31, stride_h);
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
static void convolution_transform_kernel_packed_int8_sse(const Mat& weight_data, Mat& weight_data_int8, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_int8.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g00 = weight_data_int8.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        for (int j = 0; j < elempack; j++)
                        {
                            const signed char* k00 = weight_data_r2.channel(q + i).row<const signed char>(p + j);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }
}

int Convolution_x86::create_pipeline_int8_x86(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        elempack = num_input % 8 == 0 ? 8 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __SSE2__

#if __SSE2__
    if (elempack == 8 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd42_transform_kernel_pack8to4_int8_sse(weight_data, weight_3x3_winograd42_data, num_input, num_output, opt);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_int8_sse(weight_data, weight_data_int8, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution) // TODO better condition && num_input >= 8 && num_output >= 8)
        {
            convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_int8_sse(weight_data, weight_data_int8, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd42_transform_kernel_pack8to1_int8_sse(weight_data, weight_3x3_winograd42_data, num_input, num_output, opt);
        }
        else if (opt.use_sgemm_convolution) // TODO better condition && num_input >= 8 && num_output >= 8)
        {
            convolution_im2col_sgemm_transform_kernel_pack8to1_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_int8_sse(weight_data, weight_data_int8, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            conv3x3s1_winograd23_transform_kernel_int8_sse(weight_data, weight_data_3x3_winograd23_int8, num_input, num_output, opt);
            //         conv3x3s1_winograd43_transform_kernel_int8_sse(weight_data, weight_data_3x3_winograd23_int8, num_input, num_output, opt);
        }

        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_int8_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h);
        }

        return 0;
    }

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

    //     NCNN_LOGE("Convolution_arm input %d x %d  ksize=%d %d  stride=%d %d", w, h, kernel_w, kernel_h, stride_w, stride_h);

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

    //     NCNN_LOGE("forward_int8_arm %d %d %d    %d %d", w, h, bottom_blob_bordered.c, elempack, out_elempack);

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

#if __SSE2__
    if (elempack == 8 && out_elempack_int32 == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack8to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd42_pack8to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_3x3_winograd42_data, opt);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack8to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }
        else
        {
            convolution_pack8to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_data_int8, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }

        Mat scale_in_data(num_output);
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
    }

    if (elempack == 1 && out_elempack_int32 == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (opt.use_sgemm_convolution) // TODO better condition && num_input >= 8 && num_output >= 8)
        {
            convolution_im2col_sgemm_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }
        else
        {
            convolution_pack1to4_int8_sse(bottom_blob_bordered, top_blob_int32, weight_data_int8, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }

        Mat scale_in_data(num_output);
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
    }

    if (elempack == 8 && out_elempack_int32 == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack8to1_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_pack8to1_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd42_pack8to1_int8_sse(bottom_blob_bordered, top_blob_int32, weight_3x3_winograd42_data, opt);
        }
        else if (opt.use_sgemm_convolution) // TODO better condition && num_input >= 8 && num_output >= 8)
        {
            convolution_im2col_sgemm_pack8to1_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }
        else
        {
            convolution_pack8to1_int8_sse(bottom_blob_bordered, top_blob_int32, weight_data_int8, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }

        Mat scale_in_data(num_output);
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
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack_int32 == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, opt);
        }
        else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            conv3x3s1_winograd23_int8_sse(bottom_blob_bordered, top_blob_int32, weight_data_3x3_winograd23_int8, opt);
            //             conv3x3s1_winograd43_int8_sse(bottom_blob_bordered, top_blob_int32, weight_data_3x3_winograd23_int8, opt);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_int8_sse(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }
        else
        {
            //         convolution_int8(bottom_blob_bordered, top_blob_int32, weight_data_int8, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
            convolution_int8(bottom_blob_bordered, top_blob_int32, weight_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        }

        Mat scale_in_data(num_output);
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
