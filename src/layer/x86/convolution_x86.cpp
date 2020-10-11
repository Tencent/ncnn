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

#if __AVX__
#include "avx_activation.h"
#include "avx_usability.h"
#endif

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#include "convolution_x86.h"

#include "benchmark.h"
#include "layer_type.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_sgemm_int8.h"
#if __AVX__
#include "convolution_3x3_pack1to8.h"
#include "convolution_3x3_pack8to1.h"
#include "convolution_3x3_pack8.h"
#include "convolution_2x2_pack8.h"
#include "convolution_2x2_pack8_fp16.h"
#include "convolution_1x1_pack8.h"
#include "convolution_1x1_pack8_fp16.h"
#endif

#include "convolution_1x1.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3.h"
#include "convolution_3x3_int8.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"

Convolution_x86::Convolution_x86()
{
#ifdef __AVX__
    support_packing = true;
    support_weight_fp16_storage = true;
#endif
    activation = 0;
    convolution_dilation1 = 0;
}

int Convolution_x86::create_pipeline(const Option& opt)
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
        if (use_int8_requantize)
        {
            pd.set(0, activation_params[0] * top_blob_int8_scale); // min
            pd.set(1, activation_params[1] * top_blob_int8_scale); // max
        }
        else
        {
            pd.set(0, activation_params[0]); // min
            pd.set(1, activation_params[1]); // max
        }

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

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        support_packing = false;
        return create_pipeline_int8_x86(opt);
    }

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    use_winograd3x3 = false;

    if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
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

    int elempack = (support_packing && opt.use_packing_layout && num_input % 8 == 0) ? 8 : 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 8 == 0) ? 8 : 1;

#if __AVX__
    const int maxk = kernel_w * kernel_h;

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (opt.use_weight_fp16_storage && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_fp16_pack8_avx(weight_data, weight_data_pack8, num_input, num_output);
        }
        else if (opt.use_weight_fp16_storage && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_fp16_pack8_avx(weight_data, weight_data_pack8, num_input, num_output);
        }
        else if (opt.use_weight_fp16_storage && kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv2x2s1_weight_fp16_pack8_avx(weight_data, weight_data_pack8, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack8_avx(weight_data, weight_data_pack8, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack8_avx(weight_data, weight_data_pack8, num_input, num_output);
        }
        else
        {
            // src = kw-kh-inch-outch
            // dst = 8b-8a-kw-kh-inch/8a-outch/8b
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack8.create(maxk, num_input / 8, num_output / 8, (size_t)4 * 64, 64);

            for (int q = 0; q + 7 < num_output; q += 8)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);
                const Mat k4 = weight_data_r2.channel(q + 4);
                const Mat k5 = weight_data_r2.channel(q + 5);
                const Mat k6 = weight_data_r2.channel(q + 6);
                const Mat k7 = weight_data_r2.channel(q + 7);

                Mat g0 = weight_data_pack8.channel(q / 8);

                for (int p = 0; p + 7 < num_input; p += 8)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p + 1);
                    const float* k02 = k0.row(p + 2);
                    const float* k03 = k0.row(p + 3);
                    const float* k04 = k0.row(p + 4);
                    const float* k05 = k0.row(p + 5);
                    const float* k06 = k0.row(p + 6);
                    const float* k07 = k0.row(p + 7);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p + 1);
                    const float* k12 = k1.row(p + 2);
                    const float* k13 = k1.row(p + 3);
                    const float* k14 = k1.row(p + 4);
                    const float* k15 = k1.row(p + 5);
                    const float* k16 = k1.row(p + 6);
                    const float* k17 = k1.row(p + 7);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p + 1);
                    const float* k22 = k2.row(p + 2);
                    const float* k23 = k2.row(p + 3);
                    const float* k24 = k2.row(p + 4);
                    const float* k25 = k2.row(p + 5);
                    const float* k26 = k2.row(p + 6);
                    const float* k27 = k2.row(p + 7);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p + 1);
                    const float* k32 = k3.row(p + 2);
                    const float* k33 = k3.row(p + 3);
                    const float* k34 = k3.row(p + 4);
                    const float* k35 = k3.row(p + 5);
                    const float* k36 = k3.row(p + 6);
                    const float* k37 = k3.row(p + 7);

                    const float* k40 = k4.row(p);
                    const float* k41 = k4.row(p + 1);
                    const float* k42 = k4.row(p + 2);
                    const float* k43 = k4.row(p + 3);
                    const float* k44 = k4.row(p + 4);
                    const float* k45 = k4.row(p + 5);
                    const float* k46 = k4.row(p + 6);
                    const float* k47 = k4.row(p + 7);

                    const float* k50 = k5.row(p);
                    const float* k51 = k5.row(p + 1);
                    const float* k52 = k5.row(p + 2);
                    const float* k53 = k5.row(p + 3);
                    const float* k54 = k5.row(p + 4);
                    const float* k55 = k5.row(p + 5);
                    const float* k56 = k5.row(p + 6);
                    const float* k57 = k5.row(p + 7);

                    const float* k60 = k6.row(p);
                    const float* k61 = k6.row(p + 1);
                    const float* k62 = k6.row(p + 2);
                    const float* k63 = k6.row(p + 3);
                    const float* k64 = k6.row(p + 4);
                    const float* k65 = k6.row(p + 5);
                    const float* k66 = k6.row(p + 6);
                    const float* k67 = k6.row(p + 7);

                    const float* k70 = k7.row(p);
                    const float* k71 = k7.row(p + 1);
                    const float* k72 = k7.row(p + 2);
                    const float* k73 = k7.row(p + 3);
                    const float* k74 = k7.row(p + 4);
                    const float* k75 = k7.row(p + 5);
                    const float* k76 = k7.row(p + 6);
                    const float* k77 = k7.row(p + 7);

                    float* g00 = g0.row(p / 8);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];
                        g00[4] = k40[k];
                        g00[5] = k50[k];
                        g00[6] = k60[k];
                        g00[7] = k70[k];
                        g00 += 8;
                        g00[0] = k01[k];
                        g00[1] = k11[k];
                        g00[2] = k21[k];
                        g00[3] = k31[k];
                        g00[4] = k41[k];
                        g00[5] = k51[k];
                        g00[6] = k61[k];
                        g00[7] = k71[k];

                        g00 += 8;
                        g00[0] = k02[k];
                        g00[1] = k12[k];
                        g00[2] = k22[k];
                        g00[3] = k32[k];
                        g00[4] = k42[k];
                        g00[5] = k52[k];
                        g00[6] = k62[k];
                        g00[7] = k72[k];

                        g00 += 8;
                        g00[0] = k03[k];
                        g00[1] = k13[k];
                        g00[2] = k23[k];
                        g00[3] = k33[k];
                        g00[4] = k43[k];
                        g00[5] = k53[k];
                        g00[6] = k63[k];
                        g00[7] = k73[k];

                        g00 += 8;
                        g00[0] = k04[k];
                        g00[1] = k14[k];
                        g00[2] = k24[k];
                        g00[3] = k34[k];
                        g00[4] = k44[k];
                        g00[5] = k54[k];
                        g00[6] = k64[k];
                        g00[7] = k74[k];

                        g00 += 8;
                        g00[0] = k05[k];
                        g00[1] = k15[k];
                        g00[2] = k25[k];
                        g00[3] = k35[k];
                        g00[4] = k45[k];
                        g00[5] = k55[k];
                        g00[6] = k65[k];
                        g00[7] = k75[k];

                        g00 += 8;
                        g00[0] = k06[k];
                        g00[1] = k16[k];
                        g00[2] = k26[k];
                        g00[3] = k36[k];
                        g00[4] = k46[k];
                        g00[5] = k56[k];
                        g00[6] = k66[k];
                        g00[7] = k76[k];

                        g00 += 8;
                        g00[0] = k07[k];
                        g00[1] = k17[k];
                        g00[2] = k27[k];
                        g00[3] = k37[k];
                        g00[4] = k47[k];
                        g00[5] = k57[k];
                        g00[6] = k67[k];
                        g00[7] = k77[k];

                        g00 += 8;
                    }
                }
            }
        }
    }
    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        // src = kw-kh-inch-outch
        // dst = 8b-kw-kh-inch-outch/8
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack1to8.create(maxk, num_input, num_output / 8, (size_t)4 * 8, 8);

            for (int q = 0; q + 7 < num_output; q += 8)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q + 1);
                const Mat k2 = weight_data_r2.channel(q + 2);
                const Mat k3 = weight_data_r2.channel(q + 3);
                const Mat k4 = weight_data_r2.channel(q + 4);
                const Mat k5 = weight_data_r2.channel(q + 5);
                const Mat k6 = weight_data_r2.channel(q + 6);
                const Mat k7 = weight_data_r2.channel(q + 7);

                Mat g0 = weight_data_pack1to8.channel(q / 8);

                for (int p = 0; p < num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);
                    const float* k40 = k4.row(p);
                    const float* k50 = k5.row(p);
                    const float* k60 = k6.row(p);
                    const float* k70 = k7.row(p);

                    float* g00 = g0.row(p);

                    for (int k = 0; k < maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];
                        g00[4] = k40[k];
                        g00[5] = k50[k];
                        g00[6] = k60[k];
                        g00[7] = k70[k];

                        g00 += 8;
                    }
                }
            }
        }
    }
    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_pack8to1.create(maxk, num_input / 8, num_output, (size_t)4 * 8, 8);

        for (int q = 0; q < num_output; q++)
        {
            const Mat k0 = weight_data_r2.channel(q);
            Mat g0 = weight_data_pack8to1.channel(q);

            for (int p = 0; p + 7 < num_input; p += 8)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p + 1);
                const float* k02 = k0.row(p + 2);
                const float* k03 = k0.row(p + 3);
                const float* k04 = k0.row(p + 4);
                const float* k05 = k0.row(p + 5);
                const float* k06 = k0.row(p + 6);
                const float* k07 = k0.row(p + 7);

                float* g00 = g0.row(p / 8);

                for (int k = 0; k < maxk; k++)
                {
                    g00[0] = k00[k];
                    g00[1] = k01[k];
                    g00[2] = k02[k];
                    g00[3] = k03[k];
                    g00[4] = k04[k];
                    g00[5] = k05[k];
                    g00[6] = k06[k];
                    g00[7] = k07[k];

                    g00 += 8;
                }
            }
        }
    }
#endif

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            // winograd is slow on small channel count
            use_winograd3x3 = true;

            conv3x3s1_winograd23_transform_kernel_sse(weight_data, weight_3x3_winograd23_data, num_input, num_output);
            // conv3x3s1_winograd43_transform_kernel_sse(weight_data, weight_3x3_winograd43_data, num_input, num_output);

            // for small size
            conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
        }
        else
        {
            conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
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

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if ((!support_packing || !opt.use_packing_layout) && (dilation_w > 1 || dilation_h > 1) && (stride_w > 1 || stride_h > 1))
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if ((!support_packing || !opt.use_packing_layout) && (dilation_w > 1 || dilation_h > 1) && dilation_w != dilation_h)
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
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 8 == 0) ? 8 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        if (outw >= dilation_w && outh >= dilation_h)
        {
            return forwardDilation_x86(bottom_blob_bordered, top_blob, opt);
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
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv1x1s1_sgemm_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }
            else
            {
                conv1x1s1_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv1x1s2_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }
            else
            {
                conv1x1s2_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }
            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }

        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv2x2s1_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }
            else
            {
                conv2x2s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_pack8, bias_data, opt);
            }

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
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                        }

                        const float* kptr = (const float*)weight_data_pack8 + maxk * channels * p * 64;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m256 _val0 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8));
                                __m256 _val1 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 1);
                                __m256 _val2 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 2);
                                __m256 _val3 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 3);
                                __m256 _val4 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 4);
                                __m256 _val5 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 5);
                                __m256 _val6 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 6);
                                __m256 _val7 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 7);

                                __m256 _w0 = _mm256_loadu_ps(kptr);
                                _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
                                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                                _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
                                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                                _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
                                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                                _sum = _mm256_fmadd_ps(_val3, _w3, _sum);
                                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                                _sum = _mm256_fmadd_ps(_val4, _w4, _sum);
                                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                                _sum = _mm256_fmadd_ps(_val5, _w5, _sum);
                                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                                _sum = _mm256_fmadd_ps(_val6, _w6, _sum);
                                __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                                _sum = _mm256_fmadd_ps(_val7, _w7, _sum);
                                kptr += 64;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr + j * 8, _sum);
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
            conv3x3s1_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_pack1to8, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_pack1to8, bias_data, opt);

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
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                        }

                        const float* kptr = (const float*)weight_data_pack1to8 + maxk * channels * p * 8;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                __m256 _val = _mm256_set1_ps(sptr[space_ofs[k]]);
                                __m256 _w = _mm256_loadu_ps(kptr);
                                _sum = _mm256_fmadd_ps(_val, _w, _sum);

                                kptr += 8;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }
    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_pack8to1, bias_data, opt);

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

                        const float* kptr = (const float*)weight_data_pack8to1 + maxk * channels * p * 8;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                __m256 _val = _mm256_loadu_ps(sptr + (space_ofs[k] * 8));
                                __m256 _w = _mm256_loadu_ps(kptr);
                                __m256 _s8 = _mm256_mul_ps(_val, _w);
                                sum += _mm256_reduce_add_ps(_s8); // dot
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
#endif

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (use_winograd3x3 && outw >= 8 && outh >= 8)
            {
                conv3x3s1_winograd23_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data, bias_data, opt);
                //             conv3x3s1_winograd43_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd43_data, bias_data, opt);
            }
            else
            {
                conv_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (dilation_w == 1 && dilation_h == 1)
        {
            conv_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
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

int Convolution_x86::create_pipeline_int8_x86(const Option& opt)
{
    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    use_winograd3x3_int8 = false;

    if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1
            && num_input >= 16 && num_output >= 16)
    {
        // winograd is slow on small channel count
        use_winograd3x3_int8 = true;

        conv3x3s1_winograd23_transform_kernel_int8_sse(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
        //         conv3x3s1_winograd43_transform_kernel_int8_sse(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
    }
    else
    {
        // TODO offline transform weight
    }

    return 0;
}

int Convolution_x86::forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dilation_w > 1 || dilation_h > 1)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

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
    if (use_int8_requantize)
    {
        Mat top_blob_tm;
        top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
        if (top_blob_tm.empty())
            return -100;

        if (use_winograd3x3_int8)
        {
            conv3x3s1_winograd23_int8_sse(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);
            //             conv3x3s1_winograd43_int8_sse(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);

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

            conv_im2col_sgemm_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, kernel_w, kernel_h, stride_w, stride_h, bias_data, requantize_scales, opt);
        }
    }
    else
    {
        if (use_winograd3x3_int8)
        {
            conv3x3s1_winograd23_int8_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);
            //             conv3x3s1_winograd43_int8_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);

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
        else
        {
            std::vector<float> dequantize_scales;
            for (int p = 0; p < num_output; p++)
            {
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                dequantize_scales.push_back(scale_in);
            }

            conv_im2col_sgemm_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, kernel_w, kernel_h, stride_w, stride_h, bias_data, dequantize_scales, opt);
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

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
