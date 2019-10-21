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

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "neon_activation.h"
#endif // __ARM_NEON

namespace ncnn {

#include "convolution_1x1.h"
#include "convolution_2x2.h"
#include "convolution_3x3.h"
#include "convolution_4x4.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"
#include "convolution_sgemm.h"
#include "convolution_sgemm_int8.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3_int8.h"
#include "convolution_5x5_int8.h"
#include "convolution_7x7_int8.h"

#if __ARM_NEON
#include "convolution_1x1_pack4.h"
#include "convolution_1x1_pack4to1.h"
#include "convolution_3x3_pack4.h"
#include "convolution_3x3_pack1to4.h"
#include "convolution_3x3_pack4to1.h"
#include "convolution_5x5_pack4.h"
#include "convolution_7x7_pack1to4.h"
#endif // __ARM_NEON

DEFINE_LAYER_CREATOR(Convolution_arm)

Convolution_arm::Convolution_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    activation = 0;
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
        pd.set(0, activation_params[0]);// slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]);// min
        pd.set(1, activation_params[1]);// max
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
        Option opt_cpu = opt;
        opt_cpu.use_vulkan_compute = false;
        activation->create_pipeline(opt_cpu);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-4a-kw-kh-inch/4a-outch/4b
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4.create(maxk, num_input/4, num_output/4, (size_t)4*16, 16);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack4.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p+1);
                    const float* k12 = k1.row(p+2);
                    const float* k13 = k1.row(p+3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p+1);
                    const float* k22 = k2.row(p+2);
                    const float* k23 = k2.row(p+3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p+1);
                    const float* k32 = k3.row(p+2);
                    const float* k33 = k3.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
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

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
        }
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack1to4.create(maxk, num_input, num_output/4, (size_t)4*4, 4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack1to4.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    float* g00 = g0.row(p);

                    for (int k=0; k<maxk; k++)
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
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4to1.create(maxk, num_input/4, num_output, (size_t)4*4, 4);

            for (int q=0; q<num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
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

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
        }

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
        }
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    use_winograd3x3 = false;
    use_sgemm1x1 = false;

    if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        // winograd is slow on small channel count
        if (num_input >= 16 && num_output >= 16)
            use_winograd3x3 = true;

        if (use_int8_inference)
            use_winograd3x3 = true;
    }

    // TODO assume more proper condition
    if (opt.use_sgemm_convolution && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if (num_input >= 64 && num_output >= 64)
            use_sgemm1x1 = true;
    }

    if (use_int8_inference)
    {
        if (use_winograd3x3)
        {
            // conv3x3s1_winograd23_transform_kernel_int8_neon(weight_data, weight_3x3_winograd23_int8_data, num_input, num_output);
            conv3x3s1_winograd43_transform_kernel_int8_neon(weight_data, weight_3x3_winograd23_int8_data, num_input, num_output);
        }

        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_transform_kernel_int8_neon(weight_data, weight_3x3s2_int8_data, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_int8_neon(weight_data, weight_1x1s1_sgemm_int8_data, num_input, num_output);
            use_sgemm1x1 = true;
        }
        else
        {
            conv_im2col_sgemm_transform_kernel_int8_neon(weight_data, weight_sgemm_int8_data, num_input, num_output, maxk);
        }

        return 0;
    }

    if (impl_type > 0)
    {
        switch(impl_type)
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
            case 4:
                // direct
                break;
            case 5:
                // conv3x3s2
                conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, num_input, num_output);
                break;
            default:
                return -1;
        }
        return 0;
    }

    if (use_winograd3x3)
    {
//         conv3x3s1_winograd64_transform_kernel_neon(weight_data, weight_3x3_winograd64_data, num_input, num_output);
        conv3x3s1_winograd64_transform_kernel_neon5(weight_data, weight_3x3_winograd64_data, num_input, num_output);
    }

    if (use_sgemm1x1)
    {
        conv1x1s1_sgemm_transform_kernel_neon(weight_data, weight_1x1_sgemm_data, num_input, num_output);
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, num_input, num_output);
    }

    {
        conv_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, maxk);
    }    

    return 0;
}

int Convolution_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        Option opt_cpu = opt;
        opt_cpu.use_vulkan_compute = false;
        activation->destroy_pipeline(opt_cpu);
        delete activation;
        activation = 0;
    }

    return 0;
}

int Convolution_arm::forwardDilation(const Mat& bottom_blob, Mat& top_blob, conv_func conv, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    Mat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent + (w - 1) / stride * stride - w;
        int hpad = kernel_extent + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent + (w - 1) / stride * stride - w;
        int hpad = kernel_extent + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Make (dilation * dilation) batches
    Mat inner_bottom_blob;
    Mat inner_top_blob;
    for (int x = 0; x < dilation; x ++)
    {
        for (int y = 0; y < dilation; y ++)
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
            for (int c = 0; c < bottom_blob.c; c ++)
            {
                float *outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i ++)
                {
                    const float *ptr = (const float *) bottom_blob_bordered.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j ++)
                    {
                        outptr[j] = ptr[j*dilation];
                    }
                    outptr += inner_w;
                }
            }

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            conv(inner_bottom_blob, inner_top_blob, weight_data, bias_data, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < num_output; c ++)
            {
                float *outptr = (float *) top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i ++)
                {
                    const float *ptr = (const float *) inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j ++)
                    {
                        outptr[j*dilation] = ptr[j];
                    }
                    outptr += dilation * outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

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

    // float32
    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s1_winograd64_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 5 && kernel_h == 5 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 5 && kernel_h == 5 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv5x5s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output / out_elempack; p++)
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
                    for (int q=0; q<channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                        for (int k = 0; k < maxk; k++) // 29.23
                        {
                            float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );

                            float32x4_t _w0 = vld1q_f32( kptr );
                            float32x4_t _w1 = vld1q_f32( kptr + 4 );
                            float32x4_t _w2 = vld1q_f32( kptr + 8 );
                            float32x4_t _w3 = vld1q_f32( kptr + 12 );

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

        return 0;
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s1_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 7 && kernel_h == 7 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv7x7s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output / out_elempack; p++)
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
                    for (int q=0; q<channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        const float* sptr = m.row(i*stride_h) + j*stride_w;

                        for (int k = 0; k < maxk; k++) // 29.23
                        {
                            float32x4_t _val = vdupq_n_f32( sptr[ space_ofs[k] ] );
                            float32x4_t _w = vld1q_f32( kptr );
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

        return 0;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s1_sgemm_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            conv1x1s2_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            conv3x3s1_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
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
                    for (int q=0; q<channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                        for (int k = 0; k < maxk; k++) // 29.23
                        {
                            float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                            float32x4_t _w = vld1q_f32( kptr );
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

        return 0;
    }

    } // opt.use_packed_layout
#endif // __ARM_NEON

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    //const int stride = stride_w;
    int stride = stride_w;

    if (kernel_size > 7 || stride > 4 || dilation_w != dilation_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    typedef void (*conv_func)(const Mat&, Mat&, const Mat&, const Mat&, const Option&);

    // kernel_size x stride
    conv_func conv_func_table[7][4] =
    {
        {
            conv1x1s1_neon,
            conv1x1s2_neon,
            0,
            0
        }, // kernel_size = 1
        {
            conv2x2s1_neon,
            0,
            0,
            0
        }, // kernel_size = 2
        {
            conv3x3s1_neon,
            conv3x3s2_neon,
            0,
            0
        }, // kernel_size = 3
        {
            0,
            0,
            0,
            conv4x4s4_neon
        }, // kernel_size = 4
        {
            conv5x5s1_neon,
            conv5x5s2_neon,
            0,
            0
        }, // kernel_size = 5
        {
            0,
            0,
            0,
            0
        }, // kernel_size = 6
        {
            conv7x7s1_neon,
            conv7x7s2_neon,
            0,
            0
        }  // kernel_size = 7
    };

    typedef void (*conv_int8_func)(const Mat&, Mat&, const Mat&, const Option&);

    // kernel_size x stride
    conv_int8_func conv_int8_func_table[7][4] =
    {
        {
            conv1x1s1_int8_neon,
            conv1x1s2_int8_neon,
            0,
            0
        }, // kernel_size = 1
        {
            0,
            0,
            0,
            0
        }, // kernel_size = 2
        {
            conv3x3s1_int8_neon,
            conv3x3s2_int8_neon,
            0,
            0
        }, // kernel_size = 3
        {
            0,
            0,
            0,
            0
        }, // kernel_size = 4
        {
            conv5x5s1_int8_neon,
            conv5x5s2_int8_neon,
            0,
            0
        }, // kernel_size = 5
        {
            0,
            0,
            0,
            0
        }, // kernel_size = 6
        {            
            conv7x7s1_int8_neon,           
            conv7x7s2_int8_neon,
            0,
            0
        }  // kernel_size = 7                
    };

    conv_func conv = 0;
    conv_int8_func conv_int8 = 0;

    if (use_int8_inference)
    {
        conv_int8 = conv_int8_func_table[kernel_size-1][stride-1];
        if (!conv_int8)
        {
            return Convolution::forward(bottom_blob, top_blob, opt);
        }
    }
    else
    {
        conv = conv_func_table[kernel_size-1][stride-1];
        if (!conv)
        {
            return Convolution::forward(bottom_blob, top_blob, opt);
        }

        if (dilation_w != 1)
        {
            if (stride != 1)
                return Convolution::forward(bottom_blob, top_blob, opt);

            return forwardDilation(bottom_blob, top_blob, conv, opt);
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;             
    }

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_size + (w - 1) / stride * stride - w;
        int hpad = kernel_size + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_size + (w - 1) / stride * stride - w;
        int hpad = kernel_size + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    // int8
    if (use_int8_inference)
    {
        if (use_int8_requantize == true)
        {
            Mat top_blob_tm;
            top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
            if (top_blob_tm.empty())
                return -100;
            
            top_blob.create(outw, outh, num_output, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100; 

            if (use_sgemm1x1)
            {              
                conv1x1s1_sgemm_int8_requant_neon(bottom_blob_bordered, top_blob, weight_1x1s1_sgemm_int8_data, bias_data, requantize_scales, opt);
                
                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }  

                return 0;
            }
            else if (use_winograd3x3)
            {
                // conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_int8_data, opt);
                conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_int8_data, opt);
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob_tm, weight_3x3s2_int8_data, opt);
            }
            else
            {
                conv_int8(bottom_blob_bordered, top_blob_tm, weight_sgemm_int8_data, opt);     
            }

            // requantize, reverse scale inplace
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                ncnn::Option opt_g = opt;
                opt_g.num_threads = 1;
                opt_g.blob_allocator = top_blob.allocator;

                Mat top_blob_tm_g = top_blob_tm.channel_range(p, 1);
                Mat top_blob_g = top_blob.channel_range(p, 1);
                requantize_ops[p]->forward(top_blob_tm_g, top_blob_g, opt_g);
            }                     
        }
        else
        {
            top_blob.create(outw, outh, num_output, (size_t)4u, opt.blob_allocator);
            if (top_blob.empty())
                return -100; 

            if (use_sgemm1x1)
            {
                conv1x1s1_sgemm_int8_neon(bottom_blob_bordered, top_blob, weight_1x1s1_sgemm_int8_data, opt);
            }
            else if (use_winograd3x3)
            {
                // conv3x3s1_winograd23_int8_neon(bottom_blob_bordered, top_blob, weight_3x3_winograd23_int8_data, opt);
                // conv3x3s1_winograd43_int8_neon(bottom_blob_bordered, top_blob, weight_3x3_winograd23_int8_data, opt);
                conv3x3s1_winograd43_dequant_int8_neon(bottom_blob_bordered, top_blob, weight_3x3_winograd23_int8_data, bias_data, dequantize_scales, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }  

                return 0;
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                conv3x3s2_packed_int8_neon(bottom_blob_bordered, top_blob, weight_3x3s2_int8_data, opt);
            }
            else
            {
                conv_int8(bottom_blob_bordered, top_blob, weight_sgemm_int8_data, opt);
            }        

            // dequantize, reverse scale inplace
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                ncnn::Option opt_g = opt;
                opt_g.num_threads = 1;
                opt_g.blob_allocator = top_blob.allocator;

                Mat top_blob_g = top_blob.channel_range(p, 1);
                dequantize_ops[p]->forward_inplace(top_blob_g, opt_g);
            }          
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }           

        return 0;
    }

    // float32
    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (impl_type > 0)
    {
        // engineering is magic.
        switch(impl_type)
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
            case 4:
                conv(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
                break;
            case 5:
                conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data, bias_data, opt);
                break;
            default:
                return -1;
        }

    } else 
    {
        if (use_winograd3x3 && w <= 120 && h <= 120)
        {
//             conv3x3s1_winograd64_neon4(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
            conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data, opt);
        }
        else if (use_sgemm1x1)
        {
            conv1x1s1_sgemm_neon(bottom_blob_bordered, top_blob, weight_1x1_sgemm_data, bias_data, opt);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            if (outw >=8 && outh >=8)
                conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data, bias_data, opt);
            else
                conv_im2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
        }     
        else
            conv(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
    }


    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
