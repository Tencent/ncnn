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

#include "deconvolutiondepthwise_arm.h"
#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "neon_activation.h"
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise_arm)

DeconvolutionDepthWise_arm::DeconvolutionDepthWise_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    activation = 0;
}

int DeconvolutionDepthWise_arm::create_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

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
        activation->create_pipeline(opt_cpu);
    }

    // create Deconvolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i=0; i<(channels/group)*(num_output/group)*group; i++)
        {
            for (int k=0; k<maxk; k++)
            {
                pt[maxk-1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // depth-wise
    if (channels == group && group == num_output)
    {
        // pack4
        if (num_output % 4 == 0)
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4);
        }
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    for (int i=0; i<(int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g=0; g<group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g);// num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);// pad_w
        pd.set(14, 0);// pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g);// weight_data_size

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = weight_data_g;

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt_cpu);

        group_ops[g] = op;
    }

    return 0;
}

int DeconvolutionDepthWise_arm::destroy_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

    if (activation)
    {
        activation->destroy_pipeline(opt_cpu);
        delete activation;
        activation = 0;
    }

    for (int i=0; i<(int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt_cpu);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int DeconvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

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

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group / elempack; g++)
        {
            float* outptr = top_blob_bordered.channel(g);
            const float* kptr = (const float*)weight_data_pack4 + maxk * g * 4;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                    }

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy < 0 || sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx < 0 || sx >= w)
                                continue;

                            const float* sptr = m.row(sy) + sx * 4;

                            float32x4_t _val = vld1q_f32( sptr );

                            int k = y * kernel_w + x;

                            float32x4_t _w = vld1q_f32( kptr + k * 4 );

                            _sum = vmlaq_f32(_sum, _val, _w);
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    vst1q_f32(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }

        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            Mat top_blob_bordered_adj = top_blob_bordered;
            if (output_pad_right > 0 || output_pad_bottom > 0)
            {
                Option opt_b = opt;
                opt_b.blob_allocator = opt.workspace_allocator;
                copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
                if (top_blob_bordered_adj.empty())
                    return -100;
            }

            copy_cut_border(top_blob_bordered_adj, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
            if (top_blob.empty())
                return -100;

            outw = top_blob.w;
            outh = top_blob.h;
        }
        else if (output_w > 0 && output_h > 0)
        {
            Mat top_blob_bordered_adj = top_blob_bordered;
            if (output_pad_right > 0 || output_pad_bottom > 0)
            {
                Option opt_b = opt;
                opt_b.blob_allocator = opt.workspace_allocator;
                copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
                if (top_blob_bordered_adj.empty())
                    return -100;
            }

            int wcut = top_blob_bordered_adj.w - output_w;
            int hcut = top_blob_bordered_adj.h - output_h;

            if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
            {
                // onnx padding=SAME_UPPER
                copy_cut_border(top_blob_bordered_adj, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
            }
            else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
            {
                // onnx padding=SAME_LOWER
                copy_cut_border(top_blob_bordered_adj, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
            }
            if (top_blob.empty())
                return -100;

            outw = top_blob.w;
            outh = top_blob.h;
        }
        else
        {
            if (output_pad_right > 0 || output_pad_bottom > 0)
            {
                copy_make_border(top_blob_bordered, top_blob, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt);
                if (top_blob.empty())
                    return -100;
            }
            else
            {
                top_blob = top_blob_bordered;
            }
        }

        return 0;
    }
    }

    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    // unpacking
    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack == 4 && channels_g % 4 != 0)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
    }

    Mat top_blob_bordered_unpacked = top_blob_bordered;
    if (num_output_g % 4 != 0 && out_elempack == 4)
    {
        top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
        if (top_blob_bordered_unpacked.empty())
            return -100;
    }

    if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        for (int g=0; g<group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / 4, channels_g / 4);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / 4, num_output_g / 4);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }
    }

    if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        for (int g=0; g<group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g, channels_g);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / 4, num_output_g / 4);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }
    }

    if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        for (int g=0; g<group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / 4, channels_g / 4);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g, num_output_g);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }
    }

    // packing
    if (num_output_g % 4 != 0 && out_elempack == 4)
    {
        convert_packing(top_blob_bordered_unpacked, top_blob_bordered, 4, opt);
    }
    else
    {
        top_blob_bordered = top_blob_bordered_unpacked;
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Mat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        copy_cut_border(top_blob_bordered_adj, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else if (output_w > 0 && output_h > 0)
    {
        Mat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        int wcut = top_blob_bordered_adj.w - output_w;
        int hcut = top_blob_bordered_adj.h - output_h;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border(top_blob_bordered_adj, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border(top_blob_bordered_adj, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
        }
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            copy_make_border(top_blob_bordered, top_blob, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt);
            if (top_blob.empty())
                return -100;
        }
        else
        {
            top_blob = top_blob_bordered;
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;

    } // opt.use_packing_layout
#endif // __ARM_NEON

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_g = bottom_blob.channel_range(g, 1);
            Mat top_blob_bordered_g = top_blob_bordered.channel_range(g, 1);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob_bordered.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }
    }
    else
    {
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            Mat top_blob_bordered_g = top_blob_bordered.channel_range(num_output_g * g, num_output_g);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Mat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        copy_cut_border(top_blob_bordered_adj, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else if (output_w > 0 && output_h > 0)
    {
        Mat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(top_blob_bordered, top_blob_bordered_adj, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt_b);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        int wcut = top_blob_bordered_adj.w - output_w;
        int hcut = top_blob_bordered_adj.h - output_h;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border(top_blob_bordered_adj, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border(top_blob_bordered_adj, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
        }
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            copy_make_border(top_blob_bordered, top_blob, 0, output_pad_bottom, 0, output_pad_right, BORDER_CONSTANT, 0.f, opt);
            if (top_blob.empty())
                return -100;
        }
        else
        {
            top_blob = top_blob_bordered;
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
