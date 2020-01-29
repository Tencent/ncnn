// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolutiondepthwise_vulkan.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ConvolutionDepthWise_vulkan)

ConvolutionDepthWise_vulkan::ConvolutionDepthWise_vulkan()
{
    support_vulkan = true;

    padding = 0;
    packing_pack1 = 0;
    packing_pack4 = 0;
    packing_pack8 = 0;

    pipeline_convolutiondepthwise = 0;
    pipeline_convolutiondepthwise_pack4 = 0;
    pipeline_convolutiondepthwise_pack8 = 0;

    pipeline_convolutiondepthwise_group = 0;
    pipeline_convolutiondepthwise_group_pack4 = 0;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;
    pipeline_convolutiondepthwise_group_pack8 = 0;
    pipeline_convolutiondepthwise_group_pack1to8 = 0;
    pipeline_convolutiondepthwise_group_pack4to8 = 0;
    pipeline_convolutiondepthwise_group_pack8to4 = 0;
    pipeline_convolutiondepthwise_group_pack8to1 = 0;
}

int ConvolutionDepthWise_vulkan::create_pipeline(const Option& opt)
{
    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(11);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = group;
    specializations[8].i = activation_type;
    specializations[9].f = activation_params.w == 1 ? activation_params[0] : 0.f;
    specializations[10].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    // depth-wise
    if (channels == group && group == num_output)
    {
        // pack1
        if (elempack == 1)
        {
            pipeline_convolutiondepthwise = new Pipeline(vkdev);
            pipeline_convolutiondepthwise->set_optimal_local_size_xyz(32, 32, num_output);
            pipeline_convolutiondepthwise->create("convolutiondepthwise", opt, specializations, 4, 10);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 4));
            pipeline_convolutiondepthwise_pack4->create("convolutiondepthwise_pack4", opt, specializations, 4, 10);
        }

        // pack8
        if (elempack == 8)
        {
            pipeline_convolutiondepthwise_pack8 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack8->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
            pipeline_convolutiondepthwise_pack8->create("convolutiondepthwise_pack8", opt, specializations, 4, 10);
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    int elempack_g = opt.use_shader_pack8 && channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = opt.use_shader_pack8 && num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;

    // pack1
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group->create("convolutiondepthwise_group", opt, specializations, 4, 10);
    }

    // pack4
    if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack4->create("convolutiondepthwise_group_pack4", opt, specializations, 4, 10);
    }

    // pack1to4
    if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack1to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack1to4->create("convolutiondepthwise_group_pack1to4", opt, specializations, 4, 10);
    }

    // pack4to1
    if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group_pack4to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to1->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack4to1->create("convolutiondepthwise_group_pack4to1", opt, specializations, 4, 10);
    }

    // pack8
    if (elempack_g == 8 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack8->create("convolutiondepthwise_group_pack8", opt, specializations, 4, 10);
    }

    // pack1to8
    if (elempack_g == 1 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack1to8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to8->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack1to8->create("convolutiondepthwise_group_pack1to8", opt, specializations, 4, 10);
    }

    // pack4to8
    if (elempack_g == 4 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack4to8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to8->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack4to8->create("convolutiondepthwise_group_pack4to8", opt, specializations, 4, 10);
    }

    // pack8to4
    if (elempack_g == 8 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack8to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8to4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack8to4->create("convolutiondepthwise_group_pack8to4", opt, specializations, 4, 10);
    }

    // pack8to1
    if (elempack_g == 8 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group_pack8to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8to1->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack8to1->create("convolutiondepthwise_group_pack8to1", opt, specializations, 4, 10);
    }

    if (elempack > elempack_g && elempack_g == 1)
    {
        packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack1->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 1);

        packing_pack1->load_param(pd);

        packing_pack1->create_pipeline(opt);
    }

    if ((elempack > elempack_g && elempack_g == 4)
        || (out_elempack_g < out_elempack && out_elempack == 4))
    {
        packing_pack4 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack4->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 4);

        packing_pack4->load_param(pd);

        packing_pack4->create_pipeline(opt);
    }

    if (out_elempack_g < out_elempack && out_elempack == 8)
    {
        packing_pack8 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack8->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 8);

        packing_pack8->load_param(pd);

        packing_pack8->create_pipeline(opt);
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    if (packing_pack1)
    {
        packing_pack1->destroy_pipeline(opt);
        delete packing_pack1;
        packing_pack1 = 0;
    }

    if (packing_pack4)
    {
        packing_pack4->destroy_pipeline(opt);
        delete packing_pack4;
        packing_pack4 = 0;
    }

    if (packing_pack8)
    {
        packing_pack8->destroy_pipeline(opt);
        delete packing_pack8;
        packing_pack8 = 0;
    }

    delete pipeline_convolutiondepthwise;
    pipeline_convolutiondepthwise = 0;

    delete pipeline_convolutiondepthwise_pack4;
    pipeline_convolutiondepthwise_pack4 = 0;

    delete pipeline_convolutiondepthwise_pack8;
    pipeline_convolutiondepthwise_pack8 = 0;

    delete pipeline_convolutiondepthwise_group;
    pipeline_convolutiondepthwise_group = 0;

    delete pipeline_convolutiondepthwise_group_pack4;
    pipeline_convolutiondepthwise_group_pack4 = 0;

    delete pipeline_convolutiondepthwise_group_pack1to4;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;

    delete pipeline_convolutiondepthwise_group_pack4to1;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;

    delete pipeline_convolutiondepthwise_group_pack8;
    pipeline_convolutiondepthwise_group_pack8 = 0;

    delete pipeline_convolutiondepthwise_group_pack1to8;
    pipeline_convolutiondepthwise_group_pack1to8 = 0;

    delete pipeline_convolutiondepthwise_group_pack4to8;
    pipeline_convolutiondepthwise_group_pack4to8 = 0;

    delete pipeline_convolutiondepthwise_group_pack8to4;
    pipeline_convolutiondepthwise_group_pack8to4 = 0;

    delete pipeline_convolutiondepthwise_group_pack8to1;
    pipeline_convolutiondepthwise_group_pack8to1 = 0;

    return 0;
}

int ConvolutionDepthWise_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
//     int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    // depth-wise
    if (channels == group && group == num_output)
    {
        Mat weight_data_packed;
        Mat weight_data_r2 = weight_data.reshape(maxk, group);
        convert_packing(weight_data_r2, weight_data_packed, elempack);

        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

        if (bias_term)
        {
            cmd.record_upload(bias_data, bias_data_gpu, opt);
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    int elempack_g = opt.use_shader_pack8 && channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = opt.use_shader_pack8 && num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb
    Mat weight_data_packed_groups;
    {
        Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

        weight_data_packed_groups.create(maxk, channels_g/elempack_g, num_output_g/out_elempack_g * group, (size_t)4*elempack_g*out_elempack_g, elempack_g*out_elempack_g);

        for (int g=0; g<group; g++)
        {
            const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

            Mat weight_data_packed = weight_data_packed_groups.channel_range(num_output_g/out_elempack_g * g, num_output_g/out_elempack_g);

            for (int q=0; q+(out_elempack_g-1)<num_output_g; q+=out_elempack_g)
            {
                Mat g0 = weight_data_packed.channel(q/out_elempack_g);

                for (int p=0; p+(elempack_g-1)<channels_g; p+=elempack_g)
                {
                    float* g00 = g0.row(p/elempack_g);

                    for (int k=0; k<maxk; k++)
                    {

                        for (int i=0; i<out_elempack_g; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q+i);

                            for (int j=0; j<elempack_g; j++)
                            {
                                const float* k00 = k0.row(p+j);

                                g00[0] = k00[k];

                                g00++;
                            }
                        }

                    }
                }
            }
        }
    }

    cmd.record_upload(weight_data_packed_groups, weight_data_gpu, opt);

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            padding_param_blob.prepare_staging_buffer();
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            padding_param_blob.prepare_staging_buffer();
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8*2u;
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_convolutiondepthwise_pack8
                                 : elempack == 4 ? pipeline_convolutiondepthwise_pack4
                                 : pipeline_convolutiondepthwise;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int elempack_g = opt.use_shader_pack8 && channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = opt.use_shader_pack8 && num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;

    // unpacking
    VkMat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > elempack_g)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        const Layer* packing = elempack_g == 4 ? packing_pack4 : packing_pack1;
        packing->forward(bottom_blob_bordered, bottom_blob_bordered_unpacked, cmd, opt_pack1);
    }

    VkMat top_blob_unpacked = top_blob;
    if (out_elempack_g < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_elempack_g, out_elemsize / out_elempack * out_elempack_g, out_elempack_g, opt.workspace_vkallocator, opt.staging_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered_unpacked;
    bindings[1] = top_blob_unpacked;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered_unpacked.dims;
    constants[1].i = bottom_blob_bordered_unpacked.w;
    constants[2].i = bottom_blob_bordered_unpacked.h;
    constants[3].i = bottom_blob_bordered_unpacked.c;
    constants[4].i = bottom_blob_bordered_unpacked.cstep;
    constants[5].i = top_blob_unpacked.dims;
    constants[6].i = top_blob_unpacked.w;
    constants[7].i = top_blob_unpacked.h;
    constants[8].i = top_blob_unpacked.c;
    constants[9].i = top_blob_unpacked.cstep;

    const Pipeline* pipeline = 0;
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise_group;
    }
    else if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4;
    }
    else if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack1to4;
    }
    else if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4to1;
    }
    else if (elempack_g == 8 && out_elempack_g == 8)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack8;
    }
    else if (elempack_g == 1 && out_elempack_g == 8)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack1to8;
    }
    else if (elempack_g == 4 && out_elempack_g == 8)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4to8;
    }
    else if (elempack_g == 8 && out_elempack_g == 4)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack8to4;
    }
    else if (elempack_g == 8 && out_elempack_g == 1)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob_unpacked);

    // packing
    if (out_elempack_g < out_elempack)
    {
        const Layer* packing = out_elempack == 8 ? packing_pack8 : packing_pack4;
        packing->forward(top_blob_unpacked, top_blob, cmd, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

} // namespace ncnn
