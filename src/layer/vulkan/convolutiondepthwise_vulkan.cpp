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

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

ConvolutionDepthWise_vulkan::ConvolutionDepthWise_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    padding = 0;

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

int ConvolutionDepthWise_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c, (void*)0);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h + hpad, shape.c, (void*)0);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_bordered_packed;
    if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h, shape_bordered.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    int elempack_g = opt.use_shader_pack8 && channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
    int out_elempack_g = opt.use_shader_pack8 && num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;

    size_t elemsize_g;
    size_t out_elemsize_g;
    if (opt.use_fp16_storage)
    {
        elemsize_g = elempack_g * 2u;
        out_elemsize_g = out_elempack_g * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize_g = elempack_g == 1 ? 4u : elempack_g * 2u;
        out_elemsize_g = out_elempack_g == 1 ? 4u : out_elempack_g * 2u;
    }
    else
    {
        elemsize_g = elempack_g * 4u;
        out_elemsize_g = out_elempack_g * 4u;
    }

    Mat shape_bordered_g_packed;
    if (shape_bordered.dims == 3) shape_bordered_g_packed = Mat(shape_bordered.w, shape_bordered.h, shape_bordered.c / elempack_g, (void*)0, elemsize_g, elempack_g);

    Mat out_shape_g_packed;
    if (out_shape.dims == 3) out_shape_g_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack_g, (void*)0, out_elemsize_g, out_elempack_g);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_bordered_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    // check weight shape
    if (channels == group && group == num_output)
    {
        Mat weight_data_packed(maxk, group / elempack, (void*)0, (size_t)4 * elempack, elempack);
        if (!vkdev->shape_support_image_storage(weight_data_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }
    }
    else
    {
        // check blob shape
        if (!vkdev->shape_support_image_storage(shape_bordered_g_packed) || !vkdev->shape_support_image_storage(out_shape_g_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }

        Mat weight_data_packed_groups(maxk, channels_g / elempack_g, num_output_g / out_elempack_g * group, (size_t)4 * elempack_g * out_elempack_g, elempack_g * out_elempack_g);
        if (!vkdev->shape_support_image_storage(weight_data_packed_groups))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }
    }

    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

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

    std::vector<vk_specialization_type> specializations(11 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = group;
    specializations[8].i = activation_type;
    specializations[9].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[10].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    // depth-wise
    if (channels == group && group == num_output)
    {
        specializations[11 + 0].i = shape_bordered_packed.dims;
        specializations[11 + 1].i = shape_bordered_packed.w;
        specializations[11 + 2].i = shape_bordered_packed.h;
        specializations[11 + 3].i = shape_bordered_packed.c;
        specializations[11 + 4].i = shape_bordered_packed.cstep;
        specializations[11 + 5].i = out_shape_packed.dims;
        specializations[11 + 6].i = out_shape_packed.w;
        specializations[11 + 7].i = out_shape_packed.h;
        specializations[11 + 8].i = out_shape_packed.c;
        specializations[11 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack), (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape_packed.w);
            local_size_xyz.h = std::min(8, out_shape_packed.h);
            local_size_xyz.c = std::min(4, out_shape_packed.c);
        }

        // pack1
        if (elempack == 1)
        {
            pipeline_convolutiondepthwise = new Pipeline(vkdev);
            pipeline_convolutiondepthwise->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise->create(LayerShaderType::convolutiondepthwise, opt, specializations);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise_pack4->create(LayerShaderType::convolutiondepthwise_pack4, opt, specializations);
        }

        // pack8
        if (elempack == 8)
        {
            pipeline_convolutiondepthwise_pack8 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolutiondepthwise_pack8->create(LayerShaderType::convolutiondepthwise_pack8, opt, specializations);
        }

        return 0;
    }

    specializations[11 + 0].i = shape_bordered_g_packed.dims;
    specializations[11 + 1].i = shape_bordered_g_packed.w;
    specializations[11 + 2].i = shape_bordered_g_packed.h;
    specializations[11 + 3].i = shape_bordered_g_packed.c;
    specializations[11 + 4].i = shape_bordered_g_packed.cstep;
    specializations[11 + 5].i = out_shape_g_packed.dims;
    specializations[11 + 6].i = out_shape_g_packed.w;
    specializations[11 + 7].i = out_shape_g_packed.h;
    specializations[11 + 8].i = out_shape_g_packed.c;
    specializations[11 + 9].i = out_shape_g_packed.cstep;

    Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack_g), (void*)0);
    if (out_shape_g_packed.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_g_packed.w);
        local_size_xyz.h = std::min(8, out_shape_g_packed.h);
        local_size_xyz.c = std::min(4, out_shape_g_packed.c);
    }

    // pack1
    if (elempack_g == 1 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group->create(LayerShaderType::convolutiondepthwise_group, opt, specializations);
    }

    // pack4
    if (elempack_g == 4 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack4->create(LayerShaderType::convolutiondepthwise_group_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack_g == 1 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack1to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack1to4->create(LayerShaderType::convolutiondepthwise_group_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack_g == 4 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group_pack4to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack4to1->create(LayerShaderType::convolutiondepthwise_group_pack4to1, opt, specializations);
    }

    // pack8
    if (elempack_g == 8 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack8->create(LayerShaderType::convolutiondepthwise_group_pack8, opt, specializations);
    }

    // pack1to8
    if (elempack_g == 1 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack1to8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack1to8->create(LayerShaderType::convolutiondepthwise_group_pack1to8, opt, specializations);
    }

    // pack4to8
    if (elempack_g == 4 && out_elempack_g == 8)
    {
        pipeline_convolutiondepthwise_group_pack4to8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack4to8->create(LayerShaderType::convolutiondepthwise_group_pack4to8, opt, specializations);
    }

    // pack8to4
    if (elempack_g == 8 && out_elempack_g == 4)
    {
        pipeline_convolutiondepthwise_group_pack8to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack8to4->create(LayerShaderType::convolutiondepthwise_group_pack8to4, opt, specializations);
    }

    // pack8to1
    if (elempack_g == 8 && out_elempack_g == 1)
    {
        pipeline_convolutiondepthwise_group_pack8to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_group_pack8to1->create(LayerShaderType::convolutiondepthwise_group_pack8to1, opt, specializations);
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
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    // depth-wise
    if (channels == group && group == num_output)
    {
        Mat weight_data_packed;
        Mat weight_data_r2 = weight_data.reshape(maxk, group);
        convert_packing(weight_data_r2, weight_data_packed, elempack);

        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);

        if (bias_term)
        {
            Mat bias_data_packed;
            convert_packing(bias_data, bias_data_packed, out_elempack);

            if (support_image_storage && opt.use_image_storage)
            {
                cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
            }
            else
            {
                cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
            }
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

        weight_data_packed_groups.create(maxk, channels_g / elempack_g, num_output_g / out_elempack_g * group, (size_t)4 * elempack_g * out_elempack_g, elempack_g * out_elempack_g);

        for (int g = 0; g < group; g++)
        {
            const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

            Mat weight_data_packed = weight_data_packed_groups.channel_range(num_output_g / out_elempack_g * g, num_output_g / out_elempack_g);

            for (int q = 0; q + (out_elempack_g - 1) < num_output_g; q += out_elempack_g)
            {
                Mat g0 = weight_data_packed.channel(q / out_elempack_g);

                for (int p = 0; p + (elempack_g - 1) < channels_g; p += elempack_g)
                {
                    float* g00 = g0.row(p / elempack_g);

                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack_g; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q + i);

                            for (int j = 0; j < elempack_g; j++)
                            {
                                const float* k00 = k0.row(p + j);

                                g00[0] = k00[k];

                                g00++;
                            }
                        }
                    }
                }
            }
        }
    }

    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed_groups, weight_data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(weight_data_packed_groups, weight_data_gpu, opt);
    }

    if (bias_term)
    {
        Mat bias_data_packed;
        convert_packing(bias_data, bias_data_packed, out_elempack_g);

        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }
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

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

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

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

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
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

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
    size_t out_elemsize_g = elemsize / elempack * out_elempack_g;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack_g == 8) out_elemsize_g = 8 * 2u;
        if (out_elempack_g == 4) out_elemsize_g = 4 * 2u;
        if (out_elempack_g == 1) out_elemsize_g = 4u;
    }

    // unpacking
    VkMat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > elempack_g)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, elempack_g, cmd, opt_pack1);
    }

    VkMat top_blob_unpacked = top_blob;
    if (out_elempack_g < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_elempack_g, out_elemsize_g, out_elempack_g, opt.workspace_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered_unpacked;
    bindings[1] = top_blob_unpacked;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

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
        vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkImageMat bottom_blob_bordered = bottom_blob;
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

            VkImageMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkImageMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkImageMat> padding_outputs(1);
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

            VkImageMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkImageMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkImageMat> padding_outputs(1);
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
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
        std::vector<VkImageMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = weight_data_gpu_image;
        bindings[3] = bias_data_gpu_image;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = 0; //bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;

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
    size_t out_elemsize_g = elemsize / elempack * out_elempack_g;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack_g == 8) out_elemsize_g = 8 * 2u;
        if (out_elempack_g == 4) out_elemsize_g = 4 * 2u;
        if (out_elempack_g == 1) out_elemsize_g = 4u;
    }

    // unpacking
    VkImageMat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > elempack_g)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, elempack_g, cmd, opt_pack1);
    }

    VkImageMat top_blob_unpacked = top_blob;
    if (out_elempack_g < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_elempack_g, out_elemsize_g, out_elempack_g, opt.workspace_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkImageMat> bindings(4);
    bindings[0] = bottom_blob_bordered_unpacked;
    bindings[1] = top_blob_unpacked;
    bindings[2] = weight_data_gpu_image;
    bindings[3] = bias_data_gpu_image;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered_unpacked.dims;
    constants[1].i = bottom_blob_bordered_unpacked.w;
    constants[2].i = bottom_blob_bordered_unpacked.h;
    constants[3].i = bottom_blob_bordered_unpacked.c;
    constants[4].i = 0; //bottom_blob_bordered_unpacked.cstep;
    constants[5].i = top_blob_unpacked.dims;
    constants[6].i = top_blob_unpacked.w;
    constants[7].i = top_blob_unpacked.h;
    constants[8].i = top_blob_unpacked.c;
    constants[9].i = 0; //top_blob_unpacked.cstep;

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
        vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

} // namespace ncnn
