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

#include "deconvolution_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Deconvolution_vulkan::Deconvolution_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    crop = 0;
    output_pad = 0;
    output_crop = 0;

    pipeline_deconvolution = 0;
    pipeline_deconvolution_pack4 = 0;
    pipeline_deconvolution_pack1to4 = 0;
    pipeline_deconvolution_pack4to1 = 0;
    pipeline_deconvolution_pack8 = 0;
    pipeline_deconvolution_pack1to8 = 0;
    pipeline_deconvolution_pack4to8 = 0;
    pipeline_deconvolution_pack8to4 = 0;
    pipeline_deconvolution_pack8to1 = 0;
}

int Deconvolution_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape before unpadding
    Mat out_shape_bordered;
    // the shape after output adj
    Mat out_shape_bordered_adj;
    if (shape.dims != 0)
    {
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        int outw = (shape.w - 1) * stride_w + kernel_extent_w;
        int outh = (shape.h - 1) * stride_h + kernel_extent_h;

        out_shape_bordered = Mat(outw, outh, out_shape.c, (void*)0);

        out_shape_bordered_adj = Mat(outw + output_pad_right, outh + output_pad_bottom, out_shape.c, (void*)0);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
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

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_bordered_packed;
    if (out_shape_bordered.dims == 1) out_shape_bordered_packed = Mat(out_shape_bordered.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_bordered.dims == 2) out_shape_bordered_packed = Mat(out_shape_bordered.w, out_shape_bordered.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_bordered.dims == 3) out_shape_bordered_packed = Mat(out_shape_bordered.w, out_shape_bordered.h, out_shape_bordered.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_bordered_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    // check weight shape
    Mat weight_data_packed(maxk, num_input / elempack, num_output / out_elempack, (void*)0, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
    if (!vkdev->shape_support_image_storage(weight_data_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    {
        crop = ncnn::create_layer(ncnn::LayerType::Crop);
        crop->vkdev = vkdev;

        crop->bottom_shapes.resize(1);
        crop->bottom_shapes[0] = out_shape_bordered_adj;
        crop->top_shapes.resize(1);
        crop->top_shapes[0] = out_shape;

        ncnn::ParamDict pd;
        pd.set(0, pad_left);
        pd.set(1, pad_top);
        pd.set(2, 0);

        crop->load_param(pd);

        crop->create_pipeline(opt);
    }

    {
        output_pad = ncnn::create_layer(ncnn::LayerType::Padding);
        output_pad->vkdev = vkdev;

        output_pad->bottom_shapes.resize(1);
        output_pad->bottom_shapes[0] = out_shape_bordered;
        output_pad->top_shapes.resize(1);
        output_pad->top_shapes[0] = out_shape_bordered_adj;

        ncnn::ParamDict pd;
        pd.set(0, 0);
        pd.set(1, output_pad_bottom);
        pd.set(2, 0);
        pd.set(3, output_pad_right);
        pd.set(4, 0);
        pd.set(5, 0.f);

        output_pad->load_param(pd);

        output_pad->create_pipeline(opt);
    }

    {
        output_crop = ncnn::create_layer(ncnn::LayerType::Crop);
        output_crop->vkdev = vkdev;

        output_crop->bottom_shapes.resize(1);
        output_crop->bottom_shapes[0] = out_shape_bordered_adj;
        output_crop->top_shapes.resize(1);
        output_crop->top_shapes[0] = out_shape;

        ncnn::ParamDict pd;
        pd.set(0, -233);
        pd.set(1, -233);
        pd.set(2, -233);

        output_crop->load_param(pd);

        output_crop->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(10 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = activation_type;
    specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[10 + 0].i = shape_packed.dims;
    specializations[10 + 1].i = shape_packed.w;
    specializations[10 + 2].i = shape_packed.h;
    specializations[10 + 3].i = shape_packed.c;
    specializations[10 + 4].i = shape_packed.cstep;
    specializations[10 + 5].i = out_shape_bordered_packed.dims;
    specializations[10 + 6].i = out_shape_bordered_packed.w;
    specializations[10 + 7].i = out_shape_bordered_packed.h;
    specializations[10 + 8].i = out_shape_bordered_packed.c;
    specializations[10 + 9].i = out_shape_bordered_packed.cstep;

    Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack), (void*)0);
    if (out_shape_bordered_packed.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_bordered_packed.w);
        local_size_xyz.h = std::min(8, out_shape_bordered_packed.h);
        local_size_xyz.c = std::min(4, out_shape_bordered_packed.c);
    }

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline_deconvolution = new Pipeline(vkdev);
        pipeline_deconvolution->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution->create(LayerShaderType::deconvolution, opt, specializations);
    }

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        pipeline_deconvolution_pack4 = new Pipeline(vkdev);
        pipeline_deconvolution_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack4->create(LayerShaderType::deconvolution_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        pipeline_deconvolution_pack1to4 = new Pipeline(vkdev);
        pipeline_deconvolution_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack1to4->create(LayerShaderType::deconvolution_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        pipeline_deconvolution_pack4to1 = new Pipeline(vkdev);
        pipeline_deconvolution_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack4to1->create(LayerShaderType::deconvolution_pack4to1, opt, specializations);
    }

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        pipeline_deconvolution_pack8 = new Pipeline(vkdev);
        pipeline_deconvolution_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack8->create(LayerShaderType::deconvolution_pack8, opt, specializations);
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        pipeline_deconvolution_pack1to8 = new Pipeline(vkdev);
        pipeline_deconvolution_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack1to8->create(LayerShaderType::deconvolution_pack1to8, opt, specializations);
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        pipeline_deconvolution_pack4to8 = new Pipeline(vkdev);
        pipeline_deconvolution_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack4to8->create(LayerShaderType::deconvolution_pack4to8, opt, specializations);
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        pipeline_deconvolution_pack8to4 = new Pipeline(vkdev);
        pipeline_deconvolution_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack8to4->create(LayerShaderType::deconvolution_pack8to4, opt, specializations);
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        pipeline_deconvolution_pack8to1 = new Pipeline(vkdev);
        pipeline_deconvolution_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deconvolution_pack8to1->create(LayerShaderType::deconvolution_pack8to1, opt, specializations);
    }

    return 0;
}

int Deconvolution_vulkan::destroy_pipeline(const Option& opt)
{
    if (crop)
    {
        crop->destroy_pipeline(opt);
        delete crop;
        crop = 0;
    }

    if (output_pad)
    {
        output_pad->destroy_pipeline(opt);
        delete output_pad;
        output_pad = 0;
    }

    if (output_crop)
    {
        output_crop->destroy_pipeline(opt);
        delete output_crop;
        output_crop = 0;
    }

    delete pipeline_deconvolution;
    pipeline_deconvolution = 0;

    delete pipeline_deconvolution_pack4;
    pipeline_deconvolution_pack4 = 0;

    delete pipeline_deconvolution_pack1to4;
    pipeline_deconvolution_pack1to4 = 0;

    delete pipeline_deconvolution_pack4to1;
    pipeline_deconvolution_pack4to1 = 0;

    delete pipeline_deconvolution_pack8;
    pipeline_deconvolution_pack8 = 0;

    delete pipeline_deconvolution_pack1to8;
    pipeline_deconvolution_pack1to8 = 0;

    delete pipeline_deconvolution_pack4to8;
    pipeline_deconvolution_pack4to8 = 0;

    delete pipeline_deconvolution_pack8to4;
    pipeline_deconvolution_pack8to4 = 0;

    delete pipeline_deconvolution_pack8to1;
    pipeline_deconvolution_pack8to1 = 0;

    return 0;
}

int Deconvolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (crop)
    {
        crop->upload_model(cmd, opt);
    }

    if (output_pad)
    {
        output_pad->upload_model(cmd, opt);
    }

    if (output_crop)
    {
        output_crop->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

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
    // dst = pa-pb-kw-kh-inch/pa-outch/pb
    Mat weight_data_packed;
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            Mat g0 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                float* g00 = g0.row(p / elempack);

                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        const Mat k0 = weight_data_r2.channel(q + i);

                        for (int j = 0; j < elempack; j++)
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

    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

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

int Deconvolution_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    VkMat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
    }
    else
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob_bordered;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob_bordered.dims;
    constants[6].i = top_blob_bordered.w;
    constants[7].i = top_blob_bordered.h;
    constants[8].i = top_blob_bordered.c;
    constants[9].i = top_blob_bordered.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution_pack4to1;
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob_bordered);

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        VkMat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;
            output_pad->forward(top_blob_bordered, top_blob_bordered_adj, cmd, opt_pad);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        {
            VkMat reference_blob;
            reference_blob.dims = 2;
            reference_blob.w = top_blob_bordered_adj.w - pad_left - pad_right;
            reference_blob.h = top_blob_bordered_adj.h - pad_top - pad_bottom;
            reference_blob.elempack = 1;

            std::vector<VkMat> crop_bottom_blobs(2);
            crop_bottom_blobs[0] = top_blob_bordered_adj;
            crop_bottom_blobs[1] = reference_blob;
            std::vector<VkMat> crop_top_blobs(1);
            crop->forward(crop_bottom_blobs, crop_top_blobs, cmd, opt);
            top_blob = crop_top_blobs[0];
        }
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else if (output_w > 0 && output_h > 0)
    {
        VkMat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;
            output_pad->forward(top_blob_bordered, top_blob_bordered_adj, cmd, opt_pad);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        int wcut = top_blob_bordered_adj.w - output_w;
        int hcut = top_blob_bordered_adj.h - output_h;

        VkMat crop_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator);
        int* crop_params = crop_param_blob.mapped();

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            crop_params[0] = wcut / 2;
            crop_params[1] = hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered_adj.w - wcut;
            crop_params[4] = top_blob_bordered_adj.h - hcut;
            crop_params[5] = top_blob_bordered_adj.c * out_elempack;
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            crop_params[0] = wcut - wcut / 2;
            crop_params[1] = hcut - hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered_adj.w - wcut;
            crop_params[4] = top_blob_bordered_adj.h - hcut;
            crop_params[5] = top_blob_bordered_adj.c * out_elempack;
        }

        std::vector<VkMat> crop_inputs(2);
        crop_inputs[0] = top_blob_bordered_adj;
        crop_inputs[1] = crop_param_blob;

        std::vector<VkMat> crop_outputs(1);
        output_crop->forward(crop_inputs, crop_outputs, cmd, opt);
        top_blob = crop_outputs[0];
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            output_pad->forward(top_blob_bordered, top_blob, cmd, opt);
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

int Deconvolution_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    VkImageMat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
    }
    else
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    std::vector<VkImageMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob_bordered;
    bindings[2] = weight_data_gpu_image;
    bindings[3] = bias_data_gpu_image;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = 0; //bottom_blob.cstep;
    constants[5].i = top_blob_bordered.dims;
    constants[6].i = top_blob_bordered.w;
    constants[7].i = top_blob_bordered.h;
    constants[8].i = top_blob_bordered.c;
    constants[9].i = 0; //top_blob_bordered.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution_pack4to1;
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_deconvolution_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_deconvolution_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_deconvolution_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob_bordered);

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        VkImageMat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;
            output_pad->forward(top_blob_bordered, top_blob_bordered_adj, cmd, opt_pad);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        {
            VkImageMat reference_blob;
            reference_blob.dims = 2;
            reference_blob.w = top_blob_bordered_adj.w - pad_left - pad_right;
            reference_blob.h = top_blob_bordered_adj.h - pad_top - pad_bottom;
            reference_blob.elempack = 1;

            std::vector<VkImageMat> crop_bottom_blobs(2);
            crop_bottom_blobs[0] = top_blob_bordered_adj;
            crop_bottom_blobs[1] = reference_blob;
            std::vector<VkImageMat> crop_top_blobs(1);
            crop->forward(crop_bottom_blobs, crop_top_blobs, cmd, opt);
            top_blob = crop_top_blobs[0];
        }
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else if (output_w > 0 && output_h > 0)
    {
        VkImageMat top_blob_bordered_adj = top_blob_bordered;
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;
            output_pad->forward(top_blob_bordered, top_blob_bordered_adj, cmd, opt_pad);
            if (top_blob_bordered_adj.empty())
                return -100;
        }

        int wcut = top_blob_bordered_adj.w - output_w;
        int hcut = top_blob_bordered_adj.h - output_h;

        VkImageMat crop_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator);
        int* crop_params = crop_param_blob.mapped();

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            crop_params[0] = wcut / 2;
            crop_params[1] = hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered_adj.w - wcut;
            crop_params[4] = top_blob_bordered_adj.h - hcut;
            crop_params[5] = top_blob_bordered_adj.c * out_elempack;
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            crop_params[0] = wcut - wcut / 2;
            crop_params[1] = hcut - hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered_adj.w - wcut;
            crop_params[4] = top_blob_bordered_adj.h - hcut;
            crop_params[5] = top_blob_bordered_adj.c * out_elempack;
        }

        std::vector<VkImageMat> crop_inputs(2);
        crop_inputs[0] = top_blob_bordered_adj;
        crop_inputs[1] = crop_param_blob;

        std::vector<VkImageMat> crop_outputs(1);
        output_crop->forward(crop_inputs, crop_outputs, cmd, opt);
        top_blob = crop_outputs[0];
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        if (output_pad_right > 0 || output_pad_bottom > 0)
        {
            output_pad->forward(top_blob_bordered, top_blob, cmd, opt);
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

} // namespace ncnn
