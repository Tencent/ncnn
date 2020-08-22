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

#include "pooling_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

#include <float.h>

namespace ncnn {

Pooling_vulkan::Pooling_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    padding = 0;
    pipeline_pooling = 0;
    pipeline_pooling_pack4 = 0;
    pipeline_pooling_pack8 = 0;
    pipeline_pooling_global = 0;
    pipeline_pooling_global_pack4 = 0;
    pipeline_pooling_global_pack8 = 0;
}

int Pooling_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_mode == 0)
        {
            int wtail = (shape.w + pad_left + pad_right - kernel_w) % stride_w;
            int htail = (shape.h + pad_top + pad_bottom - kernel_h) % stride_h;

            int wtailpad = 0;
            int htailpad = 0;
            if (wtail != 0)
                wtailpad = stride_w - wtail;
            if (htail != 0)
                htailpad = stride_h - htail;

            shape_bordered = Mat(shape.w + pad_left + pad_right + wtailpad, shape.h + pad_top + pad_bottom + htailpad, shape.c, (void*)0);
        }
        else if (pad_mode == 1)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c, (void*)0);
        }
        else if (pad_mode == 2 || pad_mode == 3)
        {
            int wpad = kernel_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_h + (shape.h - 1) / stride_h * stride_h - shape.h;
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

    int elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

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
    if (shape_bordered.dims == 1) shape_bordered_packed = Mat(shape_bordered.w / elempack, (void*)0, elemsize, elempack);
    if (shape_bordered.dims == 2) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h / elempack, (void*)0, elemsize, elempack);
    if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h, shape_bordered.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_bordered_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
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

        if (pooling_type == PoolMethod_MAX)
        {
            pd.set(5, -FLT_MAX);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            pd.set(5, 0.f);
        }

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    if (global_pooling)
    {
        std::vector<vk_specialization_type> specializations(1 + 10);
        specializations[0].i = pooling_type;
        specializations[1 + 0].i = shape_bordered_packed.dims;
        specializations[1 + 1].i = shape_bordered_packed.w;
        specializations[1 + 2].i = shape_bordered_packed.h;
        specializations[1 + 3].i = shape_bordered_packed.c;
        specializations[1 + 4].i = shape_bordered_packed.cstep;
        specializations[1 + 5].i = out_shape_packed.dims;
        specializations[1 + 6].i = out_shape_packed.w;
        specializations[1 + 7].i = out_shape_packed.h;
        specializations[1 + 8].i = out_shape_packed.c;
        specializations[1 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz(64, 1, 1, (void*)0);
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, out_shape_packed.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_pooling_global = new Pipeline(vkdev);
            pipeline_pooling_global->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global->create(LayerShaderType::pooling_global, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_pooling_global_pack4 = new Pipeline(vkdev);
            pipeline_pooling_global_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global_pack4->create(LayerShaderType::pooling_global_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_pooling_global_pack8 = new Pipeline(vkdev);
            pipeline_pooling_global_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global_pack8->create(LayerShaderType::pooling_global_pack8, opt, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(12 + 10);
        specializations[0].i = pooling_type;
        specializations[1].i = kernel_w;
        specializations[2].i = kernel_h;
        specializations[3].i = stride_w;
        specializations[4].i = stride_h;
        specializations[5].i = pad_left;
        specializations[6].i = pad_right;
        specializations[7].i = pad_top;
        specializations[8].i = pad_bottom;
        specializations[9].i = global_pooling;
        specializations[10].i = pad_mode;
        specializations[11].i = avgpool_count_include_pad;
        specializations[12 + 0].i = shape_bordered_packed.dims;
        specializations[12 + 1].i = shape_bordered_packed.w;
        specializations[12 + 2].i = shape_bordered_packed.h;
        specializations[12 + 3].i = shape_bordered_packed.c;
        specializations[12 + 4].i = shape_bordered_packed.cstep;
        specializations[12 + 5].i = out_shape_packed.dims;
        specializations[12 + 6].i = out_shape_packed.w;
        specializations[12 + 7].i = out_shape_packed.h;
        specializations[12 + 8].i = out_shape_packed.c;
        specializations[12 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz;
        if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, out_shape_packed.w);
            local_size_xyz.h = std::min(4, out_shape_packed.h);
            local_size_xyz.c = std::min(4, out_shape_packed.c);
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_pooling = new Pipeline(vkdev);
            pipeline_pooling->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling->create(LayerShaderType::pooling, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_pooling_pack4 = new Pipeline(vkdev);
            pipeline_pooling_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_pack4->create(LayerShaderType::pooling_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_pooling_pack8 = new Pipeline(vkdev);
            pipeline_pooling_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_pack8->create(LayerShaderType::pooling_pack8, opt, specializations);
        }
    }

    return 0;
}

int Pooling_vulkan::destroy_pipeline(const Option& _opt)
{
    Option opt = _opt;
    opt.use_image_storage = support_image_storage;

    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_pooling;
    pipeline_pooling = 0;

    delete pipeline_pooling_pack4;
    pipeline_pooling_pack4 = 0;

    delete pipeline_pooling_pack8;
    pipeline_pooling_pack8 = 0;

    delete pipeline_pooling_global;
    pipeline_pooling_global = 0;

    delete pipeline_pooling_global_pack4;
    pipeline_pooling_global_pack4 = 0;

    delete pipeline_pooling_global_pack8;
    pipeline_pooling_global_pack8 = 0;

    return 0;
}

int Pooling_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    return 0;
}

int Pooling_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_pooling_global_pack8
                                   : elempack == 4 ? pipeline_pooling_global_pack4
                                   : pipeline_pooling_global;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    VkMat bottom_blob_bordered = bottom_blob;

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
        int* padding_params = padding_param_blob.mapped();

        padding_params[0] = pad_top;
        padding_params[1] = pad_bottom + htailpad;
        padding_params[2] = pad_left;
        padding_params[3] = pad_right + wtailpad;
        padding_params[4] = 0;
        padding_params[5] = 0;

        std::vector<VkMat> padding_inputs(2);
        padding_inputs[0] = bottom_blob;
        padding_inputs[1] = padding_param_blob;

        std::vector<VkMat> padding_outputs(1);
        padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
        bottom_blob_bordered = padding_outputs[0];
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
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
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
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

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = wtailpad;
    constants[11].i = htailpad;

    const Pipeline* pipeline = elempack == 8 ? pipeline_pooling_pack8
                               : elempack == 4 ? pipeline_pooling_pack4
                               : pipeline_pooling;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Pooling_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = 0; //bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_pooling_global_pack8
                                   : elempack == 4 ? pipeline_pooling_global_pack4
                                   : pipeline_pooling_global;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    VkImageMat bottom_blob_bordered = bottom_blob;

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        VkImageMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
        int* padding_params = padding_param_blob.mapped();

        padding_params[0] = pad_top;
        padding_params[1] = pad_bottom + htailpad;
        padding_params[2] = pad_left;
        padding_params[3] = pad_right + wtailpad;
        padding_params[4] = 0;
        padding_params[5] = 0;

        std::vector<VkImageMat> padding_inputs(2);
        padding_inputs[0] = bottom_blob;
        padding_inputs[1] = padding_param_blob;

        std::vector<VkImageMat> padding_outputs(1);
        padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
        bottom_blob_bordered = padding_outputs[0];
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
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
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
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

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = wtailpad;
    constants[11].i = htailpad;

    const Pipeline* pipeline = elempack == 8 ? pipeline_pooling_pack8
                               : elempack == 4 ? pipeline_pooling_pack4
                               : pipeline_pooling;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
