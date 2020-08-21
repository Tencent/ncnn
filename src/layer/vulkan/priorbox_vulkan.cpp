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

#include "priorbox_vulkan.h"

#include "layer_shader_type.h"
#include "platform.h"

#include <math.h>

namespace ncnn {

PriorBox_vulkan::PriorBox_vulkan()
{
    support_vulkan = true;

    pipeline_priorbox = 0;
    pipeline_priorbox_mxnet = 0;
}

int PriorBox_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    // caffe style
    {
        int num_min_size = min_sizes.w;
        int num_max_size = max_sizes.w;
        int num_aspect_ratio = aspect_ratios.w;

        int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
        if (flip)
            num_prior += num_min_size * num_aspect_ratio;

        std::vector<vk_specialization_type> specializations(11 + 2);
        specializations[0].i = flip;
        specializations[1].i = clip;
        specializations[2].f = offset;
        specializations[3].f = variances[0];
        specializations[4].f = variances[1];
        specializations[5].f = variances[2];
        specializations[6].f = variances[3];
        specializations[7].i = num_min_size;
        specializations[8].i = num_max_size;
        specializations[9].i = num_aspect_ratio;
        specializations[10].i = num_prior;
        specializations[11 + 0].i = shape_packed.w;
        specializations[11 + 1].i = shape_packed.h;

        pipeline_priorbox = new Pipeline(vkdev);
        pipeline_priorbox->set_optimal_local_size_xyz();
        pipeline_priorbox->create(LayerShaderType::priorbox, opt, specializations);
    }

    // mxnet style
    {
        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        std::vector<vk_specialization_type> specializations(5 + 2);
        specializations[0].i = clip;
        specializations[1].f = offset;
        specializations[2].i = num_sizes;
        specializations[3].i = num_ratios;
        specializations[4].i = num_prior;
        specializations[5 + 0].i = shape_packed.w;
        specializations[5 + 1].i = shape_packed.h;

        pipeline_priorbox_mxnet = new Pipeline(vkdev);
        pipeline_priorbox_mxnet->set_optimal_local_size_xyz();
        pipeline_priorbox_mxnet->create(LayerShaderType::priorbox_mxnet, opt, specializations);
    }

    return 0;
}

int PriorBox_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_priorbox;
    pipeline_priorbox = 0;

    delete pipeline_priorbox_mxnet;
    pipeline_priorbox_mxnet = 0;

    return 0;
}

int PriorBox_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(min_sizes, min_sizes_gpu, opt);

    if (max_sizes.w > 0)
        cmd.record_upload(max_sizes, max_sizes_gpu, opt);

    cmd.record_upload(aspect_ratios, aspect_ratios_gpu, opt);

    return 0;
}

int PriorBox_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    if (bottom_blobs.size() == 1 && image_width == -233 && image_height == -233 && max_sizes.empty())
    {
        // mxnet style _contrib_MultiBoxPrior
        float step_w = step_width;
        float step_h = step_height;
        if (step_w == -233)
            step_w = 1.f / (float)w;
        if (step_h == -233)
            step_h = 1.f / (float)h;

        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        int elempack = 4;

        size_t elemsize = elempack * 4u;
        if (opt.use_fp16_packed || opt.use_fp16_storage)
        {
            elemsize = elempack * 2u;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(4 * w * h * num_prior / elempack, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = min_sizes_gpu;
        bindings[2] = aspect_ratios_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = h;
        constants[2].f = step_w;
        constants[3].f = step_h;

        VkMat dispatcher;
        dispatcher.w = num_sizes;
        dispatcher.h = w;
        dispatcher.c = h;

        cmd.record_pipeline(pipeline_priorbox_mxnet, bindings, constants, dispatcher);

        return 0;
    }

    int image_w = image_width;
    int image_h = image_height;
    if (image_w == -233)
        image_w = bottom_blobs[1].w;
    if (image_h == -233)
        image_h = bottom_blobs[1].h;

    float step_w = step_width;
    float step_h = step_height;
    if (step_w == -233)
        step_w = (float)image_w / w;
    if (step_h == -233)
        step_h = (float)image_h / h;

    int num_min_size = min_sizes.w;
    int num_max_size = max_sizes.w;
    int num_aspect_ratio = aspect_ratios.w;

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    size_t elemsize = 4u;
    if (opt.use_fp16_storage)
    {
        elemsize = 2u;
    }

    VkMat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = top_blob;
    bindings[1] = min_sizes_gpu;
    bindings[2] = num_max_size > 0 ? max_sizes_gpu : min_sizes_gpu;
    bindings[3] = aspect_ratios_gpu;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = w;
    constants[1].i = h;
    constants[2].f = image_w;
    constants[3].f = image_h;
    constants[4].f = step_w;
    constants[5].f = step_h;

    VkMat dispatcher;
    dispatcher.w = num_min_size;
    dispatcher.h = w;
    dispatcher.c = h;

    cmd.record_pipeline(pipeline_priorbox, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
