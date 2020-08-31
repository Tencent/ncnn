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

#include "normalize_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Normalize_vulkan::Normalize_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_normalize_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_normalize_reduce_sum4_fp32[0] = 0;
    pipeline_normalize_reduce_sum4_fp32[1] = 0;
    pipeline_normalize_coeffs = 0;
    pipeline_normalize_norm = 0;

    pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_normalize_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_normalize_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_normalize_coeffs_pack4 = 0;
    pipeline_normalize_norm_pack4 = 0;

    pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8 = 0;
    pipeline_normalize_reduce_sum4_fp32_pack8[0] = 0;
    pipeline_normalize_reduce_sum4_fp32_pack8[1] = 0;
    pipeline_normalize_coeffs_pack8 = 0;
    pipeline_normalize_norm_pack8 = 0;
}

int Normalize_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

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

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;

        Mat local_size_xyz; // TODO select by across_channel / across_spatial

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_normalize_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp16_to_fp32->create(LayerShaderType::normalize_reduce_sum4_fp16_to_fp32, opt, specializations);

            pipeline_normalize_reduce_sum4_fp32[0] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32[0]->create(LayerShaderType::normalize_reduce_sum4_fp32, opt, specializations);
            pipeline_normalize_reduce_sum4_fp32[1] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32[1]->create(LayerShaderType::normalize_reduce_sum4_fp32, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4->create(LayerShaderType::normalize_reduce_sum4_fp16_to_fp32_pack4, opt, specializations);

            pipeline_normalize_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32_pack4[0]->create(LayerShaderType::normalize_reduce_sum4_fp32_pack4, opt, specializations);
            pipeline_normalize_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32_pack4[1]->create(LayerShaderType::normalize_reduce_sum4_fp32_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8 = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8->create(LayerShaderType::normalize_reduce_sum4_fp16_to_fp32_pack8, opt, specializations);

            pipeline_normalize_reduce_sum4_fp32_pack8[0] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32_pack8[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32_pack8[0]->create(LayerShaderType::normalize_reduce_sum4_fp32_pack8, opt, specializations);
            pipeline_normalize_reduce_sum4_fp32_pack8[1] = new Pipeline(vkdev);
            pipeline_normalize_reduce_sum4_fp32_pack8[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_reduce_sum4_fp32_pack8[1]->create(LayerShaderType::normalize_reduce_sum4_fp32_pack8, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;
        specializations[2].f = eps;
        specializations[3].i = eps_mode;

        Mat local_size_xyz; // TODO resolve sqsum_workspace shape

        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_normalize_coeffs = new Pipeline(vkdev);
            pipeline_normalize_coeffs->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_coeffs->create(LayerShaderType::normalize_coeffs, opt, specializations);
        }

        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_normalize_coeffs_pack4 = new Pipeline(vkdev);
            pipeline_normalize_coeffs_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_coeffs_pack4->create(LayerShaderType::normalize_coeffs_pack4, opt, specializations);
        }

        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_normalize_coeffs_pack8 = new Pipeline(vkdev);
            pipeline_normalize_coeffs_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_coeffs_pack8->create(LayerShaderType::normalize_coeffs_pack8, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(5 + 5);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;
        specializations[2].i = channel_shared;
        specializations[3].i = (scale_data_size == 1 && scale_data[0] == 1.f) ? 0 : 1;
        specializations[4].f = channel_shared ? scale_data[0] : 1.f;
        specializations[5 + 0].i = shape_packed.dims;
        specializations[5 + 1].i = shape_packed.w;
        specializations[5 + 2].i = shape_packed.h;
        specializations[5 + 3].i = shape_packed.c;
        specializations[5 + 4].i = shape_packed.cstep;

        Mat local_size_xyz;
        if (shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, shape_packed.w);
            local_size_xyz.h = std::min(4, shape_packed.h);
            local_size_xyz.c = std::min(4, shape_packed.c);
        }

        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_normalize_norm = new Pipeline(vkdev);
            pipeline_normalize_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_norm->create(LayerShaderType::normalize_norm, opt, specializations);
        }

        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_normalize_norm_pack4 = new Pipeline(vkdev);
            pipeline_normalize_norm_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_norm_pack4->create(LayerShaderType::normalize_norm_pack4, opt, specializations);
        }

        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_normalize_norm_pack8 = new Pipeline(vkdev);
            pipeline_normalize_norm_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_norm_pack8->create(LayerShaderType::normalize_norm_pack8, opt, specializations);
        }
    }

    return 0;
}

int Normalize_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_normalize_reduce_sum4_fp16_to_fp32;
    pipeline_normalize_reduce_sum4_fp16_to_fp32 = 0;

    delete pipeline_normalize_reduce_sum4_fp32[0];
    delete pipeline_normalize_reduce_sum4_fp32[1];
    pipeline_normalize_reduce_sum4_fp32[0] = 0;
    pipeline_normalize_reduce_sum4_fp32[1] = 0;

    delete pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4;
    pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4 = 0;

    delete pipeline_normalize_reduce_sum4_fp32_pack4[0];
    delete pipeline_normalize_reduce_sum4_fp32_pack4[1];
    pipeline_normalize_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_normalize_reduce_sum4_fp32_pack4[1] = 0;

    delete pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8;
    pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8 = 0;

    delete pipeline_normalize_reduce_sum4_fp32_pack8[0];
    delete pipeline_normalize_reduce_sum4_fp32_pack8[1];
    pipeline_normalize_reduce_sum4_fp32_pack8[0] = 0;
    pipeline_normalize_reduce_sum4_fp32_pack8[1] = 0;

    delete pipeline_normalize_coeffs;
    pipeline_normalize_coeffs = 0;

    delete pipeline_normalize_coeffs_pack4;
    pipeline_normalize_coeffs_pack4 = 0;

    delete pipeline_normalize_coeffs_pack8;
    pipeline_normalize_coeffs_pack8 = 0;

    delete pipeline_normalize_norm;
    pipeline_normalize_norm = 0;

    delete pipeline_normalize_norm_pack4;
    pipeline_normalize_norm_pack4 = 0;

    delete pipeline_normalize_norm_pack8;
    pipeline_normalize_norm_pack8 = 0;

    return 0;
}

int Normalize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (!channel_shared && !(scale_data_size == 1 && scale_data[0] == 1.f))
    {
        int elempack = opt.use_shader_pack8 && scale_data_size % 8 == 0 ? 8 : scale_data_size % 4 == 0 ? 4 : 1;

        Mat scale_data_packed;
        convert_packing(scale_data, scale_data_packed, elempack);

        if (opt.use_image_storage)
        {
            cmd.record_upload(scale_data_packed, scale_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(scale_data_packed, scale_data_gpu, opt);
        }
    }

    return 0;
}

int Normalize_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    // int w = bottom_top_blob.w;
    // int h = bottom_top_blob.h;
    // int size = w * h;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    // reduce square sum
    VkMat sqsum_workspace;
    {
        {
            int reduced_w;
            int reduced_h;
            int reduced_c;

            if (across_spatial && across_channel)
            {
                reduced_w = (bottom_top_blob.w * bottom_top_blob.h + 1) / 2;
                reduced_h = 1;
                reduced_c = (bottom_top_blob.c + 1) / 2;
            }
            else if (across_spatial && !across_channel)
            {
                reduced_w = (bottom_top_blob.w * bottom_top_blob.h + 3) / 4;
                reduced_h = 1;
                reduced_c = bottom_top_blob.c;
            }
            else // if (!across_spatial && across_channel)
            {
                reduced_w = bottom_top_blob.w * bottom_top_blob.h;
                reduced_h = 1;
                reduced_c = (bottom_top_blob.c + 3) / 4;
            }

            sqsum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
            {
                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_top_blob;
                bindings[1] = sqsum_workspace;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = bottom_top_blob.w * bottom_top_blob.h;
                constants[1].i = 1;
                constants[2].i = bottom_top_blob.c;
                constants[3].i = bottom_top_blob.cstep;
                constants[4].i = sqsum_workspace.w;
                constants[5].i = 1;
                constants[6].i = sqsum_workspace.c;
                constants[7].i = sqsum_workspace.cstep;

                const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8
                                           : elempack == 4 ? pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4
                                           : pipeline_normalize_reduce_sum4_fp16_to_fp32;

                cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
            }
        }

        int pb = 0;
        while ((across_spatial && sqsum_workspace.w > 1) || (across_channel && sqsum_workspace.c > 1))
        {
            int reduced_w;
            int reduced_h;
            int reduced_c;

            if (across_spatial && across_channel)
            {
                reduced_w = (sqsum_workspace.w + 1) / 2;
                reduced_h = 1;
                reduced_c = (sqsum_workspace.c + 1) / 2;
            }
            else if (across_spatial && !across_channel)
            {
                reduced_w = (sqsum_workspace.w + 3) / 4;
                reduced_h = 1;
                reduced_c = sqsum_workspace.c;
            }
            else // if (!across_spatial && across_channel)
            {
                reduced_w = sqsum_workspace.w;
                reduced_h = 1;
                reduced_c = (sqsum_workspace.c + 3) / 4;
            }

            VkMat sqsum_workspace_reduced;
            sqsum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = sqsum_workspace;
                bindings[1] = sqsum_workspace_reduced;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = sqsum_workspace.w;
                constants[1].i = 1;
                constants[2].i = sqsum_workspace.c;
                constants[3].i = sqsum_workspace.cstep;
                constants[4].i = sqsum_workspace_reduced.w;
                constants[5].i = 1;
                constants[6].i = sqsum_workspace_reduced.c;
                constants[7].i = sqsum_workspace_reduced.cstep;

                const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_reduce_sum4_fp32_pack8[pb % 2]
                                           : elempack == 4 ? pipeline_normalize_reduce_sum4_fp32_pack4[pb % 2]
                                           : pipeline_normalize_reduce_sum4_fp32[pb % 2];

                cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

                pb++;
            }

            sqsum_workspace = sqsum_workspace_reduced;
        }
    }

    // coeffs
    VkMat coeffs_workspace;
    coeffs_workspace.create(sqsum_workspace.w * sqsum_workspace.h * sqsum_workspace.c, elemsize, elempack, opt.workspace_vkallocator);
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = sqsum_workspace;
        bindings[1] = coeffs_workspace;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = sqsum_workspace.w;
        constants[1].i = sqsum_workspace.h;
        constants[2].i = sqsum_workspace.c;
        constants[3].i = sqsum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_coeffs_pack8
                                   : elempack == 4 ? pipeline_normalize_coeffs_pack4
                                   : pipeline_normalize_coeffs;

        cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
    }

    // norm
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = coeffs_workspace;
        bindings[2] = scale_data_gpu;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_norm_pack8
                                   : elempack == 4 ? pipeline_normalize_norm_pack4
                                   : pipeline_normalize_norm;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

int Normalize_vulkan::forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    // int w = bottom_top_blob.w;
    // int h = bottom_top_blob.h;
    // int size = w * h;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    // reduce square sum
    VkImageMat sqsum_workspace;
    {
        {
            int reduced_w;
            int reduced_h;
            int reduced_c;

            if (across_spatial && across_channel)
            {
                reduced_w = (bottom_top_blob.w + 1) / 2;
                reduced_h = (bottom_top_blob.h + 1) / 2;
                reduced_c = (bottom_top_blob.c + 1) / 2;
            }
            else if (across_spatial && !across_channel)
            {
                reduced_w = (bottom_top_blob.w + 1) / 2;
                reduced_h = (bottom_top_blob.h + 1) / 2;
                reduced_c = bottom_top_blob.c;
            }
            else // if (!across_spatial && across_channel)
            {
                reduced_w = bottom_top_blob.w;
                reduced_h = bottom_top_blob.h;
                reduced_c = (bottom_top_blob.c + 3) / 4;
            }

            sqsum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
            {
                std::vector<VkImageMat> bindings(2);
                bindings[0] = bottom_top_blob;
                bindings[1] = sqsum_workspace;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = bottom_top_blob.w;
                constants[1].i = bottom_top_blob.h;
                constants[2].i = bottom_top_blob.c;
                constants[3].i = 0; //bottom_top_blob.cstep;
                constants[4].i = sqsum_workspace.w;
                constants[5].i = sqsum_workspace.h;
                constants[6].i = sqsum_workspace.c;
                constants[7].i = 0; //sqsum_workspace.cstep;

                const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_reduce_sum4_fp16_to_fp32_pack8
                                           : elempack == 4 ? pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4
                                           : pipeline_normalize_reduce_sum4_fp16_to_fp32;

                cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
            }
        }

        int pb = 0;
        while ((across_spatial && sqsum_workspace.w * sqsum_workspace.h > 1) || (across_channel && sqsum_workspace.c > 1))
        {
            int reduced_w;
            int reduced_h;
            int reduced_c;

            if (across_spatial && across_channel)
            {
                reduced_w = (sqsum_workspace.w + 1) / 2;
                reduced_h = (sqsum_workspace.h + 1) / 2;
                reduced_c = (sqsum_workspace.c + 1) / 2;
            }
            else if (across_spatial && !across_channel)
            {
                reduced_w = (sqsum_workspace.w + 1) / 2;
                reduced_h = (sqsum_workspace.h + 1) / 2;
                reduced_c = sqsum_workspace.c;
            }
            else // if (!across_spatial && across_channel)
            {
                reduced_w = sqsum_workspace.w;
                reduced_h = sqsum_workspace.h;
                reduced_c = (sqsum_workspace.c + 3) / 4;
            }

            VkImageMat sqsum_workspace_reduced;
            sqsum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);

            {
                std::vector<VkImageMat> bindings(2);
                bindings[0] = sqsum_workspace;
                bindings[1] = sqsum_workspace_reduced;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = sqsum_workspace.w;
                constants[1].i = sqsum_workspace.h;
                constants[2].i = sqsum_workspace.c;
                constants[3].i = 0; //sqsum_workspace.cstep;
                constants[4].i = sqsum_workspace_reduced.w;
                constants[5].i = sqsum_workspace_reduced.h;
                constants[6].i = sqsum_workspace_reduced.c;
                constants[7].i = 0; //sqsum_workspace_reduced.cstep;

                const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_reduce_sum4_fp32_pack8[pb % 2]
                                           : elempack == 4 ? pipeline_normalize_reduce_sum4_fp32_pack4[pb % 2]
                                           : pipeline_normalize_reduce_sum4_fp32[pb % 2];

                cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

                pb++;
            }

            sqsum_workspace = sqsum_workspace_reduced;
        }
    }

    // coeffs
    VkImageMat coeffs_workspace;
    coeffs_workspace.create(sqsum_workspace.w * sqsum_workspace.h * sqsum_workspace.c, elemsize, elempack, opt.workspace_vkallocator);
    {
        std::vector<VkImageMat> bindings(2);
        bindings[0] = sqsum_workspace;
        bindings[1] = coeffs_workspace;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = sqsum_workspace.w;
        constants[1].i = sqsum_workspace.h;
        constants[2].i = sqsum_workspace.c;
        constants[3].i = 0; //sqsum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_coeffs_pack8
                                   : elempack == 4 ? pipeline_normalize_coeffs_pack4
                                   : pipeline_normalize_coeffs;

        cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
    }

    // norm
    {
        std::vector<VkImageMat> bindings(4);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = coeffs_workspace;
        bindings[3] = scale_data_gpu_image;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_normalize_norm_pack8
                                   : elempack == 4 ? pipeline_normalize_norm_pack4
                                   : pipeline_normalize_norm;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
