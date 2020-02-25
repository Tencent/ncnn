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

#include "instancenorm_vulkan.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(InstanceNorm_vulkan)

InstanceNorm_vulkan::InstanceNorm_vulkan()
{
    support_vulkan = true;

    pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_instancenorm_reduce_sum4_fp32[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32[1] = 0;
    pipeline_instancenorm_reduce_mean = 0;
    pipeline_instancenorm_sub_mean_square = 0;
    pipeline_instancenorm_coeffs = 0;
    pipeline_instancenorm_norm = 0;

    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_instancenorm_reduce_mean_pack4 = 0;
    pipeline_instancenorm_sub_mean_square_pack4 = 0;
    pipeline_instancenorm_coeffs_pack4 = 0;
    pipeline_instancenorm_norm_pack4 = 0;

    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8 = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack8[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack8[1] = 0;
    pipeline_instancenorm_reduce_mean_pack8 = 0;
    pipeline_instancenorm_sub_mean_square_pack8 = 0;
    pipeline_instancenorm_coeffs_pack8 = 0;
    pipeline_instancenorm_norm_pack8 = 0;
}

int InstanceNorm_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;

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
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, channels / elempack, (void*)0, elemsize, elempack);

    // TODO resolve workspace_shape.w
    Mat workspace_shape_packed(1, 1, channels / elempack, (void*)0, elemsize, elempack);

    {
        Mat local_size_xyz(16, 1, std::min(4, channels / elempack), (void*)0);
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = 16;
            local_size_xyz.h = 1;
            local_size_xyz.c = std::min(4, workspace_shape_packed.c);
        }

        // pack1
        if (elempack == 1)
        {
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32->create("instancenorm_reduce_sum4_fp16_to_fp32", opt, std::vector<vk_specialization_type>(), 2, 6);

            pipeline_instancenorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32[0]->create("instancenorm_reduce_sum4_fp32", opt, std::vector<vk_specialization_type>(), 2, 6);
            pipeline_instancenorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32[1]->create("instancenorm_reduce_sum4_fp32", opt, std::vector<vk_specialization_type>(), 2, 6);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->create("instancenorm_reduce_sum4_fp16_to_fp32_pack4", opt, std::vector<vk_specialization_type>(), 2, 6);

            pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->create("instancenorm_reduce_sum4_fp32_pack4", opt, std::vector<vk_specialization_type>(), 2, 6);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->create("instancenorm_reduce_sum4_fp32_pack4", opt, std::vector<vk_specialization_type>(), 2, 6);
        }

        // pack8
        if (elempack == 8)
        {
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8->create("instancenorm_reduce_sum4_fp16_to_fp32_pack8", opt, std::vector<vk_specialization_type>(), 2, 6);

            pipeline_instancenorm_reduce_sum4_fp32_pack8[0] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack8[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack8[0]->create("instancenorm_reduce_sum4_fp32_pack8", opt, std::vector<vk_specialization_type>(), 2, 6);
            pipeline_instancenorm_reduce_sum4_fp32_pack8[1] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack8[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack8[1]->create("instancenorm_reduce_sum4_fp32_pack8", opt, std::vector<vk_specialization_type>(), 2, 6);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(0 + 3);
        specializations[0].i = 0;// TODO resolve workspace_shape_packed.w;
        specializations[1].i = workspace_shape_packed.c;
        specializations[2].i = 0;// TODO resolve workspace_shape_packed.cstep;

        Mat local_size_xyz(std::min(64, channels / elempack), 1, 1, (void*)0);
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.c);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        if (elempack == 1)
        {
            pipeline_instancenorm_reduce_mean = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_mean->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_mean->create("instancenorm_reduce_mean", opt, specializations, 2, 4);
        }

        if (elempack == 4)
        {
            pipeline_instancenorm_reduce_mean_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_mean_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_mean_pack4->create("instancenorm_reduce_mean_pack4", opt, specializations, 2, 4);
        }

        if (elempack == 8)
        {
            pipeline_instancenorm_reduce_mean_pack8 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_mean_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_mean_pack8->create("instancenorm_reduce_mean_pack8", opt, specializations, 2, 4);
        }
    }

    Mat square_workspace_packed;
    if (shape.dims == 3) square_workspace_packed = Mat(shape.w, shape.h, channels / elempack, (void*)0, elempack * 4u, elempack);

    {
        std::vector<vk_specialization_type> specializations(0 + 10);
        specializations[0 + 0].i = shape_packed.dims;
        specializations[0 + 1].i = shape_packed.w;
        specializations[0 + 2].i = shape_packed.h;
        specializations[0 + 3].i = shape_packed.c;
        specializations[0 + 4].i = shape_packed.cstep;
        specializations[0 + 5].i = square_workspace_packed.dims;
        specializations[0 + 6].i = square_workspace_packed.w;
        specializations[0 + 7].i = square_workspace_packed.h;
        specializations[0 + 8].i = square_workspace_packed.c;
        specializations[0 + 9].i = square_workspace_packed.cstep;

        Mat local_size_xyz(4, 4, std::min(4, channels / elempack), (void*)0);
        if (square_workspace_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, square_workspace_packed.w);
            local_size_xyz.h = std::min(4, square_workspace_packed.h);
            local_size_xyz.c = std::min(4, square_workspace_packed.c);
        }

        if (elempack == 1)
        {
            pipeline_instancenorm_sub_mean_square = new Pipeline(vkdev);
            pipeline_instancenorm_sub_mean_square->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_sub_mean_square->create("instancenorm_sub_mean_square", opt, specializations, 3, 10);
        }

        if (elempack == 4)
        {
            pipeline_instancenorm_sub_mean_square_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_sub_mean_square_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_sub_mean_square_pack4->create("instancenorm_sub_mean_square_pack4", opt, specializations, 3, 10);
        }

        if (elempack == 8)
        {
            pipeline_instancenorm_sub_mean_square_pack8 = new Pipeline(vkdev);
            pipeline_instancenorm_sub_mean_square_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_sub_mean_square_pack8->create("instancenorm_sub_mean_square_pack8", opt, specializations, 3, 10);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].f = eps;
        specializations[1].i = channels / elempack;

        Mat local_size_xyz(std::min(64, channels / elempack), 1, 1, (void*)0);
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.c);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        if (elempack == 1)
        {
            pipeline_instancenorm_coeffs = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs->create("instancenorm_coeffs", opt, specializations, 5, 0);
        }

        if (elempack == 4)
        {
            pipeline_instancenorm_coeffs_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs_pack4->create("instancenorm_coeffs_pack4", opt, specializations, 5, 0);
        }

        if (elempack == 8)
        {
            pipeline_instancenorm_coeffs_pack8 = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs_pack8->create("instancenorm_coeffs_pack8", opt, specializations, 5, 0);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(0 + 5);
        specializations[0 + 0].i = shape_packed.dims;
        specializations[0 + 1].i = shape_packed.w;
        specializations[0 + 2].i = shape_packed.h;
        specializations[0 + 3].i = shape_packed.c;
        specializations[0 + 4].i = shape_packed.cstep;

        Mat local_size_xyz(4, 4, std::min(4, channels / elempack), (void*)0);
        if (shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, shape_packed.w);
            local_size_xyz.h = std::min(4, shape_packed.h);
            local_size_xyz.c = std::min(4, shape_packed.c);
        }

        if (elempack == 1)
        {
            pipeline_instancenorm_norm = new Pipeline(vkdev);
            pipeline_instancenorm_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm->create("instancenorm_norm", opt, specializations, 2, 5);
        }

        if (elempack == 4)
        {
            pipeline_instancenorm_norm_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_norm_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm_pack4->create("instancenorm_norm_pack4", opt, specializations, 2, 5);
        }

        if (elempack == 8)
        {
            pipeline_instancenorm_norm_pack8 = new Pipeline(vkdev);
            pipeline_instancenorm_norm_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm_pack8->create("instancenorm_norm_pack8", opt, specializations, 2, 5);
        }
    }

    return 0;
}

int InstanceNorm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_instancenorm_reduce_sum4_fp16_to_fp32;
    pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp32[0];
    delete pipeline_instancenorm_reduce_sum4_fp32[1];
    pipeline_instancenorm_reduce_sum4_fp32[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32[1] = 0;

    delete pipeline_instancenorm_reduce_mean;
    pipeline_instancenorm_reduce_mean = 0;

    delete pipeline_instancenorm_sub_mean_square;
    pipeline_instancenorm_sub_mean_square = 0;

    delete pipeline_instancenorm_coeffs;
    pipeline_instancenorm_coeffs = 0;

    delete pipeline_instancenorm_norm;
    pipeline_instancenorm_norm = 0;

    delete pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4;
    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp32_pack4[0];
    delete pipeline_instancenorm_reduce_sum4_fp32_pack4[1];
    pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = 0;

    delete pipeline_instancenorm_reduce_mean_pack4;
    pipeline_instancenorm_reduce_mean_pack4 = 0;

    delete pipeline_instancenorm_sub_mean_square_pack4;
    pipeline_instancenorm_sub_mean_square_pack4 = 0;

    delete pipeline_instancenorm_coeffs_pack4;
    pipeline_instancenorm_coeffs_pack4 = 0;

    delete pipeline_instancenorm_norm_pack4;
    pipeline_instancenorm_norm_pack4 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8;
    pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8 = 0;

    delete pipeline_instancenorm_reduce_sum4_fp32_pack8[0];
    delete pipeline_instancenorm_reduce_sum4_fp32_pack8[1];
    pipeline_instancenorm_reduce_sum4_fp32_pack8[0] = 0;
    pipeline_instancenorm_reduce_sum4_fp32_pack8[1] = 0;

    delete pipeline_instancenorm_reduce_mean_pack8;
    pipeline_instancenorm_reduce_mean_pack8 = 0;

    delete pipeline_instancenorm_sub_mean_square_pack8;
    pipeline_instancenorm_sub_mean_square_pack8 = 0;

    delete pipeline_instancenorm_coeffs_pack8;
    pipeline_instancenorm_coeffs_pack8 = 0;

    delete pipeline_instancenorm_norm_pack8;
    pipeline_instancenorm_norm_pack8 = 0;

    return 0;
}

int InstanceNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;

    Mat gamma_data_packed;
    convert_packing(gamma_data, gamma_data_packed, elempack);

    cmd.record_upload(gamma_data_packed, gamma_data_gpu, opt);

    Mat beta_data_packed;
    convert_packing(beta_data, beta_data_packed, elempack);

    cmd.record_upload(beta_data_packed, beta_data_gpu, opt);

    return 0;
}

int InstanceNorm_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    // mean
    VkMat mean_workspace(c, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    {
        // reduce sum
        VkMat sum_workspace;
        {
        int reduced_w = (bottom_top_blob.w * bottom_top_blob.h + 3) / 4;
        int reduced_h = 1;
        int reduced_c = bottom_top_blob.c;

        sum_workspace.create(reduced_w, reduced_h, reduced_c, 4u*elempack, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = bottom_top_blob.w * bottom_top_blob.h;
        constants[1].i = bottom_top_blob.c;
        constants[2].i = bottom_top_blob.cstep;
        constants[3].i = sum_workspace.w;
        constants[4].i = sum_workspace.c;
        constants[5].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8
                                 : elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4
                                 : pipeline_instancenorm_reduce_sum4_fp16_to_fp32;

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
        }
        }

        int pb = 0;
        while (sum_workspace.w > 4)
        {
        int reduced_w = (sum_workspace.w + 3) / 4;
        int reduced_h = 1;
        int reduced_c = sum_workspace.c;

        VkMat sum_workspace_reduced;
        sum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u*elempack, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);

        {
        std::vector<VkMat> bindings(2);
        bindings[0] = sum_workspace;
        bindings[1] = sum_workspace_reduced;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = sum_workspace.w;
        constants[1].i = sum_workspace.c;
        constants[2].i = sum_workspace.cstep;
        constants[3].i = sum_workspace_reduced.w;
        constants[4].i = sum_workspace_reduced.c;
        constants[5].i = sum_workspace_reduced.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_reduce_sum4_fp32_pack8[pb%2]
                                 : elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb%2]
                                 : pipeline_instancenorm_reduce_sum4_fp32[pb%2];

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace_reduced);

        pb++;
        }

        sum_workspace = sum_workspace_reduced;
        }

        {
        std::vector<VkMat> bindings(2);
        bindings[0] = sum_workspace;
        bindings[1] = mean_workspace;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = sum_workspace.w;
        constants[1].i = sum_workspace.c;
        constants[2].i = sum_workspace.cstep;
        constants[3].f = size;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_reduce_mean_pack8
                                 : elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4
                                 : pipeline_instancenorm_reduce_mean;

        cmd.record_pipeline(pipeline, bindings, constants, mean_workspace);
        }
    }

    // var
    VkMat var_workspace(c, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    {
        // sub mean and square
        VkMat square_workspace;
        square_workspace.create(w, h, c, 4u*elempack, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = mean_workspace;
        bindings[2] = square_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = square_workspace.dims;
        constants[6].i = square_workspace.w;
        constants[7].i = square_workspace.h;
        constants[8].i = square_workspace.c;
        constants[9].i = square_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_sub_mean_square_pack8
                                 : elempack == 4 ? pipeline_instancenorm_sub_mean_square_pack4
                                 : pipeline_instancenorm_sub_mean_square;

        cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
        }

        // reduce square
        VkMat sqsum_workspace = square_workspace;
        sqsum_workspace.w = sqsum_workspace.w * sqsum_workspace.h;
        sqsum_workspace.h = 1;

        int pb = 0;
        while (sqsum_workspace.w > 4)
        {
        int reduced_w = (sqsum_workspace.w + 3) / 4;
        int reduced_h = 1;
        int reduced_c = sqsum_workspace.c;

        VkMat sqsum_workspace_reduced;
        sqsum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u*elempack, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);

        {
        std::vector<VkMat> bindings(2);
        bindings[0] = sqsum_workspace;
        bindings[1] = sqsum_workspace_reduced;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = sqsum_workspace.w;
        constants[1].i = sqsum_workspace.c;
        constants[2].i = sqsum_workspace.cstep;
        constants[3].i = sqsum_workspace_reduced.w;
        constants[4].i = sqsum_workspace_reduced.c;
        constants[5].i = sqsum_workspace_reduced.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_reduce_sum4_fp32_pack8[pb%2]
                                 : elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb%2]
                                 : pipeline_instancenorm_reduce_sum4_fp32[pb%2];

        cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

        pb++;
        }

        sqsum_workspace = sqsum_workspace_reduced;
        }

        {
        std::vector<VkMat> bindings(2);
        bindings[0] = sqsum_workspace;
        bindings[1] = var_workspace;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = sqsum_workspace.w;
        constants[1].i = sqsum_workspace.c;
        constants[2].i = sqsum_workspace.cstep;
        constants[3].f = size;

        const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_reduce_mean_pack8
                                 : elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4
                                 : pipeline_instancenorm_reduce_mean;

        cmd.record_pipeline(pipeline, bindings, constants, var_workspace);
        }
    }

    // coeffs
    VkMat coeffs_workspace;
    coeffs_workspace.create(c, elemsize * 2, elempack * 2, opt.workspace_vkallocator, opt.staging_vkallocator);
    {
    std::vector<VkMat> bindings(5);
    bindings[0] = coeffs_workspace;
    bindings[1] = mean_workspace;
    bindings[2] = var_workspace;
    bindings[3] = gamma_data_gpu;
    bindings[4] = beta_data_gpu;

    std::vector<vk_constant_type> constants(0);

    const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_coeffs_pack8
                             : elempack == 4 ? pipeline_instancenorm_coeffs_pack4
                             : pipeline_instancenorm_coeffs;

    cmd.record_pipeline(pipeline, bindings, constants, coeffs_workspace);
    }

    // norm
    {
    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = coeffs_workspace;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_instancenorm_norm_pack8
                             : elempack == 4 ? pipeline_instancenorm_norm_pack4
                             : pipeline_instancenorm_norm;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
