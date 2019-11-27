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

namespace ncnn {

DEFINE_LAYER_CREATOR(Normalize_vulkan)

Normalize_vulkan::Normalize_vulkan()
{
    support_vulkan = true;

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
}

int Normalize_vulkan::create_pipeline(const Option& opt)
{
    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;

        // pack1
        pipeline_normalize_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp16_to_fp32->create("normalize_reduce_sum4_fp16_to_fp32", opt, specializations, 2, 6);

        pipeline_normalize_reduce_sum4_fp32[0] = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp32[0]->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp32[0]->create("normalize_reduce_sum4_fp32", opt, specializations, 2, 6);
        pipeline_normalize_reduce_sum4_fp32[1] = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp32[1]->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp32[1]->create("normalize_reduce_sum4_fp32", opt, specializations, 2, 6);

        // pack4
        pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4->create("normalize_reduce_sum4_fp16_to_fp32_pack4", opt, specializations, 2, 6);

        pipeline_normalize_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp32_pack4[0]->create("normalize_reduce_sum4_fp32_pack4", opt, specializations, 2, 6);
        pipeline_normalize_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
        pipeline_normalize_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz();
        pipeline_normalize_reduce_sum4_fp32_pack4[1]->create("normalize_reduce_sum4_fp32_pack4", opt, specializations, 2, 6);
    }

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].f = eps;
        specializations[1].i = eps_mode;

        pipeline_normalize_coeffs = new Pipeline(vkdev);
        pipeline_normalize_coeffs->set_optimal_local_size_xyz();
        pipeline_normalize_coeffs->create("normalize_coeffs", opt, specializations, 2, 3);

        pipeline_normalize_coeffs_pack4 = new Pipeline(vkdev);
        pipeline_normalize_coeffs_pack4->set_optimal_local_size_xyz();
        pipeline_normalize_coeffs_pack4->create("normalize_coeffs_pack4", opt, specializations, 2, 3);
    }

    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;
        specializations[2].i = channel_shared;
        specializations[3].i = (scale_data_size == 1 && scale_data[0] == 1.f) ? 0 : 1;

        pipeline_normalize_norm = new Pipeline(vkdev);
        pipeline_normalize_norm->set_optimal_local_size_xyz();
        pipeline_normalize_norm->create("normalize_norm", opt, specializations, 3, 5);

        pipeline_normalize_norm_pack4 = new Pipeline(vkdev);
        pipeline_normalize_norm_pack4->set_optimal_local_size_xyz();
        pipeline_normalize_norm_pack4->create("normalize_norm_pack4", opt, specializations, 3, 5);
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

    delete pipeline_normalize_coeffs;
    pipeline_normalize_coeffs = 0;

    delete pipeline_normalize_coeffs_pack4;
    pipeline_normalize_coeffs_pack4 = 0;

    delete pipeline_normalize_norm;
    pipeline_normalize_norm = 0;

    delete pipeline_normalize_norm_pack4;
    pipeline_normalize_norm_pack4 = 0;

    return 0;
}

int Normalize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_data_size == 1 && scale_data[0] != 1.f)
    {
        // dup4 for pack4
        Mat scale_data4(4);
        scale_data4.fill(scale_data[0]);
        cmd.record_upload(scale_data4, scale_data_gpu, opt);

        Mat scale_data_pack4;
        convert_packing(scale_data4, scale_data_pack4, 4);
        cmd.record_upload(scale_data_pack4, scale_data_gpu_pack4, opt);
    }
    else if (scale_data_size % 4 == 0)
    {
        Mat scale_data_pack4;
        convert_packing(scale_data, scale_data_pack4, 4);
        cmd.record_upload(scale_data_pack4, scale_data_gpu_pack4, opt);
    }
    else
    {
        cmd.record_upload(scale_data, scale_data_gpu, opt);
    }

    return 0;
}

int Normalize_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;
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

        sqsum_workspace.create(reduced_w, reduced_h, reduced_c, 4u*elempack, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
        {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sqsum_workspace;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = bottom_top_blob.w * bottom_top_blob.h;
        constants[1].i = bottom_top_blob.c;
        constants[2].i = bottom_top_blob.cstep;
        constants[3].i = sqsum_workspace.w;
        constants[4].i = sqsum_workspace.c;
        constants[5].i = sqsum_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4 : pipeline_normalize_reduce_sum4_fp16_to_fp32;

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

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_reduce_sum4_fp32_pack4[pb%2] : pipeline_normalize_reduce_sum4_fp32[pb%2];

        cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

        pb++;
        }

        sqsum_workspace = sqsum_workspace_reduced;
        }
    }

    // coeffs
    VkMat coeffs_workspace;
    coeffs_workspace.create(sqsum_workspace.w, sqsum_workspace.h, sqsum_workspace.c, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = sqsum_workspace;
        bindings[1] = coeffs_workspace;

        std::vector<vk_constant_type> constants(3);
        constants[0].i = sqsum_workspace.w;
        constants[1].i = sqsum_workspace.c;
        constants[2].i = sqsum_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_coeffs_pack4 : pipeline_normalize_coeffs;

        cmd.record_pipeline(pipeline, bindings, constants, coeffs_workspace);
    }

    // norm
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = coeffs_workspace;
        bindings[2] = (scale_data_size == 1 && scale_data[0] == 1.f) ? coeffs_workspace : elempack == 4 ? scale_data_gpu_pack4 : scale_data_gpu;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_norm_pack4 : pipeline_normalize_norm;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
