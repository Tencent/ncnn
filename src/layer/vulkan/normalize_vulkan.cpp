// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Normalize_vulkan::Normalize_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

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

    // subgroup
    pipeline_normalize_reduce_subgroup = 0;
    pipeline_normalize_reduce_subgroup_pack4 = 0;
}

int Normalize_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the subgroup reduce performs the whole square sum in a single pass
    const int subgroup_size = vkdev->info.subgroup_size();
    const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
    const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    bool use_subgroup_ops = opt.use_subgroup_ops && ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize wired subgroup_size
        use_subgroup_ops = false;
    }
    if (!(across_spatial == 1 && across_channel == 0))
    {
        // the across-channel reduce is only a handful of elements long and measured slower
        // than the existing multi pass scheme, the global square sum produces a single value
        // that the multi pass scheme spreads much wider
        use_subgroup_ops = false;
    }

    if (shape.dims == 0 || shape.w * shape.h * shape.d < subgroup_size || shape.c < 2)
    {
        // the subgroup reduce computes one channel per workgroup, only worth it when the
        // spatial plane is at least one subgroup wide
        use_subgroup_ops = false;
    }

    // widen the workgroup with a few subgroups per channel so that a wide gpu has enough lanes
    // to work with, while a narrow one does not get empty subgroups
    const int spatial_size = shape.w * shape.h * shape.d;
    const int max_unroll = std::max(1, (int)vkdev->info.max_workgroup_invocations() / subgroup_size);
    const int subgroup_unroll = std::max(1, std::min(max_unroll, spatial_size / (subgroup_size * 8)));
    const int subgroup_local_size = subgroup_size * subgroup_unroll;

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;

        Mat local_size_xyz; // TODO select by across_channel / across_spatial

        if (!use_subgroup_ops)
        {
            // pack1
            if (shape.dims == 0 || shape.elempack == 1)
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
            if (shape.dims == 0 || shape.elempack == 4)
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
        }

        if (use_subgroup_ops)
        {
            // one channel per workgroup, the lanes split its spatial plane
            std::vector<vk_specialization_type> subgroup_specializations(1);
            subgroup_specializations[0].i = subgroup_local_size;

            if (shape.dims == 0 || shape.elempack == 1)
            {
                pipeline_normalize_reduce_subgroup = new Pipeline(vkdev);
                pipeline_normalize_reduce_subgroup->set_subgroup_size(subgroup_size);
                pipeline_normalize_reduce_subgroup->set_local_size_xyz(subgroup_local_size, 1, 1);
                pipeline_normalize_reduce_subgroup->create(LayerShaderType::normalize_reduce_subgroup, opt, subgroup_specializations);
            }

            if (shape.dims == 0 || shape.elempack == 4)
            {
                pipeline_normalize_reduce_subgroup_pack4 = new Pipeline(vkdev);
                pipeline_normalize_reduce_subgroup_pack4->set_subgroup_size(subgroup_size);
                pipeline_normalize_reduce_subgroup_pack4->set_local_size_xyz(subgroup_local_size, 1, 1);
                pipeline_normalize_reduce_subgroup_pack4->create(LayerShaderType::normalize_reduce_subgroup_pack4, opt, subgroup_specializations);
            }
        }
    }

    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;
        specializations[2].f = eps;
        specializations[3].i = eps_mode;

        Mat local_size_xyz; // TODO resolve sqsum_workspace shape

        if (shape.dims == 0 || shape.elempack == 1)
        {
            pipeline_normalize_coeffs = new Pipeline(vkdev);
            pipeline_normalize_coeffs->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_coeffs->create(LayerShaderType::normalize_coeffs, opt, specializations);
        }

        if (shape.dims == 0 || shape.elempack == 4)
        {
            pipeline_normalize_coeffs_pack4 = new Pipeline(vkdev);
            pipeline_normalize_coeffs_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_coeffs_pack4->create(LayerShaderType::normalize_coeffs_pack4, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(5 + 5);
        specializations[0].i = across_spatial;
        specializations[1].i = across_channel;
        specializations[2].i = channel_shared;
        specializations[3].i = (scale_data_size == 1 && scale_data[0] == 1.f) ? 0 : 1;
        specializations[4].f = channel_shared ? scale_data[0] : 1.f;
        specializations[5 + 0].i = std::min(3, shape.dims);
        specializations[5 + 1].i = shape.w;
        specializations[5 + 2].i = shape.h * shape.d;
        specializations[5 + 3].i = shape.c;
        specializations[5 + 4].i = shape.cstep;

        Mat local_size_xyz;
        if (shape.dims != 0)
        {
            local_size_xyz.w = std::min(4, shape.w);
            local_size_xyz.h = std::min(4, shape.h * shape.d);
            local_size_xyz.c = std::min(4, shape.c);
        }

        if (shape.dims == 0 || shape.elempack == 1)
        {
            pipeline_normalize_norm = new Pipeline(vkdev);
            pipeline_normalize_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_norm->create(LayerShaderType::normalize_norm, opt, specializations);
        }

        if (shape.dims == 0 || shape.elempack == 4)
        {
            pipeline_normalize_norm_pack4 = new Pipeline(vkdev);
            pipeline_normalize_norm_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_normalize_norm_pack4->create(LayerShaderType::normalize_norm_pack4, opt, specializations);
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

    delete pipeline_normalize_coeffs;
    pipeline_normalize_coeffs = 0;

    delete pipeline_normalize_coeffs_pack4;
    pipeline_normalize_coeffs_pack4 = 0;

    delete pipeline_normalize_norm;
    pipeline_normalize_norm = 0;

    delete pipeline_normalize_norm_pack4;
    pipeline_normalize_norm_pack4 = 0;

    delete pipeline_normalize_reduce_subgroup;
    pipeline_normalize_reduce_subgroup = 0;

    delete pipeline_normalize_reduce_subgroup_pack4;
    pipeline_normalize_reduce_subgroup_pack4 = 0;

    return 0;
}

int Normalize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (!channel_shared && !(scale_data_size == 1 && scale_data[0] == 1.f))
    {
        cmd.record_upload(scale_data, scale_data_gpu, opt);

        if (opt.lightmode)
            scale_data.release();
    }

    return 0;
}

int Normalize_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h * bottom_top_blob.d;
    int size = w * h;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    // reduce square sum
    VkMat sqsum_workspace;
    const Pipeline* pipeline_reduce_subgroup = elempack == 4 ? pipeline_normalize_reduce_subgroup_pack4 : pipeline_normalize_reduce_subgroup;

    if (pipeline_reduce_subgroup)
    {
        // one channel per workgroup, the lanes split the spatial plane
        const int outc = bottom_top_blob.c;

        sqsum_workspace.create(outc, 4u * elempack, elempack, opt.workspace_vkallocator);

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sqsum_workspace;

        std::vector<vk_constant_type> constants(3);
        constants[0].i = size;
        constants[1].i = bottom_top_blob.cstep;
        constants[2].i = outc;

        // split the channels over x and y to stay within the dispatch limits
        const int max_workgroup_count_y = (int)vkdev->info.max_workgroup_count_y();
        const int group_count_x = (outc - 1) / max_workgroup_count_y + 1;
        const int group_count_y = (outc - 1) / group_count_x + 1;

        VkMat dispatcher;
        dispatcher.w = group_count_x * pipeline_reduce_subgroup->local_size_x();
        dispatcher.h = group_count_y;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_reduce_subgroup, bindings, constants, dispatcher);
    }
    else
    {
        {
            {
                int reduced_w;
                int reduced_h;
                int reduced_c;

                if (across_spatial && across_channel)
                {
                    reduced_w = (size + 1) / 2;
                    reduced_h = 1;
                    reduced_c = (bottom_top_blob.c + 1) / 2;
                }
                else if (across_spatial && !across_channel)
                {
                    reduced_w = (size + 3) / 4;
                    reduced_h = 1;
                    reduced_c = bottom_top_blob.c;
                }
                else // if (!across_spatial && across_channel)
                {
                    reduced_w = size;
                    reduced_h = 1;
                    reduced_c = (bottom_top_blob.c + 3) / 4;
                }

                sqsum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
                {
                    std::vector<VkMat> bindings(2);
                    bindings[0] = bottom_top_blob;
                    bindings[1] = sqsum_workspace;

                    std::vector<vk_constant_type> constants(8);
                    constants[0].i = size;
                    constants[1].i = 1;
                    constants[2].i = bottom_top_blob.c;
                    constants[3].i = bottom_top_blob.cstep;
                    constants[4].i = sqsum_workspace.w;
                    constants[5].i = 1;
                    constants[6].i = sqsum_workspace.c;
                    constants[7].i = sqsum_workspace.cstep;

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

                    const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_reduce_sum4_fp32_pack4[pb % 2] : pipeline_normalize_reduce_sum4_fp32[pb % 2];

                    cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

                    pb++;
                }

                sqsum_workspace = sqsum_workspace_reduced;
            }
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

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_coeffs_pack4 : pipeline_normalize_coeffs;

        cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace);
    }

    // norm
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = coeffs_workspace;
        bindings[2] = scale_data_gpu;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = std::min(3, bottom_top_blob.dims);
        constants[1].i = bottom_top_blob.w;
        constants[2].i = h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_normalize_norm_pack4 : pipeline_normalize_norm;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn