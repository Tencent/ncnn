// Copyright 2025 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_vulkan.h"
#include "layer_shader_type.h"

namespace ncnn {

RMSNorm_vulkan::RMSNorm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    // pack1
    pipeline_rmsnorm_square = 0;
    pipeline_rmsnorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_rmsnorm_reduce_sum4_fp32[0] = 0;
    pipeline_rmsnorm_reduce_sum4_fp32[1] = 0;
    pipeline_rmsnorm_reduce_mean = 0;
    pipeline_rmsnorm_coeffs = 0;
    pipeline_rmsnorm_norm = 0;

    // pack4
    pipeline_rmsnorm_square_pack4 = 0;
    pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_rmsnorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_rmsnorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_rmsnorm_reduce_mean_pack4 = 0;
    pipeline_rmsnorm_coeffs_pack4 = 0;
    pipeline_rmsnorm_norm_pack4 = 0;
}

int RMSNorm_vulkan::create_pipeline(const Option& opt)
{
    // Reuse the LN reduce_sum4 / reduce_mean size and local workgroup configuration [from LN code].
    // square: only compute x^2; coeffs: a = 1 / sqrt(rms + eps); norm: v = v * a; (if affine: v *= gamma)
    {
        pipeline_rmsnorm_square = new Pipeline(vkdev);
        pipeline_rmsnorm_square->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_square->create(LayerShaderType::rmsnorm_square, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_square_pack4 = new Pipeline(vkdev);
        pipeline_rmsnorm_square_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_square_pack4->create(LayerShaderType::rmsnorm_square_pack4, opt, std::vector<vk_specialization_type>());
    }
    {
        // Same as LN's reduce_sum4 (first pass fp16->fp32, then iterative fp32 passes alternating between two pipelines) and reduce_mean (see LN).
        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(16, 4, 1);
        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_reduce_sum4_fp32[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
        pipeline_rmsnorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_reduce_sum4_fp32[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(16, 4, 1);
        pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32_pack4, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_reduce_sum4_fp32_pack4[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
        pipeline_rmsnorm_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_reduce_sum4_fp32_pack4[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_reduce_mean = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_mean->set_optimal_local_size_xyz(1, 8, 8);
        pipeline_rmsnorm_reduce_mean->create(LayerShaderType::layernorm_reduce_mean, opt, std::vector<vk_specialization_type>());

        pipeline_rmsnorm_reduce_mean_pack4 = new Pipeline(vkdev);
        pipeline_rmsnorm_reduce_mean_pack4->set_optimal_local_size_xyz(1, 8, 8);
        pipeline_rmsnorm_reduce_mean_pack4->create(LayerShaderType::layernorm_reduce_mean_pack4, opt, std::vector<vk_specialization_type>());
    }
    {
        // coeffs: only eps is used as a specialization constant
        std::vector<vk_specialization_type> spec(1);
        spec[0].f = eps;

        pipeline_rmsnorm_coeffs = new Pipeline(vkdev);
        pipeline_rmsnorm_coeffs->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_coeffs->create(LayerShaderType::rmsnorm_coeffs, opt, spec);

        pipeline_rmsnorm_coeffs_pack4 = new Pipeline(vkdev);
        pipeline_rmsnorm_coeffs_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_coeffs_pack4->create(LayerShaderType::rmsnorm_coeffs_pack4, opt, spec);
    }
    {
        // norm: requires affine and affine_size (which determine the mapping from inner_id to group_id, identical to LayerNorm)
        std::vector<vk_specialization_type> spec(2);
        spec[0].i = affine;
        spec[1].i = affine_size;

        pipeline_rmsnorm_norm = new Pipeline(vkdev);
        pipeline_rmsnorm_norm->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_norm->create(LayerShaderType::rmsnorm_norm, opt, spec);

        pipeline_rmsnorm_norm_pack4 = new Pipeline(vkdev);
        pipeline_rmsnorm_norm_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_rmsnorm_norm_pack4->create(LayerShaderType::rmsnorm_norm_pack4, opt, spec);
    }
    return 0;
}

int RMSNorm_vulkan::destroy_pipeline(const Option&)
{
    // pack1
    delete pipeline_rmsnorm_square;
    delete pipeline_rmsnorm_reduce_sum4_fp16_to_fp32;
    delete pipeline_rmsnorm_reduce_sum4_fp32[0];
    delete pipeline_rmsnorm_reduce_sum4_fp32[1];
    delete pipeline_rmsnorm_reduce_mean;
    delete pipeline_rmsnorm_coeffs;
    delete pipeline_rmsnorm_norm;
    pipeline_rmsnorm_square = 0;
    pipeline_rmsnorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_rmsnorm_reduce_sum4_fp32[0] = 0;
    pipeline_rmsnorm_reduce_sum4_fp32[1] = 0;
    pipeline_rmsnorm_reduce_mean = 0;
    pipeline_rmsnorm_coeffs = 0;
    pipeline_rmsnorm_norm = 0;

    // pack4
    delete pipeline_rmsnorm_square_pack4;
    delete pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4;
    delete pipeline_rmsnorm_reduce_sum4_fp32_pack4[0];
    delete pipeline_rmsnorm_reduce_sum4_fp32_pack4[1];
    delete pipeline_rmsnorm_reduce_mean_pack4;
    delete pipeline_rmsnorm_coeffs_pack4;
    delete pipeline_rmsnorm_norm_pack4;
    pipeline_rmsnorm_square_pack4 = 0;
    pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_rmsnorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_rmsnorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_rmsnorm_reduce_mean_pack4 = 0;
    pipeline_rmsnorm_coeffs_pack4 = 0;
    pipeline_rmsnorm_norm_pack4 = 0;

    return 0;
}

int RMSNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (affine == 0) return 0;
    cmd.record_upload(gamma_data, gamma_data_gpu, opt);
    return 0;
}

int RMSNorm_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    // Same 1D unpacking logic and cstep/elempack handling as LayerNorm (see LN for reference).
    const int dims = bottom_top_blob.dims;
    const int w = dims == 1 ? bottom_top_blob.w * bottom_top_blob.elempack : bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const size_t elemsize = dims == 1 ? bottom_top_blob.elemsize * bottom_top_blob.elempack : bottom_top_blob.elemsize;
    const int elempack = dims == 1 ? 1 : bottom_top_blob.elempack;
    const size_t cstep = dims == 1 ? bottom_top_blob.cstep * bottom_top_blob.elempack : bottom_top_blob.cstep;

    if (affine_size == 0)
        return 0;

    int group_size = 0;
    int num_groups_per_channel = 0;

    if (dims == 1)
    {
        group_size = w;
        num_groups_per_channel = 1;
    }
    else if (dims == 2)
    {
        group_size = w;
        num_groups_per_channel = h;
    }
    else
    {
        if (affine_size == w)
        {
            group_size = w;
            num_groups_per_channel = h;
        }
        else
        {
            group_size = w * h;
            num_groups_per_channel = 1;
        } // affine_size == w*h
    }
    int num_groups_total = num_groups_per_channel * channels;

    // 1) x -> x^2
    VkMat square_workspace(w, h, channels, elemsize, elempack, opt.workspace_vkallocator);
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = square_workspace;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = h;
        constants[2].i = channels;
        constants[3].i = cstep;

        const Pipeline* pipe_sq = elempack == 4 ? pipeline_rmsnorm_square_pack4 : pipeline_rmsnorm_square;

        cmd.record_pipeline(pipe_sq, bindings, constants, square_workspace);
    }

    // 2) reduce sum4 (square) -> ... -> mean
    VkMat rms_workspace(num_groups_total, 4u * elempack, elempack, opt.workspace_vkallocator);
    {
        int reduced_w = (group_size + 3) / 4;
        VkMat sqsum_workspace;
        sqsum_workspace.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = square_workspace;
            bindings[1] = sqsum_workspace;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = group_size;
            constants[1].i = num_groups_per_channel;
            constants[2].i = channels;
            constants[3].i = square_workspace.cstep;
            constants[4].i = reduced_w;
            constants[5].i = num_groups_per_channel;
            constants[6].i = channels;
            constants[7].i = sqsum_workspace.cstep;

            VkMat dispatcher;
            dispatcher.w = reduced_w;
            dispatcher.h = num_groups_per_channel;
            dispatcher.c = channels;
            const Pipeline* p_reduce = elempack == 4 ? pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4
                                       : pipeline_rmsnorm_reduce_sum4_fp16_to_fp32;
            cmd.record_pipeline(p_reduce, bindings, constants, dispatcher);
        }
        int pb = 1;
        while (sqsum_workspace.w > 1)
        {
            int current_w = sqsum_workspace.w;
            reduced_w = (current_w + 3) / 4;

            VkMat sqsum_reduced;
            sqsum_reduced.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

            std::vector<VkMat> bindings(2);
            bindings[0] = sqsum_workspace;
            bindings[1] = sqsum_reduced;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = current_w;
            constants[1].i = num_groups_per_channel;
            constants[2].i = channels;
            constants[3].i = sqsum_workspace.cstep;
            constants[4].i = reduced_w;
            constants[5].i = num_groups_per_channel;
            constants[6].i = channels;
            constants[7].i = sqsum_reduced.cstep;

            VkMat dispatcher;
            dispatcher.w = reduced_w;
            dispatcher.h = num_groups_per_channel;
            dispatcher.c = channels;
            const Pipeline* p_iter = elempack == 4 ? pipeline_rmsnorm_reduce_sum4_fp32_pack4[pb % 2]
                                     : pipeline_rmsnorm_reduce_sum4_fp32[pb % 2];
            cmd.record_pipeline(p_iter, bindings, constants, dispatcher);
            pb++;
            sqsum_workspace = sqsum_reduced;
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = sqsum_workspace;
            bindings[1] = rms_workspace;

            std::vector<vk_constant_type> constants(5);
            constants[0].i = sqsum_workspace.w;
            constants[1].i = num_groups_per_channel;
            constants[2].i = channels;
            constants[3].i = sqsum_workspace.cstep;
            constants[4].f = (float)group_size;

            VkMat dispatcher;
            dispatcher.w = 1;
            dispatcher.h = num_groups_per_channel;
            dispatcher.c = channels;
            const Pipeline* p_mean = elempack == 4 ? pipeline_rmsnorm_reduce_mean_pack4 : pipeline_rmsnorm_reduce_mean;
            cmd.record_pipeline(p_mean, bindings, constants, dispatcher);
        }
    }

    // 3) coeffs (a) from rms
    VkMat coeffs_workspace(num_groups_total, elemsize, elempack, opt.workspace_vkallocator); // only a, no b
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = coeffs_workspace;
        bindings[1] = rms_workspace;

        std::vector<vk_constant_type> constants(2);
        constants[0].i = num_groups_per_channel;
        constants[1].i = channels;

        VkMat dispatcher;
        dispatcher.w = 1;
        dispatcher.h = num_groups_per_channel;
        dispatcher.c = channels;
        const Pipeline* p_coeffs = elempack == 4 ? pipeline_rmsnorm_coeffs_pack4 : pipeline_rmsnorm_coeffs;
        cmd.record_pipeline(p_coeffs, bindings, constants, dispatcher);
    }

    // 4) norm: v = v * a; (affine? v *= gamma)
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = coeffs_workspace;
        bindings[2] = gamma_data_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = h;
        constants[2].i = channels;
        constants[3].i = cstep;

        const Pipeline* p_norm = elempack == 4 ? pipeline_rmsnorm_norm_pack4 : pipeline_rmsnorm_norm;

        VkMat dispatcher;
        dispatcher.w = w;
        dispatcher.h = h;
        dispatcher.c = channels;
        cmd.record_pipeline(p_norm, bindings, constants, dispatcher);
    }
    return 0;
}

} // namespace ncnn
