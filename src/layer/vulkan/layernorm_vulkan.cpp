// Copyright 2025 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_vulkan.h"
#include "layer_shader_type.h"

namespace ncnn {

LayerNorm_vulkan::LayerNorm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    // pack1
    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_layernorm_reduce_sum4_fp32[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32[1] = 0;
    pipeline_layernorm_reduce_mean = 0;
    pipeline_layernorm_sub_mean_square = 0;
    pipeline_layernorm_coeffs = 0;
    pipeline_layernorm_norm = 0;

    // pack4
    pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_layernorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_layernorm_reduce_mean_pack4 = 0;
    pipeline_layernorm_sub_mean_square_pack4 = 0;
    pipeline_layernorm_coeffs_pack4 = 0;
    pipeline_layernorm_norm_pack4 = 0;
}

int LayerNorm_vulkan::create_pipeline(const Option& opt)
{
    {
        pipeline_layernorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(16, 4, 1);
        pipeline_layernorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32, opt, std::vector<vk_specialization_type>());

        pipeline_layernorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_reduce_sum4_fp32[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
        pipeline_layernorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_reduce_sum4_fp32[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());

        pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(16, 4, 1);
        pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32_pack4, opt, std::vector<vk_specialization_type>());

        pipeline_layernorm_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_reduce_sum4_fp32_pack4[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
        pipeline_layernorm_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
        pipeline_layernorm_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_reduce_sum4_fp32_pack4[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
    }

    {
        pipeline_layernorm_reduce_mean = new Pipeline(vkdev);
        pipeline_layernorm_reduce_mean->set_optimal_local_size_xyz(1, 8, 8);
        pipeline_layernorm_reduce_mean->create(LayerShaderType::layernorm_reduce_mean, opt, std::vector<vk_specialization_type>());

        pipeline_layernorm_reduce_mean_pack4 = new Pipeline(vkdev);
        pipeline_layernorm_reduce_mean_pack4->set_optimal_local_size_xyz(1, 8, 8);
        pipeline_layernorm_reduce_mean_pack4->create(LayerShaderType::layernorm_reduce_mean_pack4, opt, std::vector<vk_specialization_type>());
    }

    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = affine_size;

        pipeline_layernorm_sub_mean_square = new Pipeline(vkdev);
        pipeline_layernorm_sub_mean_square->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_sub_mean_square->create(LayerShaderType::layernorm_sub_mean_square, opt, specializations);

        pipeline_layernorm_sub_mean_square_pack4 = new Pipeline(vkdev);
        pipeline_layernorm_sub_mean_square_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_sub_mean_square_pack4->create(LayerShaderType::layernorm_sub_mean_square_pack4, opt, specializations);
    }

    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].f = eps;

        pipeline_layernorm_coeffs = new Pipeline(vkdev);
        pipeline_layernorm_coeffs->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_coeffs->create(LayerShaderType::layernorm_coeffs, opt, specializations);

        pipeline_layernorm_coeffs_pack4 = new Pipeline(vkdev);
        pipeline_layernorm_coeffs_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_coeffs_pack4->create(LayerShaderType::layernorm_coeffs_pack4, opt, specializations);
    }

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = affine;
        specializations[1].i = affine_size;

        pipeline_layernorm_norm = new Pipeline(vkdev);
        pipeline_layernorm_norm->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_norm->create(LayerShaderType::layernorm_norm, opt, specializations);

        pipeline_layernorm_norm_pack4 = new Pipeline(vkdev);
        pipeline_layernorm_norm_pack4->set_optimal_local_size_xyz(8, 8, 1);
        pipeline_layernorm_norm_pack4->create(LayerShaderType::layernorm_norm_pack4, opt, specializations);
    }

    return 0;
}

int LayerNorm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    // pack1
    delete pipeline_layernorm_reduce_sum4_fp16_to_fp32;
    delete pipeline_layernorm_reduce_sum4_fp32[0];
    delete pipeline_layernorm_reduce_sum4_fp32[1];
    delete pipeline_layernorm_reduce_mean;
    delete pipeline_layernorm_sub_mean_square;
    delete pipeline_layernorm_coeffs;
    delete pipeline_layernorm_norm;
    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_layernorm_reduce_sum4_fp32[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32[1] = 0;
    pipeline_layernorm_reduce_mean = 0;
    pipeline_layernorm_sub_mean_square = 0;
    pipeline_layernorm_coeffs = 0;
    pipeline_layernorm_norm = 0;

    // pack4
    delete pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4;
    delete pipeline_layernorm_reduce_sum4_fp32_pack4[0];
    delete pipeline_layernorm_reduce_sum4_fp32_pack4[1];
    delete pipeline_layernorm_reduce_mean_pack4;
    delete pipeline_layernorm_sub_mean_square_pack4;
    delete pipeline_layernorm_coeffs_pack4;
    delete pipeline_layernorm_norm_pack4;
    pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4 = 0;
    pipeline_layernorm_reduce_sum4_fp32_pack4[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32_pack4[1] = 0;
    pipeline_layernorm_reduce_mean_pack4 = 0;
    pipeline_layernorm_sub_mean_square_pack4 = 0;
    pipeline_layernorm_coeffs_pack4 = 0;
    pipeline_layernorm_norm_pack4 = 0;

    return 0;
}

int LayerNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (affine == 0)
        return 0;

    cmd.record_upload(gamma_data, gamma_data_gpu, opt);

    cmd.record_upload(beta_data, beta_data_gpu, opt);

    return 0;
}

int LayerNorm_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    // treat 1d blob as unpacked layout
    const int dims = bottom_top_blob.dims;
    const int w = dims == 1 ? bottom_top_blob.w * bottom_top_blob.elempack : bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const size_t elemsize = dims == 1 ? bottom_top_blob.elemsize * bottom_top_blob.elempack : bottom_top_blob.elemsize;
    const int elempack = dims == 1 ? 1 : bottom_top_blob.elempack;
    const size_t cstep = dims == 1 ? bottom_top_blob.cstep * bottom_top_blob.elempack : bottom_top_blob.cstep;

    if (affine_size == 0)
        return 0;

    int group_size;
    int num_groups_per_channel;
    if (dims == 1)
    {
        // (w)
        group_size = w;
        num_groups_per_channel = 1;
    }
    else if (dims == 2)
    {
        // (w, h)
        group_size = w;
        num_groups_per_channel = h;
    }
    else
    {
        // (w, h, c)
        if (affine_size == w)
        {
            group_size = w;
            num_groups_per_channel = h;
        }
        else
        {   // affine_size == w * h, like InstanceNorm
            group_size = w * h;
            num_groups_per_channel = 1;
        }
    }
    int num_groups_total = num_groups_per_channel * channels;

    VkMat mean_workspace(num_groups_total, 4u * elempack, elempack, opt.workspace_vkallocator);
    {
        int reduced_w = (group_size + 3) / 4;
        VkMat sum_workspace;
        sum_workspace.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = group_size;
        constants[1].i = num_groups_per_channel;
        constants[2].i = channels;
        constants[3].i = cstep;
        constants[4].i = reduced_w;
        constants[5].i = num_groups_per_channel;
        constants[6].i = channels;
        constants[7].i = sum_workspace.cstep;

        VkMat dispatcher;
        dispatcher.w = reduced_w;
        dispatcher.h = num_groups_per_channel;
        dispatcher.c = channels;

        const Pipeline* pipeline_reduce_sum4 = elempack == 4 ? pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4 : pipeline_layernorm_reduce_sum4_fp16_to_fp32;

        cmd.record_pipeline(pipeline_reduce_sum4, bindings, constants, dispatcher);

        int pb = 1;
        while (sum_workspace.w > 1)
        {
            int current_w = sum_workspace.w;
            reduced_w = (current_w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

            std::vector<VkMat> bindings_iter(2);
            bindings_iter[0] = sum_workspace;
            bindings_iter[1] = sum_workspace_reduced;

            std::vector<vk_constant_type> constants_iter(8);
            constants_iter[0].i = current_w;
            constants_iter[1].i = num_groups_per_channel;
            constants_iter[2].i = channels;
            constants_iter[3].i = sum_workspace.cstep;
            constants_iter[4].i = reduced_w;
            constants_iter[5].i = num_groups_per_channel;
            constants_iter[6].i = channels;
            constants_iter[7].i = sum_workspace_reduced.cstep;

            dispatcher.w = reduced_w;

            const Pipeline* pipeline_reduce_iter = elempack == 4 ? pipeline_layernorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_layernorm_reduce_sum4_fp32[pb % 2];
            cmd.record_pipeline(pipeline_reduce_iter, bindings_iter, constants_iter, dispatcher);
            pb++;
            sum_workspace = sum_workspace_reduced;
        }

        std::vector<VkMat> mean_bindings(2);
        mean_bindings[0] = sum_workspace;
        mean_bindings[1] = mean_workspace;

        std::vector<vk_constant_type> mean_constants(5);
        mean_constants[0].i = sum_workspace.w;
        mean_constants[1].i = num_groups_per_channel;
        mean_constants[2].i = channels;
        mean_constants[3].i = sum_workspace.cstep;
        mean_constants[4].f = (float)group_size;

        dispatcher.w = 1;
        const Pipeline* pipeline_reduce_mean = elempack == 4 ? pipeline_layernorm_reduce_mean_pack4 : pipeline_layernorm_reduce_mean;
        cmd.record_pipeline(pipeline_reduce_mean, mean_bindings, mean_constants, dispatcher);
    }

    VkMat var_workspace(num_groups_total, 4u * elempack, elempack, opt.workspace_vkallocator);
    {
        VkMat square_workspace(w, h, channels, elemsize, elempack, opt.workspace_vkallocator);
        {
            std::vector<VkMat> sq_bindings(3);
            sq_bindings[0] = bottom_top_blob;
            sq_bindings[1] = mean_workspace;
            sq_bindings[2] = square_workspace;

            std::vector<vk_constant_type> sq_constants(4);
            sq_constants[0].i = w;
            sq_constants[1].i = h;
            sq_constants[2].i = channels;
            sq_constants[3].i = cstep;

            const Pipeline* pipeline_sub_mean_square = elempack == 4 ? pipeline_layernorm_sub_mean_square_pack4 : pipeline_layernorm_sub_mean_square;
            cmd.record_pipeline(pipeline_sub_mean_square, sq_bindings, sq_constants, square_workspace);
        }

        // Reduce sum of squares
        int reduced_w = (group_size + 3) / 4;
        VkMat sqsum_workspace;
        sqsum_workspace.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

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

        const Pipeline* pipeline_reduce_sum4 = elempack == 4 ? pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4 : pipeline_layernorm_reduce_sum4_fp16_to_fp32;

        cmd.record_pipeline(pipeline_reduce_sum4, bindings, constants, dispatcher);

        int pb = 1;
        while (sqsum_workspace.w > 1)
        {
            int current_w = sqsum_workspace.w;
            reduced_w = (current_w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, num_groups_per_channel, channels, 4u * elempack, elempack, opt.workspace_vkallocator);

            std::vector<VkMat> bindings_iter(2);
            bindings_iter[0] = sqsum_workspace;
            bindings_iter[1] = sum_workspace_reduced;
            std::vector<vk_constant_type> constants_iter(8);
            constants_iter[0].i = current_w;
            constants_iter[1].i = num_groups_per_channel;
            constants_iter[2].i = channels;
            constants_iter[3].i = sqsum_workspace.cstep;
            constants_iter[4].i = reduced_w;
            constants_iter[5].i = num_groups_per_channel;
            constants_iter[6].i = channels;
            constants_iter[7].i = sum_workspace_reduced.cstep;

            dispatcher.w = reduced_w;

            const Pipeline* pipeline_reduce_iter = elempack == 4 ? pipeline_layernorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_layernorm_reduce_sum4_fp32[pb % 2];
            cmd.record_pipeline(pipeline_reduce_iter, bindings_iter, constants_iter, dispatcher);
            pb++;
            sqsum_workspace = sum_workspace_reduced;
        }

        std::vector<VkMat> var_bindings(2);
        var_bindings[0] = sqsum_workspace;
        var_bindings[1] = var_workspace;
        std::vector<vk_constant_type> var_constants(5);
        var_constants[0].i = sqsum_workspace.w;
        var_constants[1].i = num_groups_per_channel;
        var_constants[2].i = channels;
        var_constants[3].i = sqsum_workspace.cstep;
        var_constants[4].f = (float)group_size;

        dispatcher.w = 1;

        const Pipeline* pipeline_reduce_mean = elempack == 4 ? pipeline_layernorm_reduce_mean_pack4 : pipeline_layernorm_reduce_mean;
        cmd.record_pipeline(pipeline_reduce_mean, var_bindings, var_constants, dispatcher);
    }

    // coeffs a and b ---
    // coeffs_workspace stores {a, b} for each group, so size is num_groups_total * 2
    VkMat coeffs_workspace(num_groups_total * 2, elemsize, elempack, opt.workspace_vkallocator);
    {
        std::vector<VkMat> coeff_bindings(3);
        coeff_bindings[0] = coeffs_workspace;
        coeff_bindings[1] = mean_workspace;
        coeff_bindings[2] = var_workspace;

        std::vector<vk_constant_type> coeff_constants(2);
        coeff_constants[0].i = num_groups_per_channel;
        coeff_constants[1].i = channels;

        VkMat dispatcher_coeffs;
        dispatcher_coeffs.w = 1;
        dispatcher_coeffs.h = num_groups_per_channel;
        dispatcher_coeffs.c = channels;

        const Pipeline* pipeline_coeffs = elempack == 4 ? pipeline_layernorm_coeffs_pack4 : pipeline_layernorm_coeffs;
        cmd.record_pipeline(pipeline_coeffs, coeff_bindings, coeff_constants, dispatcher_coeffs);
    }

    // apply norm
    {
        std::vector<VkMat> norm_bindings(4);
        norm_bindings[0] = bottom_top_blob;
        norm_bindings[1] = coeffs_workspace;
        norm_bindings[2] = gamma_data_gpu;
        norm_bindings[3] = beta_data_gpu;

        std::vector<vk_constant_type> norm_constants(4);
        norm_constants[0].i = w;
        norm_constants[1].i = h;
        norm_constants[2].i = channels;
        norm_constants[3].i = cstep;

        const Pipeline* pipeline_norm = elempack == 4 ? pipeline_layernorm_norm_pack4 : pipeline_layernorm_norm;

        VkMat dispatcher;
        dispatcher.w = w;
        dispatcher.h = h;
        dispatcher.c = channels;

        cmd.record_pipeline(pipeline_norm, norm_bindings, norm_constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
