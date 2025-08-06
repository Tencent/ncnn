// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_vulkan.h"
#include "layer_shader_type.h"

namespace ncnn {

LayerNorm_vulkan::LayerNorm_vulkan()
{
    support_vulkan = true;

    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = 0;
    pipeline_layernorm_reduce_sum4_fp32[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32[1] = 0;
    pipeline_layernorm_reduce_mean = 0;
    pipeline_layernorm_sub_mean_square = 0;
    pipeline_layernorm_coeffs = 0;
    pipeline_layernorm_norm = 0;
}

int LayerNorm_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> no_specializations;

    // Generic reduction pipelines
    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32, opt, no_specializations);

    pipeline_layernorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp32[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, no_specializations);
    pipeline_layernorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp32[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, no_specializations);

    pipeline_layernorm_reduce_mean = new Pipeline(vkdev);
    pipeline_layernorm_reduce_mean->create(LayerShaderType::layernorm_reduce_mean, opt, no_specializations);

    // Layer-specific pipelines
    pipeline_layernorm_sub_mean_square = new Pipeline(vkdev);
    pipeline_layernorm_sub_mean_square->create(LayerShaderType::layernorm_sub_mean_square, opt, no_specializations);

    std::vector<vk_specialization_type> coeffs_specializations(1);
    coeffs_specializations[0].f = eps;
    pipeline_layernorm_coeffs = new Pipeline(vkdev);
    pipeline_layernorm_coeffs->create(LayerShaderType::layernorm_coeffs, opt, coeffs_specializations);

    std::vector<vk_specialization_type> norm_specializations(1);
    norm_specializations[0].i = affine;
    pipeline_layernorm_norm = new Pipeline(vkdev);
    pipeline_layernorm_norm->create(LayerShaderType::layernorm_norm, opt, norm_specializations);

    return 0;
}

int LayerNorm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_layernorm_reduce_sum4_fp16_to_fp32;
    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = 0;

    delete pipeline_layernorm_reduce_sum4_fp32[0];
    delete pipeline_layernorm_reduce_sum4_fp32[1];
    pipeline_layernorm_reduce_sum4_fp32[0] = 0;
    pipeline_layernorm_reduce_sum4_fp32[1] = 0;

    delete pipeline_layernorm_reduce_mean;
    pipeline_layernorm_reduce_mean = 0;

    delete pipeline_layernorm_sub_mean_square;
    pipeline_layernorm_sub_mean_square = 0;

    delete pipeline_layernorm_coeffs;
    pipeline_layernorm_coeffs = 0;

    delete pipeline_layernorm_norm;
    pipeline_layernorm_norm = 0;

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
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int dims = bottom_top_blob.dims;

    if (affine_size == 0)
        return 0;

    // Determine normalization geometry based on dims and affine_size
    int group_size;
    int num_groups;
    if (dims == 1)
    {
        group_size = w;
        num_groups = 1;
    }
    else if (dims == 2)
    {
        group_size = w;
        num_groups = h;
    }
    else
    {   // dims == 3
        if (affine_size == w)
        {
            group_size = w;
            num_groups = channels * h;
        }
        else
        {   // affine_size == w * h
            group_size = w * h;
            num_groups = channels;
        }
    }

    VkMat mean_workspace(num_groups, 4u, 1, opt.workspace_vkallocator);
    VkMat var_workspace(num_groups, 4u, 1, opt.workspace_vkallocator);

    // --- 1. CALCULATE MEAN ---
    {
        VkMat blob_reshaped = bottom_top_blob;
        blob_reshaped.w = group_size;
        blob_reshaped.h = 1;
        blob_reshaped.c = num_groups;
        blob_reshaped.cstep = group_size;

        VkMat sum_workspace;
        int reduced_w = (blob_reshaped.w + 3) / 4;
        sum_workspace.create(reduced_w, 1, num_groups, 4u, 1, opt.workspace_vkallocator);

        std::vector<VkMat> bindings(2);
        bindings[0] = blob_reshaped;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = blob_reshaped.w;
        constants[1].i = 1;
        constants[2].i = num_groups;
        constants[3].i = blob_reshaped.cstep;
        constants[4].i = reduced_w;
        constants[5].i = 1;
        constants[6].i = num_groups;
        constants[7].i = reduced_w;

        cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp16_to_fp32, bindings, constants, sum_workspace);

        int pb = 0;
        while (sum_workspace.w > 4)
        {
            reduced_w = (sum_workspace.w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, 1, num_groups, 4u, 1, opt.workspace_vkallocator);

            std::vector<VkMat> bindings_iter(2);
            bindings_iter[0] = sum_workspace;
            bindings_iter[1] = sum_workspace_reduced;

            std::vector<vk_constant_type> constants_iter(8);
            constants_iter[0].i = sum_workspace.w;
            constants_iter[1].i = 1;
            constants_iter[2].i = num_groups;
            constants_iter[3].i = sum_workspace.w;
            constants_iter[4].i = reduced_w;
            constants_iter[5].i = 1;
            constants_iter[6].i = num_groups;
            constants_iter[7].i = reduced_w;

            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[pb % 2], bindings_iter, constants_iter, sum_workspace_reduced);
            pb++;
            sum_workspace = sum_workspace_reduced;
        }

        std::vector<VkMat> mean_bindings(2);
        mean_bindings[0] = sum_workspace;
        mean_bindings[1] = mean_workspace;

        std::vector<vk_constant_type> mean_constants(5);
        mean_constants[0].i = sum_workspace.w;
        mean_constants[1].i = 1;
        mean_constants[2].i = num_groups;
        mean_constants[3].i = sum_workspace.w;
        mean_constants[4].f = (float)group_size;

        cmd.record_pipeline(pipeline_layernorm_reduce_mean, mean_bindings, mean_constants, mean_workspace);
    }

    // --- 2. CALCULATE VARIANCE ---
    {
        VkMat square_workspace;
        square_workspace.create(w, h, channels, 4u, 1, opt.workspace_vkallocator);

        std::vector<VkMat> sq_bindings(3);
        sq_bindings[0] = bottom_top_blob;
        sq_bindings[1] = mean_workspace;
        sq_bindings[2] = square_workspace;

        std::vector<vk_constant_type> sq_constants(5);
        sq_constants[0].i = w;
        sq_constants[1].i = h;
        sq_constants[2].i = channels;
        sq_constants[3].i = bottom_top_blob.cstep;
        sq_constants[4].i = affine_size;

        cmd.record_pipeline(pipeline_layernorm_sub_mean_square, sq_bindings, sq_constants, square_workspace);

        VkMat square_reshaped = square_workspace;
        square_reshaped.w = group_size;
        square_reshaped.h = 1;
        square_reshaped.c = num_groups;
        square_reshaped.cstep = group_size;

        VkMat sqsum_workspace;
        int reduced_w = (square_reshaped.w + 3) / 4;
        sqsum_workspace.create(reduced_w, 1, num_groups, 4u, 1, opt.workspace_vkallocator);

        std::vector<VkMat> bindings(2);
        bindings[0] = square_reshaped;
        bindings[1] = sqsum_workspace;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = square_reshaped.w;
        constants[1].i = 1;
        constants[2].i = num_groups;
        constants[3].i = square_reshaped.cstep;
        constants[4].i = reduced_w;
        constants[5].i = 1;
        constants[6].i = num_groups;
        constants[7].i = reduced_w;

        cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[0], bindings, constants, sqsum_workspace);

        int pb = 0;
        while (sqsum_workspace.w > 4)
        {
            reduced_w = (sqsum_workspace.w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, 1, num_groups, 4u, 1, opt.workspace_vkallocator);

            std::vector<VkMat> bindings_iter(2);
            bindings_iter[0] = sqsum_workspace;
            bindings_iter[1] = sum_workspace_reduced;

            std::vector<vk_constant_type> constants_iter(8);
            constants_iter[0].i = sqsum_workspace.w;
            constants_iter[1].i = 1;
            constants_iter[2].i = num_groups;
            constants_iter[3].i = sqsum_workspace.w;
            constants_iter[4].i = reduced_w;
            constants_iter[5].i = 1;
            constants_iter[6].i = num_groups;
            constants_iter[7].i = reduced_w;

            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[(pb + 1) % 2], bindings_iter, constants_iter, sum_workspace_reduced);
            pb++;
            sqsum_workspace = sum_workspace_reduced;
        }

        std::vector<VkMat> var_bindings(2);
        var_bindings[0] = sqsum_workspace;
        var_bindings[1] = var_workspace;

        std::vector<vk_constant_type> var_constants(5);
        var_constants[0].i = sqsum_workspace.w;
        var_constants[1].i = 1;
        var_constants[2].i = num_groups;
        var_constants[3].i = sqsum_workspace.w;
        var_constants[4].f = (float)group_size;

        cmd.record_pipeline(pipeline_layernorm_reduce_mean, var_bindings, var_constants, var_workspace);
    }

    // --- 3. CALCULATE COEFFICIENTS ---
    VkMat coeffs_workspace;
    coeffs_workspace.create(num_groups * 2, 2u, 1, opt.workspace_vkallocator);

    std::vector<VkMat> coeff_bindings(3);
    coeff_bindings[0] = coeffs_workspace;
    coeff_bindings[1] = mean_workspace;
    coeff_bindings[2] = var_workspace;

    std::vector<vk_constant_type> coeff_constants(1);
    coeff_constants[0].i = num_groups;

    cmd.record_pipeline(pipeline_layernorm_coeffs, coeff_bindings, coeff_constants, coeffs_workspace);

    // --- 4. APPLY NORMALIZATION ---
    std::vector<VkMat> norm_bindings(4);
    norm_bindings[0] = bottom_top_blob;
    norm_bindings[1] = coeffs_workspace;
    norm_bindings[2] = gamma_data_gpu;
    norm_bindings[3] = beta_data_gpu;

    std::vector<vk_constant_type> norm_constants(5);
    norm_constants[0].i = w;
    norm_constants[1].i = h;
    norm_constants[2].i = channels;
    norm_constants[3].i = bottom_top_blob.cstep;
    norm_constants[4].i = affine_size;

    cmd.record_pipeline(pipeline_layernorm_norm, norm_bindings, norm_constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn