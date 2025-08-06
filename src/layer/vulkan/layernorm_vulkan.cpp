// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_vulkan.h"
#include "layer_shader_type.h"
#include "command.h" // For VkCompute
#include <stdio.h>   // For printf
#include <algorithm> // For std::min

namespace ncnn {

// =================================================================================================
// DEBUG HELPER FUNCTION
// This function downloads a VkMat from GPU to CPU and prints its contents.
// WARNING: This is extremely slow and should only be used for debugging.
// =================================================================================================
static void print_vkmat(const VkMat& m, const char* name, VkCompute& cmd, const Option& opt)
{
    if (m.empty())
    {
        printf("--- %s ---\n", name);
        printf("VkMat is empty.\n\n");
        return;
    }

    // Create a CPU Mat with a CPU-compatible allocator to be the destination for the download.
    Mat staging_mat;
    staging_mat.create_like(m, opt.blob_allocator);
    if (staging_mat.empty())
    {
        NCNN_LOGE("print_vkmat failed to create staging_mat");
        return;
    }

    // Record the download command from the GPU VkMat to the CPU Mat.
    cmd.record_download(m, staging_mat, opt);

    // Submit and wait for the command to finish.
    // This is a blocking call, ensuring data is ready on the CPU side.
    cmd.submit_and_wait();

    cmd.reset();

    Mat cpu_mat;
    convert_packing(staging_mat,cpu_mat,1);

    printf("--- %s ---\n", name);
    printf("Dims: %d, w: %d, h: %d, d: %d, c: %d, cstep: %zu, elemsize: %zu, elempack: %d\n",
           m.dims, m.w, m.h, m.d, m.c, m.cstep, m.elemsize, m.elempack);
    printf("CPU Dims: %d, w: %d, h: %d, d: %d, c: %d, cstep: %zu, elemsize: %zu, elempack: %d\n",
           cpu_mat.dims, cpu_mat.w, cpu_mat.h, cpu_mat.d, cpu_mat.c, cpu_mat.cstep, cpu_mat.elemsize, cpu_mat.elempack);

    if (cpu_mat.elemsize == 4u) // float32
    {
        const float* ptr = cpu_mat;
        for (int i = 0; i < cpu_mat.c; i++)
        {
            printf("cpu_mat[%d]: \n", i);
            // 打印矩阵
            for (int j = 0; j< cpu_mat.h; j++)
            {
                for (int k = 0; k< cpu_mat.w;k++)
                {
                    printf("%f ", ptr[i * cpu_mat.cstep + j * cpu_mat.w + k]);
                }
                printf("\n");
            }
        }
    }
    else if (cpu_mat.elemsize == 2u) // float16 or bfloat16
    {
        const unsigned short* ptr = cpu_mat;
        for (int i = 0; i < cpu_mat.c; i++)
        {
            printf("cpu_mat[%d]: \n", i);
            // 打印矩阵
            for (int j = 0; j< cpu_mat.h; j++)
            {
                for (int k = 0; k< cpu_mat.w;k++)
                {
                    printf("%f ", ncnn::float16_to_float32(ptr[i * cpu_mat.cstep + j * cpu_mat.w + k]));
                }
                printf("\n");
            }
        }

    }
    else if (cpu_mat.elemsize == 1u) // int8
    {
        const signed char* ptr = cpu_mat;
        for (int i = 0; i < cpu_mat.c; i++)
        {
            printf("cpu_mat[%d]: \n", i);
            // 打印矩阵
            for (int j = 0; j< cpu_mat.h; j++)
            {
                for (int k = 0; k< cpu_mat.w;k++)
                {
                    printf("%d ", ptr[i * cpu_mat.cstep + j * cpu_mat.w + k]);
                }
                printf("\n");
            }
        }
    }
    printf("\n\n");
}

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

    pipeline_layernorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::layernorm_reduce_sum4_fp16_to_fp32, opt, no_specializations);

    pipeline_layernorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp32[0]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, no_specializations);
    pipeline_layernorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
    pipeline_layernorm_reduce_sum4_fp32[1]->create(LayerShaderType::layernorm_reduce_sum4_fp32, opt, no_specializations);

    pipeline_layernorm_reduce_mean = new Pipeline(vkdev);
    pipeline_layernorm_reduce_mean->create(LayerShaderType::layernorm_reduce_mean, opt, no_specializations);

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
    int cstep = bottom_top_blob.cstep;
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;

    if (affine_size == 0)
        return 0;

    // ================== DEBUG PRINT ==================
    print_vkmat(bottom_top_blob, "===> INPUT to LayerNorm <===", cmd, opt);
    // ===============================================

    int group_size;
    int num_groups_per_channel;
    if (dims == 1) {
        group_size = w;
        num_groups_per_channel = 1;
        channels = 1;
    } else if (dims == 2) {
        group_size = w;
        num_groups_per_channel = h;
        channels = 1;
    } else { // dims == 3
        if (affine_size == w) {
            group_size = w;
            num_groups_per_channel = h;
        } else { // affine_size == w * h
            group_size = w * h;
            num_groups_per_channel = 1;
        }
    }
    int num_groups_total = num_groups_per_channel * channels;

    VkMat mean_workspace;
    mean_workspace.create(num_groups_total, 4u, 1, opt.workspace_vkallocator);
    VkMat var_workspace;
    var_workspace.create(num_groups_total, 4u, 1, opt.workspace_vkallocator);

    // --- 1. CALCULATE MEAN ---
    {
        int reduced_w = (group_size + 3) / 4;
        VkMat sum_workspace;
        sum_workspace.create(reduced_w, num_groups_per_channel, channels, 4u, 1, opt.workspace_vkallocator);

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

        int pb = 0;
        if (elemsize == 4u) {
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[pb % 2], bindings, constants, dispatcher);
        } else {
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp16_to_fp32, bindings, constants, dispatcher);
        }
        pb++;

        // ================== DEBUG PRINT ==================
        print_vkmat(sum_workspace, "1. MEAN: After Initial Reduce", cmd, opt);
        // ===============================================

        while (sum_workspace.w > 1) {
            int current_w = sum_workspace.w;
            reduced_w = (current_w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, num_groups_per_channel, channels, 4u, 1, opt.workspace_vkallocator);

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
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[pb % 2], bindings_iter, constants_iter, dispatcher);
            pb++;
            sum_workspace = sum_workspace_reduced;

            // ================== DEBUG PRINT ==================
            char msg[100];
            sprintf(msg, "1. MEAN: Iterative Reduce Output (w=%d)", sum_workspace.w);
            print_vkmat(sum_workspace, msg, cmd, opt);
            // ===============================================
        }

        std::vector<VkMat> mean_bindings(2);
        // CRITICAL FIX: Bind separate input (sum_workspace) and output (mean_workspace) buffers.
        mean_bindings[0] = sum_workspace;
        mean_bindings[1] = mean_workspace;

        std::vector<vk_constant_type> mean_constants(5);
        mean_constants[0].i = sum_workspace.w;
        mean_constants[1].i = num_groups_per_channel;
        mean_constants[2].i = channels;
        mean_constants[3].i = sum_workspace.cstep;
        mean_constants[4].f = (float)group_size;

        dispatcher.w = 1;
        cmd.record_pipeline(pipeline_layernorm_reduce_mean, mean_bindings, mean_constants, dispatcher);

        // ================== DEBUG PRINT ==================
        print_vkmat(mean_workspace, "1. MEAN: FINAL mean_workspace", cmd, opt);
        // ===============================================
    }

    // --- 2. CALCULATE VARIANCE ---
    {
        VkMat square_workspace;
        square_workspace.create(w, h, bottom_top_blob.c, elemsize, 1, opt.workspace_vkallocator);

        std::vector<VkMat> sq_bindings(3);
        sq_bindings[0] = bottom_top_blob;
        sq_bindings[1] = mean_workspace;
        sq_bindings[2] = square_workspace;
        std::vector<vk_constant_type> sq_constants(5);
        sq_constants[0].i = w;
        sq_constants[1].i = h;
        sq_constants[2].i = bottom_top_blob.c;
        sq_constants[3].i = cstep;
        sq_constants[4].i = affine_size;
        cmd.record_pipeline(pipeline_layernorm_sub_mean_square, sq_bindings, sq_constants, square_workspace);

        // ================== DEBUG PRINT ==================
        print_vkmat(square_workspace, "2. VAR: After sub_mean_square", cmd, opt);
        // ===============================================

        int reduced_w = (group_size + 3) / 4;
        VkMat sqsum_workspace;
        sqsum_workspace.create(reduced_w, num_groups_per_channel, channels, 4u, 1, opt.workspace_vkallocator);

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

        int pb = 0;
        if (elemsize == 4u) {
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[pb % 2], bindings, constants, dispatcher);
        } else {
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp16_to_fp32, bindings, constants, dispatcher);
        }
        pb++;

        // ================== DEBUG PRINT ==================
        print_vkmat(sqsum_workspace, "2. VAR: After Initial Reduce", cmd, opt);
        // ===============================================

        while (sqsum_workspace.w > 1) {
            int current_w = sqsum_workspace.w;
            reduced_w = (current_w + 3) / 4;
            VkMat sum_workspace_reduced;
            sum_workspace_reduced.create(reduced_w, num_groups_per_channel, channels, 4u, 1, opt.workspace_vkallocator);

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
            cmd.record_pipeline(pipeline_layernorm_reduce_sum4_fp32[pb % 2], bindings_iter, constants_iter, dispatcher);
            pb++;
            sqsum_workspace = sum_workspace_reduced;

            // ================== DEBUG PRINT ==================
            char msg[100];
            sprintf(msg, "2. VAR: Iterative Reduce Output (w=%d)", sqsum_workspace.w);
            print_vkmat(sqsum_workspace, msg, cmd, opt);
            // ===============================================
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
        cmd.record_pipeline(pipeline_layernorm_reduce_mean, var_bindings, var_constants, dispatcher);

        // ================== DEBUG PRINT ==================
        print_vkmat(var_workspace, "2. VAR: FINAL var_workspace", cmd, opt);
        // ===============================================
    }

    // --- 3. CALCULATE COEFFICIENTS (a and b) ---
    VkMat coeffs_workspace;
    coeffs_workspace.create(num_groups_total * 2, elemsize, 1, opt.workspace_vkallocator);

    std::vector<VkMat> coeff_bindings(3);
    coeff_bindings[0] = coeffs_workspace;
    coeff_bindings[1] = mean_workspace;
    coeff_bindings[2] = var_workspace;

    std::vector<vk_constant_type> coeff_constants(3);
    coeff_constants[0].i = 1;
    coeff_constants[1].i = num_groups_per_channel;
    coeff_constants[2].i = channels;

    VkMat dispatcher_coeffs;
    dispatcher_coeffs.w = 1;
    dispatcher_coeffs.h = num_groups_per_channel;
    dispatcher_coeffs.c = channels;
    cmd.record_pipeline(pipeline_layernorm_coeffs, coeff_bindings, coeff_constants, dispatcher_coeffs);

    // ================== DEBUG PRINT ==================
    print_vkmat(coeffs_workspace, "3. COEFFS: After calculation", cmd, opt);
    // ===============================================

    // --- 4. APPLY NORMALIZATION ---
    std::vector<VkMat> norm_bindings(4);
    norm_bindings[0] = bottom_top_blob;
    norm_bindings[1] = coeffs_workspace;
    norm_bindings[2] = gamma_data_gpu;
    norm_bindings[3] = beta_data_gpu;
    std::vector<vk_constant_type> norm_constants(5);
    norm_constants[0].i = w;
    norm_constants[1].i = h;
    norm_constants[2].i = bottom_top_blob.c;
    norm_constants[3].i = cstep;
    norm_constants[4].i = affine_size;
    cmd.record_pipeline(pipeline_layernorm_norm, norm_bindings, norm_constants, bottom_top_blob);

    // ================== DEBUG PRINT ==================
    print_vkmat(bottom_top_blob, "===> FINAL OUTPUT of LayerNorm <===", cmd, opt);
    // ===============================================

    return 0;
}

} // namespace ncnn