// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

InstanceNorm_vulkan::InstanceNorm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

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
}

int InstanceNorm_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int _channels = channels;
    if (shape.dims == 3) _channels = shape.c;

    int elempack = 1;
    if (_channels != 0) elempack = _channels % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    // TODO resolve workspace_shape.w
    Mat workspace_shape_packed;
    if (_channels != 0) workspace_shape_packed = Mat(1, 1, _channels / elempack, (void*)0, elemsize, elempack);

    {
        Mat local_size_xyz;
        {
            local_size_xyz = Mat(16, 1, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
            if (workspace_shape_packed.dims != 0)
            {
                local_size_xyz.w = 16;
                local_size_xyz.h = 1;
                local_size_xyz.c = std::min(4, workspace_shape_packed.c);
            }
        }

        // pack1
        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32->create(LayerShaderType::instancenorm_reduce_sum4_fp16_to_fp32, opt, std::vector<vk_specialization_type>());

            pipeline_instancenorm_reduce_sum4_fp32[0] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32[0]->create(LayerShaderType::instancenorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
            pipeline_instancenorm_reduce_sum4_fp32[1] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32[1]->create(LayerShaderType::instancenorm_reduce_sum4_fp32, opt, std::vector<vk_specialization_type>());
        }

        // pack4
        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4->create(LayerShaderType::instancenorm_reduce_sum4_fp16_to_fp32_pack4, opt, std::vector<vk_specialization_type>());

            pipeline_instancenorm_reduce_sum4_fp32_pack4[0] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[0]->create(LayerShaderType::instancenorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1] = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_sum4_fp32_pack4[1]->create(LayerShaderType::instancenorm_reduce_sum4_fp32_pack4, opt, std::vector<vk_specialization_type>());
        }
    }

    {
        std::vector<vk_specialization_type> specializations(0 + 4);
        specializations[0].i = 0; // TODO resolve workspace_shape_packed.w;
        specializations[1].i = 0; // TODO resolve workspace_shape_packed.h;
        specializations[2].i = workspace_shape_packed.c;
        specializations[3].i = 0; // TODO resolve workspace_shape_packed.cstep;

        Mat local_size_xyz(_channels ? std::min(64, _channels / elempack) : 64, 1, 1, (void*)0);
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.c);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_reduce_mean = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_mean->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_mean->create(LayerShaderType::instancenorm_reduce_mean, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_reduce_mean_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_reduce_mean_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_reduce_mean_pack4->create(LayerShaderType::instancenorm_reduce_mean_pack4, opt, specializations);
        }
    }

    Mat square_workspace_packed;
    if (shape.dims == 3) square_workspace_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elempack * 4u, elempack);

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

        Mat local_size_xyz(4, 4, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
        if (square_workspace_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, square_workspace_packed.w);
            local_size_xyz.h = std::min(4, square_workspace_packed.h);
            local_size_xyz.c = std::min(4, square_workspace_packed.c);
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_sub_mean_square = new Pipeline(vkdev);
            pipeline_instancenorm_sub_mean_square->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_sub_mean_square->create(LayerShaderType::instancenorm_sub_mean_square, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_sub_mean_square_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_sub_mean_square_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_sub_mean_square_pack4->create(LayerShaderType::instancenorm_sub_mean_square_pack4, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(3);
        specializations[0].f = eps;
        specializations[1].i = affine;
        specializations[2].i = _channels / elempack;

        Mat local_size_xyz(_channels ? std::min(64, _channels / elempack) : 64, 1, 1, (void*)0);
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.c);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_coeffs = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs->create(LayerShaderType::instancenorm_coeffs, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_coeffs_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_coeffs_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_coeffs_pack4->create(LayerShaderType::instancenorm_coeffs_pack4, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(0 + 5);
        specializations[0 + 0].i = shape_packed.dims;
        specializations[0 + 1].i = shape_packed.w;
        specializations[0 + 2].i = shape_packed.h;
        specializations[0 + 3].i = shape_packed.c;
        specializations[0 + 4].i = shape_packed.cstep;

        Mat local_size_xyz(4, 4, _channels ? std::min(4, _channels / elempack) : 4, (void*)0);
        if (shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, shape_packed.w);
            local_size_xyz.h = std::min(4, shape_packed.h);
            local_size_xyz.c = std::min(4, shape_packed.c);
        }

        if (elempack == 1 || _channels == 0)
        {
            pipeline_instancenorm_norm = new Pipeline(vkdev);
            pipeline_instancenorm_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm->create(LayerShaderType::instancenorm_norm, opt, specializations);
        }

        if (elempack == 4 || _channels == 0)
        {
            pipeline_instancenorm_norm_pack4 = new Pipeline(vkdev);
            pipeline_instancenorm_norm_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_instancenorm_norm_pack4->create(LayerShaderType::instancenorm_norm_pack4, opt, specializations);
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

    return 0;
}

int InstanceNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (affine == 0)
        return 0;

    cmd.record_upload(gamma_data, gamma_data_gpu, opt);

    cmd.record_upload(beta_data, beta_data_gpu, opt);

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
    VkMat mean_workspace(c, 4u * elempack, elempack, opt.workspace_vkallocator);
    {
        // reduce sum
        VkMat sum_workspace;
        {
            int reduced_w = (bottom_top_blob.w * bottom_top_blob.h + 3) / 4;
            int reduced_h = 1;
            int reduced_c = bottom_top_blob.c;

            sum_workspace.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);
            {
                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_top_blob;
                bindings[1] = sum_workspace;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = bottom_top_blob.w * bottom_top_blob.h;
                constants[1].i = 1;
                constants[2].i = bottom_top_blob.c;
                constants[3].i = bottom_top_blob.cstep;
                constants[4].i = sum_workspace.w;
                constants[5].i = 1;
                constants[6].i = sum_workspace.c;
                constants[7].i = sum_workspace.cstep;

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4 : pipeline_instancenorm_reduce_sum4_fp16_to_fp32;

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
            sum_workspace_reduced.create(reduced_w, reduced_h, reduced_c, 4u * elempack, elempack, opt.workspace_vkallocator);

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = sum_workspace;
                bindings[1] = sum_workspace_reduced;

                std::vector<vk_constant_type> constants(8);
                constants[0].i = sum_workspace.w;
                constants[1].i = 1;
                constants[2].i = sum_workspace.c;
                constants[3].i = sum_workspace.cstep;
                constants[4].i = sum_workspace_reduced.w;
                constants[5].i = 1;
                constants[6].i = sum_workspace_reduced.c;
                constants[7].i = sum_workspace_reduced.cstep;

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_instancenorm_reduce_sum4_fp32[pb % 2];

                cmd.record_pipeline(pipeline, bindings, constants, sum_workspace_reduced);

                pb++;
            }

            sum_workspace = sum_workspace_reduced;
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = sum_workspace;
            bindings[1] = mean_workspace;

            std::vector<vk_constant_type> constants(5);
            constants[0].i = sum_workspace.w;
            constants[1].i = 1;
            constants[2].i = sum_workspace.c;
            constants[3].i = sum_workspace.cstep;
            constants[4].f = size;

            const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4 : pipeline_instancenorm_reduce_mean;

            cmd.record_pipeline(pipeline, bindings, constants, mean_workspace);
        }
    }

    // var
    VkMat var_workspace(c, 4u * elempack, elempack, opt.workspace_vkallocator);
    {
        // sub mean and square
        VkMat square_workspace;
        square_workspace.create(w, h, c, 4u * elempack, elempack, opt.workspace_vkallocator);
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

            const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_sub_mean_square_pack4 : pipeline_instancenorm_sub_mean_square;

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

                const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_sum4_fp32_pack4[pb % 2] : pipeline_instancenorm_reduce_sum4_fp32[pb % 2];

                cmd.record_pipeline(pipeline, bindings, constants, sqsum_workspace_reduced);

                pb++;
            }

            sqsum_workspace = sqsum_workspace_reduced;
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = sqsum_workspace;
            bindings[1] = var_workspace;

            std::vector<vk_constant_type> constants(5);
            constants[0].i = sqsum_workspace.w;
            constants[1].i = 1;
            constants[2].i = sqsum_workspace.c;
            constants[3].i = sqsum_workspace.cstep;
            constants[4].f = size;

            const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_reduce_mean_pack4 : pipeline_instancenorm_reduce_mean;

            cmd.record_pipeline(pipeline, bindings, constants, var_workspace);
        }
    }

    // coeffs
    VkMat coeffs_workspace;
    coeffs_workspace.create(c, elemsize * 2, elempack * 2, opt.workspace_vkallocator);
    {
        std::vector<VkMat> bindings(5);
        bindings[0] = coeffs_workspace;
        bindings[1] = mean_workspace;
        bindings[2] = var_workspace;
        bindings[3] = gamma_data_gpu;
        bindings[4] = beta_data_gpu;

        std::vector<vk_constant_type> constants(1);
        constants[0].i = c;

        const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_coeffs_pack4 : pipeline_instancenorm_coeffs;

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

        const Pipeline* pipeline = elempack == 4 ? pipeline_instancenorm_norm_pack4 : pipeline_instancenorm_norm;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
