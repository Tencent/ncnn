// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution1d_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Deconvolution1D_vulkan::Deconvolution1D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_deconvolution1d = 0;

    pipeline_deconvolution1d_gemm = 0;
    pipeline_deconvolution1d_col2im = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
}

int Deconvolution1D_vulkan::load_param(const ParamDict& pd)
{
    int ret = Deconvolution1D::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int Deconvolution1D_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int maxk = kernel_w;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int outc_pack4 = num_output_packed / 4;

    // input is 2D (w, num_input/elempack) with rows contiguous,
    // reinterpret as 3D (w, 1, num_input/elempack) with channel step = w
    Mat shape_3d;
    if (shape.dims != 0)
    {
        shape_3d.dims = 3;
        shape_3d.w = shape.w;
        shape_3d.h = 1;
        shape_3d.c = shape.h;
        shape_3d.elemsize = shape.elemsize;
        shape_3d.elempack = shape.elempack;
        shape_3d.cstep = shape.w;
    }

    // gemm path output col shape (w, 1, maxk*num_output/out_elempack)
    Mat out_shape_col;
    if (shape.dims != 0 && out_shape.dims != 0)
    {
        out_shape_col = Mat(shape.w, 1, maxk * num_output / out_elempack, (void*)0, out_shape.elemsize, out_elempack);
    }

    if (opt.use_sgemm_convolution && num_input >= 8 && maxk * num_output >= 8)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (shape.dims != 0)
                size = shape.w;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, maxk * num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((maxk * num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((maxk * num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            const int blocks_n = (maxk * num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;

            Mat weight_data_r2;

            if (out_elempack == 4)
            {
                // from maxk-inch-outch to inch-4*maxk-outch/4
                weight_data_r2.create(num_input * 4 * maxk * (num_output / 4));
                for (int i = 0; i < num_output / 4; i++)
                {
                    for (int j = 0; j < maxk; j++)
                    {
                        for (int ii = 0; ii < 4; ii++)
                        {
                            for (int k = 0; k < num_input; k++)
                            {
                                weight_data_r2[((i * maxk + j) * 4 + ii) * num_input + k] = weight_data[((i * 4 + ii) * num_input + k) * maxk + j];
                            }
                        }
                    }
                }
            }
            else
            {
                // from maxk-inch-outch to inch-maxk-outch
                weight_data_r2.create(num_input * maxk * num_output);
                for (int i = 0; i < num_output; i++)
                {
                    for (int j = 0; j < maxk; j++)
                    {
                        for (int k = 0; k < num_input; k++)
                        {
                            weight_data_r2[(i * maxk + j) * num_input + k] = weight_data[(i * num_input + k) * maxk + j];
                        }
                    }
                }
            }

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < maxk * num_output && gki < num_input)
                                        {
                                            *p++ = weight_data_r2[gni * num_input + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    // const int ki = k * coopmat_K;

                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        // for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        // const int gki = k * coopmat_K + i;
                                        const int gki = k * coopmat_K + i;

                                        if (gni < maxk * num_output && gki < num_input)
                                        {
                                            *p++ = weight_data_r2[gni * num_input + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(13 + 3);
            specializations[0].u32 = coopmat_M;
            specializations[1].u32 = coopmat_N;
            specializations[2].u32 = coopmat_K;
            specializations[3].u32 = coopmat_subgroup_size;
            specializations[4].u32 = UNROLL_SG_M;
            specializations[5].u32 = UNROLL_SG_N;
            specializations[6].u32 = UNROLL_SG_K;
            specializations[7].u32 = UNROLL_WG_M;
            specializations[8].u32 = UNROLL_WG_N;
            specializations[9].u32 = num_input;
            specializations[10].u32 = maxk * num_output;
            specializations[11].u32 = elempack;
            specializations[12].u32 = out_elempack;
            specializations[13 + 0].u32 = shape_3d.dims != 0 ? shape_3d.w : 0;
            specializations[13 + 1].u32 = shape_3d.dims != 0 ? shape_3d.w : 0; // cstep = w
            specializations[13 + 2].u32 = out_shape_col.cstep;

            pipeline_deconvolution1d_gemm = new Pipeline(vkdev);
            pipeline_deconvolution1d_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_deconvolution1d_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_deconvolution1d_gemm->create(LayerShaderType::deconvolution1d_gemm_cm, opt, specializations);
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            // unified pack4x4 weight layout: always group input channels by 4
            const int num_input_packed = (num_input + 3) / 4 * 4;

            weight_data_packed.create(num_input_packed / 4, maxk * num_output / out_elempack, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    float* g00 = weight_data_packed.row(q / out_elempack * maxk + k);

                    for (int p = 0; p < num_input_packed; p += 4)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                if (q + i < num_output && p + j < num_input)
                                {
                                    const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                    g00[0] = k00[k];
                                }
                                else
                                {
                                    g00[0] = 0.f;
                                }
                                g00++;
                            }
                        }
                    }
                }
            }

            const int c_packed = num_input_packed / 4;

            std::vector<vk_specialization_type> specializations(3 + 7);
            specializations[0].i = maxk;
            specializations[1].i = elempack;
            specializations[2].i = out_elempack;
            specializations[3 + 0].i = shape_3d.w;
            specializations[3 + 1].i = shape_3d.dims != 0 ? 1 : 0;
            specializations[3 + 2].i = c_packed;
            specializations[3 + 3].i = shape_3d.dims != 0 ? shape_3d.w : 0; // cstep = w
            specializations[3 + 4].i = out_shape_col.cstep;
            specializations[3 + 5].i = out_shape_col.c;
            specializations[3 + 6].i = num_input;

            Mat local_size_xyz(8, std::min(4, num_output / out_elempack), 1, (void*)0);
            if (out_shape_col.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape_col.w);
                local_size_xyz.h = std::min(4, out_shape_col.c);
            }

            pipeline_deconvolution1d_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_deconvolution1d_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_deconvolution1d_gemm->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_deconvolution1d_gemm->create(LayerShaderType::deconvolution1d_gemm_packed, opt, specializations);
        }

        {
            std::vector<vk_specialization_type> specializations(8 + 3);
            specializations[0].i = kernel_w;
            specializations[1].i = dilation_w;
            specializations[2].i = stride_w;
            specializations[3].i = bias_term;
            specializations[4].i = activation_type;
            specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[7].i = num_output / out_elempack;
            specializations[8 + 0].i = shape_3d.w;
            specializations[8 + 1].i = out_shape_col.cstep;
            specializations[8 + 2].i = out_shape.w;

            Mat local_size_xyz(8, std::min(4, num_output / out_elempack), 1, (void*)0);
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape.w);
                local_size_xyz.h = std::min(4, num_output / out_elempack);
            }

            int shader_type_index = -1;
            if (out_elempack == 1) shader_type_index = LayerShaderType::deconvolution1d_col2im;
            if (out_elempack == 4) shader_type_index = LayerShaderType::deconvolution1d_pack4_col2im;

            pipeline_deconvolution1d_col2im = new Pipeline(vkdev);
            pipeline_deconvolution1d_col2im->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_deconvolution1d_col2im->create(shader_type_index, opt, specializations);
        }

        if (opt.lightmode)
        {
            weight_data.release();
        }

        return 0;
    }

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    // unified pack4 weight layout: output channels always packed by 4
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output_packed / 4, (size_t)4 * 4 * elempack, 4 * elempack);

        for (int q = 0; q < num_output_packed; q += 4)
        {
            float* g00 = weight_data_packed.channel(q / 4);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < elempack; j++)
                        {
                            if (q + i < num_output)
                            {
                                const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                g00[0] = k00[k];
                            }
                            else
                            {
                                g00[0] = 0.f;
                            }
                            g00++;
                        }
                    }
                }
            }
        }
    }

    std::vector<vk_specialization_type> specializations(10 + 4);
    specializations[0].i = kernel_w;
    specializations[1].i = dilation_w;
    specializations[2].i = stride_w;
    specializations[3].i = bias_term;
    specializations[4].i = activation_type;
    specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[7].i = elempack;
    specializations[8].i = out_elempack;
    specializations[9].i = num_output;
    specializations[10 + 0].i = shape_3d.w;
    specializations[10 + 1].i = num_input / elempack;
    specializations[10 + 2].i = out_shape.w;
    specializations[10 + 3].i = outc_pack4;

    Mat local_size_xyz(std::min(8, out_shape.dims != 0 ? out_shape.w : 8), std::min(4, (outc_pack4 + 1) / 2), 1, (void*)0);

    pipeline_deconvolution1d = new Pipeline(vkdev);
    pipeline_deconvolution1d->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_deconvolution1d->create(LayerShaderType::deconvolution1d, opt, specializations);

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Deconvolution1D_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_deconvolution1d;
    pipeline_deconvolution1d = 0;

    delete pipeline_deconvolution1d_gemm;
    pipeline_deconvolution1d_gemm = 0;

    delete pipeline_deconvolution1d_col2im;
    pipeline_deconvolution1d_col2im = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int Deconvolution1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    weight_data_packed.release();

    if (bias_term)
    {
        // pad bias to multiple of 4 for the unified packed shader
        const int num_output_packed = (num_output + 3) / 4 * 4;
        Mat bias_data_packed(num_output_packed, (size_t)4u, 1);
        float* bias_ptr = bias_data_packed;
        for (int i = 0; i < num_output; i++)
        {
            bias_ptr[i] = bias_data[i];
        }
        for (int i = num_output; i < num_output_packed; i++)
        {
            bias_ptr[i] = 0.f;
        }

        cmd.record_upload(bias_data_packed, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int Deconvolution1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int num_input = h * elempack;
    const int maxk = kernel_w;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    const int outw_full = (w - 1) * stride_w + kernel_extent_w + output_pad_right;

    int front_cut = 0;
    int outw = outw_full;
    if (pad_left > 0 || pad_right > 0)
    {
        front_cut = pad_left;
        outw = outw_full - pad_left - pad_right;
    }
    else if (output_w > 0)
    {
        const int wcut = outw_full - output_w;

        if (pad_left == -233 || pad_right == -233)
        {
            // onnx padding=SAME_UPPER
            front_cut = wcut / 2;
        }
        else
        {
            // onnx padding=SAME_LOWER
            front_cut = wcut - wcut / 2;
        }

        outw = output_w;
    }

    const int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_sgemm_convolution && num_input >= 8 && maxk * num_output >= 8)
    {
        // gemm + col2im


        // reinterpret 2D input as 3D with channel step = w (rows are contiguous)
        VkMat bottom_blob_3d = bottom_blob;
        bottom_blob_3d.dims = 3;
        bottom_blob_3d.c = h;
        bottom_blob_3d.h = 1;
        bottom_blob_3d.cstep = w;

        VkMat top_blob_col;
        top_blob_col.create(w, 1, maxk * num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
        if (top_blob_col.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            const int size = w;

            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_3d;
            bindings[1] = top_blob_col;
            bindings[2] = weight_data_gpu;
            bindings[3] = bottom_blob_3d; // scalar view for elempack == 1

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = size;
            constants[1].u32 = w; // cstep = w
            constants[2].u32 = top_blob_col.cstep;

            const int blocks_x = (size + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (maxk * num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_deconvolution1d_gemm, bindings, constants, dispatcher);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;

            std::vector<VkMat> bindings(5);
            bindings[0] = bottom_blob_3d;
            bindings[1] = top_blob_col;
            bindings[2] = bottom_blob_3d;
            bindings[3] = top_blob_col;
            bindings[4] = weight_data_gpu;

            std::vector<vk_constant_type> constants(7);
            constants[0].i = w;
            constants[1].i = 1;
            constants[2].i = num_input_packed / 4;
            constants[3].i = w; // cstep = w
            constants[4].i = top_blob_col.cstep;
            constants[5].i = top_blob_col.c;
            constants[6].i = num_input;

            VkMat dispatcher;
            dispatcher.w = (top_blob_col.cstep + 3) / 4;
            dispatcher.h = top_blob_col.c;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_deconvolution1d_gemm, bindings, constants, dispatcher);
        }

        // col2im
        top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = top_blob_col;
        bindings[1] = top_blob;
        bindings[2] = bias_data_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = top_blob_col.cstep;
        constants[2].i = outw;
        constants[3].i = front_cut;

        VkMat dispatcher;
        dispatcher.w = outw;
        dispatcher.h = num_output / out_elempack;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_deconvolution1d_col2im, bindings, constants, dispatcher);

        return 0;
    }

    // direct
    top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int outc_pack4 = num_output_packed / 4;

    std::vector<VkMat> bindings(6);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob;
    bindings[3] = top_blob;
    bindings[4] = weight_data_gpu;
    bindings[5] = bias_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = w;
    constants[1].i = num_input / elempack;
    constants[2].i = outw;
    constants[3].i = outc_pack4;
    constants[4].i = front_cut;

    VkMat dispatcher;
    dispatcher.w = (outw + 1) / 2;
    dispatcher.h = (outc_pack4 + 1) / 2;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_deconvolution1d, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
