// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Deconvolution_vulkan::Deconvolution_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    crop = 0;
    output_crop = 0;

    pipeline_deconvolution = 0;

    pipeline_deconvolution_gemm = 0;
    pipeline_deconvolution_col2im = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
}

int Deconvolution_vulkan::load_param(const ParamDict& pd)
{
    int ret = Deconvolution::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int Deconvolution_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // the shape before unpadding
    Mat out_shape_bordered;
    if (shape.dims != 0)
    {
        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        int outw = (shape.w - 1) * stride_w + kernel_extent_w + output_pad_right;
        int outh = (shape.h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

        out_shape_bordered = Mat(outw, outh, out_shape.c, (void*)0);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_bordered_packed;
    if (out_shape_bordered.dims == 1) out_shape_bordered_packed = Mat(out_shape_bordered.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_bordered.dims == 2) out_shape_bordered_packed = Mat(out_shape_bordered.w, out_shape_bordered.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_bordered.dims == 3) out_shape_bordered_packed = Mat(out_shape_bordered.w, out_shape_bordered.h, out_shape_bordered.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    {
        crop = ncnn::create_layer_vulkan(ncnn::LayerType::Crop);
        crop->vkdev = vkdev;

        crop->bottom_shapes.resize(1);
        crop->bottom_shapes[0] = out_shape_bordered;
        crop->top_shapes.resize(1);
        crop->top_shapes[0] = out_shape;

        ncnn::ParamDict pd;
        pd.set(0, pad_left);
        pd.set(1, pad_top);
        pd.set(2, 0);

        crop->load_param(pd);

        crop->create_pipeline(opt);
    }

    {
        output_crop = ncnn::create_layer_vulkan(ncnn::LayerType::Crop);
        output_crop->vkdev = vkdev;

        output_crop->bottom_shapes.resize(1);
        output_crop->bottom_shapes[0] = out_shape_bordered;
        output_crop->top_shapes.resize(1);
        output_crop->top_shapes[0] = out_shape;

        ncnn::ParamDict pd;
        pd.set(0, -233);
        pd.set(1, -233);
        pd.set(2, -233);

        output_crop->load_param(pd);

        output_crop->create_pipeline(opt);
    }

    if (opt.use_sgemm_convolution && num_input >= 8 && maxk * num_output >= 8)
    {
        Mat out_shape_col;
        if (shape.dims != 0 && out_shape.dims != 0)
        {
            out_shape_col = Mat(shape.w * shape.h, 1, maxk * out_shape.c, (void*)0);
        }

        Mat out_shape_col_packed;
        if (out_shape_col.dims == 3) out_shape_col_packed = Mat(out_shape_col.w, out_shape_col.h, out_shape_col.c / out_elempack, (void*)0, out_elemsize, out_elempack);

        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (shape_packed.dims == 3)
                size = shape_packed.w * shape_packed.h;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, maxk * num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((maxk * num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((maxk * num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            //        +-N-+
            //        K   |
            //        +SG_UN
            //        |   |
            //     ^  +---+
            //     |  |   |
            //   SG_UK+- -+
            //     |  |   |
            //   ^ v  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            // WG_UN  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            //   v    +---+

            //      +-N-+
            //      K   |
            //      +SG_UN
            //      |   |
            //   ^  +---+
            //   |  |   |
            // WG_UN+- -+
            //   |  |   |
            //   v  +---+

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
                                        // const int gki = (k + zk) * coopmat_K + i;
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

            std::vector<vk_specialization_type> specializations(12 + 3);
            specializations[0].u32 = coopmat_M;
            specializations[1].u32 = coopmat_N;
            specializations[2].u32 = coopmat_K;
            specializations[3].u32 = UNROLL_SG_M;
            specializations[4].u32 = UNROLL_SG_N;
            specializations[5].u32 = UNROLL_SG_K;
            specializations[6].u32 = UNROLL_WG_M;
            specializations[7].u32 = UNROLL_WG_N;
            specializations[8].u32 = num_input;
            specializations[9].u32 = maxk * num_output;
            specializations[10].u32 = elempack;
            specializations[11].u32 = out_elempack;
            specializations[12 + 0].u32 = shape_packed.w * shape_packed.h;
            specializations[12 + 1].u32 = shape_packed.cstep;
            specializations[12 + 2].u32 = out_shape_col_packed.cstep;

            const int subgroup_size = vkdev->info.subgroup_size();

            pipeline_deconvolution_gemm = new Pipeline(vkdev);
            pipeline_deconvolution_gemm->set_subgroup_size(subgroup_size);
            pipeline_deconvolution_gemm->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_deconvolution_gemm->create(LayerShaderType::deconvolution_gemm_cm, opt, specializations);
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_packed.create(num_input / elempack, maxk * num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    float* g00 = weight_data_packed.row(q / out_elempack * maxk + k);

                    for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                    {
                        for (int i = 0; i < out_elempack; i++)
                        {
                            const Mat k0 = weight_data_r2.channel(q + i);

                            for (int j = 0; j < elempack; j++)
                            {
                                const float* k00 = k0.row(p + j);
                                g00[0] = k00[k];
                                g00++;
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(1 + 6);
            specializations[0].i = maxk;
            specializations[1 + 0].i = shape_packed.w;
            specializations[1 + 1].i = shape_packed.h;
            specializations[1 + 2].i = shape_packed.c;
            specializations[1 + 3].i = shape_packed.cstep;
            specializations[1 + 4].i = out_shape_col_packed.cstep;
            specializations[1 + 5].i = out_shape_col_packed.c;

            Mat local_size_xyz(8, std::min(4, num_output / out_elempack), 1, (void*)0);
            if (out_shape_col_packed.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape_col_packed.w);
                local_size_xyz.h = std::min(4, out_shape_col_packed.c);
            }

            int shader_type_index = -1;
            if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_gemm;
            if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack4_gemm;
            if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack1to4_gemm;
            if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_pack4to1_gemm;

            pipeline_deconvolution_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_deconvolution_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_deconvolution_gemm->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_deconvolution_gemm->create(shader_type_index, opt, specializations);
        }

        {
            std::vector<vk_specialization_type> specializations(11 + 6);
            specializations[0].i = kernel_w;
            specializations[1].i = kernel_h;
            specializations[2].i = dilation_w;
            specializations[3].i = dilation_h;
            specializations[4].i = stride_w;
            specializations[5].i = stride_h;
            specializations[6].i = bias_term;
            specializations[7].i = activation_type;
            specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[10].i = num_output / out_elempack;
            specializations[11 + 0].i = shape_packed.w;
            specializations[11 + 1].i = shape_packed.h;
            specializations[11 + 2].i = out_shape_col_packed.cstep;
            specializations[11 + 3].i = out_shape_bordered_packed.w;
            specializations[11 + 4].i = out_shape_bordered_packed.h;
            specializations[11 + 5].i = out_shape_bordered_packed.cstep;

            Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack), (void*)0);
            if (out_shape_bordered_packed.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape_bordered_packed.w);
                local_size_xyz.h = std::min(8, out_shape_bordered_packed.h);
                local_size_xyz.c = std::min(4, out_shape_bordered_packed.c);
            }

            int shader_type_index = -1;
            if (out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_col2im;
            if (out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack4_col2im;

            pipeline_deconvolution_col2im = new Pipeline(vkdev);
            pipeline_deconvolution_col2im->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_deconvolution_col2im->create(shader_type_index, opt, specializations);
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

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        const Mat k0 = weight_data_r2.channel(q + i);

                        for (int j = 0; j < elempack; j++)
                        {
                            const float* k00 = k0.row(p + j);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }
    }

    std::vector<vk_specialization_type> specializations(10 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = activation_type;
    specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[10 + 0].i = shape_packed.dims;
    specializations[10 + 1].i = shape_packed.w;
    specializations[10 + 2].i = shape_packed.h;
    specializations[10 + 3].i = shape_packed.c;
    specializations[10 + 4].i = shape_packed.cstep;
    specializations[10 + 5].i = out_shape_bordered_packed.dims;
    specializations[10 + 6].i = out_shape_bordered_packed.w;
    specializations[10 + 7].i = out_shape_bordered_packed.h;
    specializations[10 + 8].i = out_shape_bordered_packed.c;
    specializations[10 + 9].i = out_shape_bordered_packed.cstep;

    Mat local_size_xyz(8, 8, std::min(4, num_output / out_elempack), (void*)0);
    if (out_shape_bordered_packed.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_bordered_packed.w);
        local_size_xyz.h = std::min(8, out_shape_bordered_packed.h);
        local_size_xyz.c = std::min(4, out_shape_bordered_packed.c);
    }

    int shader_type_index = -1;
    if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution;
    if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack4;
    if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack1to4;
    if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_pack4to1;

    pipeline_deconvolution = new Pipeline(vkdev);
    pipeline_deconvolution->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_deconvolution->create(shader_type_index, opt, specializations);

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Deconvolution_vulkan::destroy_pipeline(const Option& opt)
{
    if (crop)
    {
        crop->destroy_pipeline(opt);
        delete crop;
        crop = 0;
    }

    if (output_crop)
    {
        output_crop->destroy_pipeline(opt);
        delete output_crop;
        output_crop = 0;
    }

    delete pipeline_deconvolution;
    pipeline_deconvolution = 0;

    delete pipeline_deconvolution_gemm;
    pipeline_deconvolution_gemm = 0;

    delete pipeline_deconvolution_col2im;
    pipeline_deconvolution_col2im = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int Deconvolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (crop)
    {
        crop->upload_model(cmd, opt);
    }

    if (output_crop)
    {
        output_crop->upload_model(cmd, opt);
    }

    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    weight_data_packed.release();

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int Deconvolution_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int num_input = channels * elempack;
    const int maxk = kernel_w * kernel_h;

    VkMat top_blob_bordered;
    if (opt.use_sgemm_convolution && num_input >= 8 && maxk * num_output >= 8)
    {
        // gemm
        VkMat top_blob_col;
        {
            top_blob_col.create(w * h, 1, maxk * num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
            if (top_blob_col.empty())
                return -100;

            if (use_cooperative_matrix)
            {
                const int size = w * h;

                std::vector<VkMat> bindings(3);
                bindings[0] = bottom_blob;
                bindings[1] = top_blob_col;
                bindings[2] = weight_data_gpu;

                std::vector<vk_constant_type> constants(3);
                constants[0].u32 = size;
                constants[1].u32 = bottom_blob.cstep;
                constants[2].u32 = top_blob_col.cstep;

                const int blocks_x = (size + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                const int blocks_y = (maxk * num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                const int subgroup_size = vkdev->info.subgroup_size();

                VkMat dispatcher;
                dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                dispatcher.h = 1;
                dispatcher.c = 1;

                cmd.record_pipeline(pipeline_deconvolution_gemm, bindings, constants, dispatcher);
            }
            else
            {
                std::vector<VkMat> bindings(3);
                bindings[0] = bottom_blob;
                bindings[1] = top_blob_col;
                bindings[2] = weight_data_gpu;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = bottom_blob.w;
                constants[1].i = bottom_blob.h;
                constants[2].i = bottom_blob.c;
                constants[3].i = bottom_blob.cstep;
                constants[4].i = top_blob_col.cstep;
                constants[5].i = top_blob_col.c;

                VkMat dispatcher;
                dispatcher.w = (top_blob_col.cstep + 3) / 4;
                dispatcher.h = top_blob_col.c;
                dispatcher.c = 1;

                cmd.record_pipeline(pipeline_deconvolution_gemm, bindings, constants, dispatcher);
            }
        }

        // col2im
        {
            if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
            {
                top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
            }
            else
            {
                top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            }
            if (top_blob_bordered.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = top_blob_col;
            bindings[1] = top_blob_bordered;
            bindings[2] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = w;
            constants[1].i = h;
            constants[2].i = top_blob_col.cstep;
            constants[3].i = top_blob_bordered.w;
            constants[4].i = top_blob_bordered.h;
            constants[5].i = top_blob_bordered.cstep;

            cmd.record_pipeline(pipeline_deconvolution_col2im, bindings, constants, top_blob_bordered);
        }
    }
    else
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
        {
            top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
        }
        else
        {
            top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (top_blob_bordered.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob_bordered;
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_data_gpu;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob_bordered.dims;
        constants[6].i = top_blob_bordered.w;
        constants[7].i = top_blob_bordered.h;
        constants[8].i = top_blob_bordered.c;
        constants[9].i = top_blob_bordered.cstep;

        cmd.record_pipeline(pipeline_deconvolution, bindings, constants, top_blob_bordered);
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        {
            VkMat reference_blob;
            reference_blob.dims = 2;
            reference_blob.w = top_blob_bordered.w - pad_left - pad_right;
            reference_blob.h = top_blob_bordered.h - pad_top - pad_bottom;
            reference_blob.elempack = 1;

            std::vector<VkMat> crop_bottom_blobs(2);
            crop_bottom_blobs[0] = top_blob_bordered;
            crop_bottom_blobs[1] = reference_blob;
            std::vector<VkMat> crop_top_blobs(1);
            crop->forward(crop_bottom_blobs, crop_top_blobs, cmd, opt);
            top_blob = crop_top_blobs[0];
        }
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else if (output_w > 0 && output_h > 0)
    {
        int wcut = top_blob_bordered.w - output_w;
        int hcut = top_blob_bordered.h - output_h;

        VkMat crop_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator);
        int* crop_params = crop_param_blob.mapped();

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            crop_params[0] = wcut / 2;
            crop_params[1] = hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered.w - wcut;
            crop_params[4] = top_blob_bordered.h - hcut;
            crop_params[5] = top_blob_bordered.c * out_elempack;
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            crop_params[0] = wcut - wcut / 2;
            crop_params[1] = hcut - hcut / 2;
            crop_params[2] = 0;
            crop_params[3] = top_blob_bordered.w - wcut;
            crop_params[4] = top_blob_bordered.h - hcut;
            crop_params[5] = top_blob_bordered.c * out_elempack;
        }

        std::vector<VkMat> crop_inputs(2);
        crop_inputs[0] = top_blob_bordered;
        crop_inputs[1] = crop_param_blob;

        std::vector<VkMat> crop_outputs(1);
        output_crop->forward(crop_inputs, crop_outputs, cmd, opt);
        top_blob = crop_outputs[0];
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}

} // namespace ncnn
