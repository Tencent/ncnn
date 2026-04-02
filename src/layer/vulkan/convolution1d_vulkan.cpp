// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution1d_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Convolution1D_vulkan::Convolution1D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolution1d = 0;
    pipeline_convolution1d_1x1s1d1 = 0;
    pipeline_convolution1d_gemm = 0;

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

int Convolution1D_vulkan::load_param(const ParamDict& pd)
{
    int ret = Convolution1D::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

    return ret;
}

int Convolution1D_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const int maxk = kernel_w;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    bool use_gemm = opt.use_sgemm_convolution
                    && !is_conv1x1s1d1
                    && num_input * maxk >= 8
                    && num_output >= 8;

    if (use_gemm)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input * maxk, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input * maxk + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            Mat weight_data_r2;

            if (elempack == 4)
            {
                // from maxk-inch-outch to 4-maxk-inch/4-outch
                weight_data_r2.create(4 * maxk * (num_input / 4) * num_output);
                for (int i = 0; i < num_output; i++)
                {
                    for (int j = 0; j < num_input / 4; j++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4] = weight_data[(i * num_input + j * 4) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 1] = weight_data[(i * num_input + j * 4 + 1) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 2] = weight_data[(i * num_input + j * 4 + 2) * maxk + k];
                            weight_data_r2[((i * (num_input / 4) + j) * maxk + k) * 4 + 3] = weight_data[(i * num_input + j * 4 + 3) * maxk + k];
                        }
                    }
                }
            }
            else
            {
                weight_data_r2 = weight_data;
            }

            // from maxk-inch-outch to inch-maxk-outch
            Mat weight_data_r2_r = weight_data_r2.reshape(maxk, num_input, num_output);

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

            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            Mat weight_data_packed_fp16 = Mat(maxk * num_input_packed / 4 * 4, num_output_packed / 4 * 4, (size_t)2u, 1);

            for (int q = 0; q < num_output_packed; q += coopmat_N * UNROLL_SG_N * UNROLL_WG_N)
            {
                for (int p = 0; p < num_input_packed * maxk; p += coopmat_K * UNROLL_SG_K)
                {
                    for (int n = 0; n < coopmat_N * UNROLL_SG_N * UNROLL_WG_N; n++)
                    {
                        for (int k = 0; k < coopmat_K * UNROLL_SG_K; k++)
                        {
                            const int wgn = q + (n / (coopmat_N * UNROLL_SG_N)) * (coopmat_N * UNROLL_SG_N);
                            const int sg = (n % (coopmat_N * UNROLL_SG_N)) / coopmat_N;
                            const int nn = n % coopmat_N;

                            const int kk = p + k;

                            const int w_offset = wgn * maxk * num_input_packed + kk * (coopmat_N * UNROLL_SG_N) + sg * coopmat_N + nn;

                            float v = 0.f;
                            if (q + n < num_output && p + k < num_input * maxk)
                            {
                                const int w = kk / maxk;
                                const int kx = kk % maxk;
                                v = weight_data_r2_r.channel(q + n).row(w)[kx];
                            }

                            weight_data_packed_fp16[w_offset] = float32_to_float16(v);
                        }
                    }
                }
            }

            weight_data_packed = weight_data_packed_fp16;

            std::vector<vk_specialization_type> specializations(20 + 4);
            specializations[0].u32 = kernel_w;
            specializations[1].u32 = dilation_w;
            specializations[2].u32 = stride_w;
            specializations[3].i = bias_term;
            specializations[4].i = activation_type;
            specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[7].u32 = coopmat_M;
            specializations[8].u32 = coopmat_N;
            specializations[9].u32 = coopmat_K;
            specializations[10].u32 = coopmat_subgroup_size;
            specializations[11].u32 = UNROLL_SG_M;
            specializations[12].u32 = UNROLL_SG_N;
            specializations[13].u32 = UNROLL_SG_K;
            specializations[14].u32 = UNROLL_WG_M;
            specializations[15].u32 = UNROLL_WG_N;
            specializations[16].u32 = num_input;
            specializations[17].u32 = num_output;
            specializations[18].u32 = elempack;
            specializations[19].u32 = out_elempack;
            specializations[20 + 0].i = 0; // w
            specializations[20 + 1].i = 0; // cstep
            specializations[20 + 2].i = 0; // outw
            specializations[20 + 3].i = 0; // outcstep

            pipeline_convolution1d_gemm = new Pipeline(vkdev);
            pipeline_convolution1d_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution1d_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution1d_gemm->create(LayerShaderType::convolution1d_gemm_cm, opt, specializations);
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(maxk * num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.row(q / 4);

                for (int p = 0; p < num_input_packed; p += 4)
                {
                    for (int k = 0; k < maxk; k++)
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

            std::vector<vk_specialization_type> specializations(9 + 6);
            specializations[0].i = kernel_w;
            specializations[1].i = dilation_w;
            specializations[2].i = stride_w;
            specializations[3].i = bias_term;
            specializations[4].i = activation_type;
            specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[7].i = elempack;
            specializations[8].i = out_elempack;
            specializations[9 + 0].i = 0;
            specializations[9 + 1].i = num_input_packed / 4;
            specializations[9 + 2].i = 0;
            specializations[9 + 3].i = 0;
            specializations[9 + 4].i = num_output;
            specializations[9 + 5].i = num_input;

            pipeline_convolution1d_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_convolution1d_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution1d_gemm->set_local_size_xyz(16, std::min(4, num_output_packed / 4), 1);
            }
            pipeline_convolution1d_gemm->create(LayerShaderType::convolution1d_packed_gemm, opt, specializations);
        }
    }
    else if (is_conv1x1s1d1)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed) && num_input >= 8 && num_output >= 8;

        if (use_cooperative_matrix)
        {
            int size = 1024;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

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

            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            Mat weight_data_packed_fp16 = Mat(num_input_packed / 4 * 4, num_output_packed / 4 * 4, (size_t)2u, 1);

            for (int q = 0; q < num_output_packed; q += coopmat_N * UNROLL_SG_N * UNROLL_WG_N)
            {
                for (int p = 0; p < num_input_packed; p += coopmat_K * UNROLL_SG_K)
                {
                    for (int n = 0; n < coopmat_N * UNROLL_SG_N * UNROLL_WG_N; n++)
                    {
                        for (int k = 0; k < coopmat_K * UNROLL_SG_K; k++)
                        {
                            const int wgn = q + (n / (coopmat_N * UNROLL_SG_N)) * (coopmat_N * UNROLL_SG_N);
                            const int sg = (n % (coopmat_N * UNROLL_SG_N)) / coopmat_N;
                            const int nn = n % coopmat_N;

                            const int kk = p + k;

                            const int w_offset = wgn * num_input_packed + kk * (coopmat_N * UNROLL_SG_N) + sg * coopmat_N + nn;

                            float v = 0.f;
                            if (q + n < num_output && p + k < num_input)
                            {
                                v = weight_data[(q + n) * num_input + (p + k)];
                            }

                            weight_data_packed_fp16[w_offset] = float32_to_float16(v);
                        }
                    }
                }
            }

            weight_data_packed = weight_data_packed_fp16;

            std::vector<vk_specialization_type> specializations(17 + 3);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].u32 = coopmat_M;
            specializations[5].u32 = coopmat_N;
            specializations[6].u32 = coopmat_K;
            specializations[7].u32 = coopmat_subgroup_size;
            specializations[8].u32 = UNROLL_SG_M;
            specializations[9].u32 = UNROLL_SG_N;
            specializations[10].u32 = UNROLL_SG_K;
            specializations[11].u32 = UNROLL_WG_M;
            specializations[12].u32 = UNROLL_WG_N;
            specializations[13].u32 = num_input;
            specializations[14].u32 = num_output;
            specializations[15].u32 = elempack;
            specializations[16].u32 = out_elempack;
            specializations[17 + 0].u32 = 0; // size
            specializations[17 + 1].u32 = 0; // cstep
            specializations[17 + 2].u32 = 0; // outcstep

            pipeline_convolution1d_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution1d_1x1s1d1->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution1d_1x1s1d1->create(LayerShaderType::convolution1d_1x1s1d1_cm, opt, specializations);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.row(q / 4);

                for (int p = 0; p < num_input_packed; p += 4)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (q + i < num_output && p + j < num_input)
                            {
                                g00[0] = weight_data[(q + i) * num_input + (p + j)];
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

            const int outh_pack4 = num_output_packed / 4;

            std::vector<vk_specialization_type> specializations(6 + 6);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].i = elempack;
            specializations[5].i = out_elempack;
            specializations[6 + 0].i = 0; // w
            specializations[6 + 1].i = 0; // h
            specializations[6 + 2].i = 0; // outw
            specializations[6 + 3].i = outh_pack4;
            specializations[6 + 4].i = num_output;
            specializations[6 + 5].i = num_input_packed;

            pipeline_convolution1d_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution1d_1x1s1d1->set_local_size_xyz(8, std::min(8, outh_pack4), 1);
            pipeline_convolution1d_1x1s1d1->create(LayerShaderType::convolution1d_packed_1x1s1d1, opt, specializations);
        }
    }
    else
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        const int num_output_packed = (num_output + 3) / 4 * 4;

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

        std::vector<vk_specialization_type> specializations(9 + 5);
        specializations[0].i = kernel_w;
        specializations[1].i = dilation_w;
        specializations[2].i = stride_w;
        specializations[3].i = bias_term;
        specializations[4].i = activation_type;
        specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[7].i = elempack;
        specializations[8].i = out_elempack;
        specializations[9 + 0].i = 0;
        specializations[9 + 1].i = 0;
        specializations[9 + 2].i = 0;
        specializations[9 + 3].i = 0;
        specializations[9 + 4].i = num_output;

        pipeline_convolution1d = new Pipeline(vkdev);
        pipeline_convolution1d->set_optimal_local_size_xyz(1, 1, 1);
        pipeline_convolution1d->create(LayerShaderType::convolution1d_packed, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Convolution1D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution1d;
    pipeline_convolution1d = 0;

    delete pipeline_convolution1d_1x1s1d1;
    pipeline_convolution1d_1x1s1d1 = 0;

    delete pipeline_convolution1d_gemm;
    pipeline_convolution1d_gemm = 0;

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

int Convolution1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
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

int Convolution1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;
            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = 0;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;

    int outw = (w - kernel_extent_w) / stride_w + 1;

    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int maxk = kernel_w;
    const int num_input = bottom_blob_bordered.h * elempack;

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    bool use_gemm = opt.use_sgemm_convolution
                    && !is_conv1x1s1d1
                    && num_input * maxk >= 8
                    && num_output >= 8;

    if (use_gemm && pipeline_convolution1d_gemm)
    {
        if (use_cooperative_matrix)
        {
            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.cstep;
            constants[2].u32 = top_blob.w;
            constants[3].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);

            return 0;
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = bottom_blob_bordered;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = num_input_packed / 4;
            constants[2].i = top_blob.w;
            constants[3].i = num_output_packed / 4;
            constants[4].i = num_output;
            constants[5].i = num_input;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w + 3) / 4;
            dispatcher.h = num_output_packed / 4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_gemm, bindings, constants, dispatcher);

            return 0;
        }
    }

    if (is_conv1x1s1d1 && pipeline_convolution1d_1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.cstep;
            constants[2].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_1x1s1d1, bindings, constants, dispatcher);

            return 0;
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            const int outh_pack4 = num_output_packed / 4;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = bottom_blob_bordered;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = top_blob.w;
            constants[3].i = outh_pack4;
            constants[4].i = num_output;
            constants[5].i = num_input_packed;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w + 1) / 2;
            dispatcher.h = (outh_pack4 + 1) / 2;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution1d_1x1s1d1, bindings, constants, dispatcher);

            return 0;
        }
    }

    const int num_output_packed = (num_output + 3) / 4 * 4;

    top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const int outh_pack4 = num_output_packed / 4;

    std::vector<VkMat> bindings(6);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob_bordered;
    bindings[3] = top_blob;
    bindings[4] = weight_data_gpu;
    bindings[5] = bias_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_blob_bordered.w;
    constants[1].i = bottom_blob_bordered.h;
    constants[2].i = top_blob.w;
    constants[3].i = outh_pack4;
    constants[4].i = num_output;

    VkMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (outh_pack4 + 1) / 2;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_convolution1d, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
