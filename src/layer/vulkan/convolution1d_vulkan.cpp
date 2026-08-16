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

    pipeline_convolution1d_3s1d1_winograd23_transform_input = 0;
    pipeline_convolution1d_3s1d1_winograd23_gemm = 0;
    pipeline_convolution1d_3s1d1_winograd23_transform_output = 0;
    pipeline_convolution1d_3s1d1_winograd43_transform_input = 0;
    pipeline_convolution1d_3s1d1_winograd43_gemm = 0;
    pipeline_convolution1d_3s1d1_winograd43_transform_output = 0;

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

    use_subgroup_ops = false;
    winograd_use_cooperative_matrix = false;
    winograd_coopmat_M = 0;
    winograd_coopmat_N = 0;
    winograd_coopmat_K = 0;
    winograd_coopmat_subgroup_size = 0;
    winograd_UNROLL_SG_M = 1;
    winograd_UNROLL_SG_N = 1;
    winograd_UNROLL_SG_K = 1;
    winograd_UNROLL_WG_M = 1;
    winograd_UNROLL_WG_N = 1;
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

    const int subgroup_size = vkdev->info.subgroup_size();
    const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
    const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    use_subgroup_ops = opt.use_subgroup_ops && ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize wired subgroup_size
        use_subgroup_ops = false;
    }
    if (opt.use_fp16_arithmetic && !opt.use_bf16_storage && !opt.use_bf16_packed && !vkdev->info.queryShaderSubgroupExtendedTypesFeatures().shaderSubgroupExtendedTypes)
    {
        // sg shaders shuffle fp16 vectors, which requires subgroup extended types
        use_subgroup_ops = false;
    }

    if (use_subgroup_ops)
    {
        if (subgroup_size == 128)
        {
            UNROLL_SG_M = 16;
            UNROLL_SG_N = 8;
            UNROLL_SG_K = 8;
        }
        if (subgroup_size == 64)
        {
            UNROLL_SG_M = 8;
            UNROLL_SG_N = 8;
            UNROLL_SG_K = 8;
        }
        if (subgroup_size == 32)
        {
            UNROLL_SG_M = 8;
            UNROLL_SG_N = 4;
            UNROLL_SG_K = 4;
        }
        if (subgroup_size == 16)
        {
            UNROLL_SG_M = 4;
            UNROLL_SG_N = 4;
            UNROLL_SG_K = 4;
        }
        if (subgroup_size == 8)
        {
            UNROLL_SG_M = 4;
            UNROLL_SG_N = 2;
            UNROLL_SG_K = 2;
        }
        if (subgroup_size == 4)
        {
            UNROLL_SG_M = 2;
            UNROLL_SG_N = 2;
            UNROLL_SG_K = 2;
        }
    }

    bool is_conv3s1d1 = kernel_w == 3 && stride_w == 1 && dilation_w == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3s1d1 && num_input >= 16 && num_output >= 16)
    {
        winograd_use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (winograd_use_cooperative_matrix)
        {
            int size = 1024;
            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, winograd_coopmat_M, winograd_coopmat_N, winograd_coopmat_K, winograd_coopmat_subgroup_size);

            winograd_UNROLL_SG_M = std::min((size + winograd_coopmat_M - 1) / winograd_coopmat_M, 2);
            winograd_UNROLL_SG_N = std::min((num_output + winograd_coopmat_N - 1) / winograd_coopmat_N, 2);
            winograd_UNROLL_SG_K = std::min((num_input + winograd_coopmat_K - 1) / winograd_coopmat_K, 2);

            winograd_UNROLL_WG_M = std::min((size + winograd_coopmat_M * winograd_UNROLL_SG_M - 1) / (winograd_coopmat_M * winograd_UNROLL_SG_M), 2);
            winograd_UNROLL_WG_N = std::min((num_output + winograd_coopmat_N * winograd_UNROLL_SG_N - 1) / (winograd_coopmat_N * winograd_UNROLL_SG_N), 2);
        }

        // === F(4,3) ===
        if (opt.use_winograd43_convolution)
        {
            // 1D weight transform: G (6x3) applied once
            Mat weight_data_tm;
            weight_data_tm.create(6, num_input, num_output);

            const float sq2 = 1.41421356237f;
            const float ktm[6][3] = {
                {1.0f, 0.0f, 0.0f},
                {-2.0f / 3, -sq2 / 3, -1.0f / 3},
                {-2.0f / 3, sq2 / 3, -1.0f / 3},
                {1.0f / 6, sq2 / 6, 1.0f / 3},
                {1.0f / 6, -sq2 / 6, 1.0f / 3},
                {0.0f, 0.0f, 1.0f}
            };

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const float* kernel0 = (const float*)weight_data + p * num_input * 3 + q * 3;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    for (int i = 0; i < 6; i++)
                    {
                        kernel_tm0[i] = kernel0[0] * ktm[i][0] + kernel0[1] * ktm[i][1] + kernel0[2] * ktm[i][2];
                    }
                }
            }

            // Weight packing - follow 2D pattern but with 6 instead of 36
            if (winograd_use_cooperative_matrix)
            {
                // from 6-inch-outch to inch-outch-6
                Mat weight_data_tm_r2(num_input, num_output, 6);
                for (int k = 0; k < 6; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);
                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 6 + k];
                        }
                    }
                }

                const int blocks_n = (num_output + winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N - 1) / (winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N);
                const int kk = (num_input + winograd_coopmat_K - 1) / winograd_coopmat_K;

                weight_winograd43_data_packed.create(winograd_coopmat_N * winograd_coopmat_K * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N * kk, blocks_n, 6);
                for (int b = 0; b < 6; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd43_data_packed.channel(b).row(bn);
                        int k = 0;
                        for (; k + winograd_UNROLL_SG_K - 1 < kk; k += winograd_UNROLL_SG_K)
                        {
                            for (int wn = 0; wn < winograd_UNROLL_WG_N; wn++)
                            {
                                for (int zk = 0; zk < winograd_UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < winograd_UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < winograd_coopmat_K; i++)
                                        {
                                            for (int j = 0; j < winograd_coopmat_N; j++)
                                            {
                                                const int gni = ((bn * winograd_UNROLL_WG_N + wn) * winograd_UNROLL_SG_N + zn) * winograd_coopmat_N + j;
                                                const int gki = (k + zk) * winograd_coopmat_K + i;
                                                if (gni < num_output && gki < num_input)
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                else
                                                    *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (; k < kk; k++)
                        {
                            for (int wn = 0; wn < winograd_UNROLL_WG_N; wn++)
                            {
                                for (int zn = 0; zn < winograd_UNROLL_SG_N; zn++)
                                {
                                    for (int i = 0; i < winograd_coopmat_K; i++)
                                    {
                                        for (int j = 0; j < winograd_coopmat_N; j++)
                                        {
                                            const int gni = ((bn * winograd_UNROLL_WG_N + wn) * winograd_UNROLL_SG_N + zn) * winograd_coopmat_N + j;
                                            const int gki = k * winograd_coopmat_K + i;
                                            if (gni < num_output && gki < num_input)
                                                *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                            else
                                                *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // non-cm packing
                weight_winograd43_data_packed.create(num_input / elempack, num_output / out_elempack, 6, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
                for (int k = 0; k < 6; k++)
                {
                    float* g00 = weight_winograd43_data_packed.channel(k);
                    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                    {
                        for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                        {
                            for (int i = 0; i < out_elempack; i++)
                            {
                                const Mat k0 = weight_data_tm.channel(q + i);
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

            // Create F(4,3) pipelines
            // transform_input
            {
                std::vector<vk_specialization_type> specializations(1 + 4);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = 0; // w
                specializations[1 + 1].i = 0; // cstep
                specializations[1 + 2].i = 0; // outcstep
                specializations[1 + 3].i = 0; // block_x

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd43_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution1d_packed_3s1d1_winograd43_transform_input;

                pipeline_convolution1d_3s1d1_winograd43_transform_input = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd43_transform_input->set_local_size_xyz(8, 1, 8);
                pipeline_convolution1d_3s1d1_winograd43_transform_input->create(shader_type_index, opt, specializations);
            }

            // gemm
            if (winograd_use_cooperative_matrix)
            {
                Mat weight_winograd43_data_packed_fp16 = Mat(weight_winograd43_data_packed.w, weight_winograd43_data_packed.h, weight_winograd43_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(15 + 3);
                specializations[0].u32 = 6; // batch = number of transformed positions
                specializations[1].u32 = winograd_coopmat_M;
                specializations[2].u32 = winograd_coopmat_N;
                specializations[3].u32 = winograd_coopmat_K;
                specializations[4].u32 = winograd_UNROLL_SG_M;
                specializations[5].u32 = winograd_UNROLL_SG_N;
                specializations[6].u32 = winograd_UNROLL_SG_K;
                specializations[7].u32 = winograd_UNROLL_WG_M;
                specializations[8].u32 = winograd_UNROLL_WG_N;
                specializations[9].u32 = winograd_coopmat_subgroup_size;
                specializations[10].u32 = num_input;
                specializations[11].u32 = num_output;
                specializations[12].u32 = elempack;
                specializations[13].u32 = out_elempack;
                specializations[14].u32 = weight_winograd43_data_packed_fp16.cstep;
                specializations[15 + 0].u32 = 0; // size
                specializations[15 + 1].u32 = 0; // cstep
                specializations[15 + 2].u32 = 0; // outcstep

                pipeline_convolution1d_3s1d1_winograd43_gemm = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd43_gemm->set_subgroup_size(winograd_coopmat_subgroup_size);
                pipeline_convolution1d_3s1d1_winograd43_gemm->set_local_size_xyz(winograd_coopmat_subgroup_size * winograd_UNROLL_WG_M * winograd_UNROLL_WG_N, 1, 1);
                pipeline_convolution1d_3s1d1_winograd43_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 6; // batch
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = 0; // cstep
                specializations[3 + 1].i = 0; // outw
                specializations[3 + 2].i = 0; // outcstep

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_pack4_3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_pack1to4_3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_pack4to1_3s1d1_winograd_gemm;

                pipeline_convolution1d_3s1d1_winograd43_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                    pipeline_convolution1d_3s1d1_winograd43_gemm->set_local_size_xyz(8, 8, 1);
                else
                    pipeline_convolution1d_3s1d1_winograd43_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                pipeline_convolution1d_3s1d1_winograd43_gemm->create(shader_type_index, opt, specializations);
            }

            // transform_output
            {
                std::vector<vk_specialization_type> specializations(5 + 4);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = 0; // cstep
                specializations[5 + 1].i = 0; // block_x
                specializations[5 + 2].i = 0; // outw
                specializations[5 + 3].i = 0; // outcstep

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd43_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_packed_3s1d1_winograd43_transform_output;

                pipeline_convolution1d_3s1d1_winograd43_transform_output = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd43_transform_output->set_local_size_xyz(8, 1, 8);
                pipeline_convolution1d_3s1d1_winograd43_transform_output->create(shader_type_index, opt, specializations);
            }
        }

        // === F(2,3) ===
        if (opt.use_winograd23_convolution)
        {
            // 1D weight transform: G (4x3) applied once
            Mat weight_data_tm;
            weight_data_tm.create(4, num_input, num_output);

            const float ktm[4][3] = {
                {1.0f, 0.0f, 0.0f},
                {1.0f / 2, 1.0f / 2, 1.0f / 2},
                {1.0f / 2, -1.0f / 2, 1.0f / 2},
                {0.0f, 0.0f, 1.0f}
            };

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const float* kernel0 = (const float*)weight_data + p * num_input * 3 + q * 3;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    for (int i = 0; i < 4; i++)
                    {
                        kernel_tm0[i] = kernel0[0] * ktm[i][0] + kernel0[1] * ktm[i][1] + kernel0[2] * ktm[i][2];
                    }
                }
            }

            // Weight packing - same pattern as F(4,3) but with 4 instead of 6
            if (winograd_use_cooperative_matrix)
            {
                Mat weight_data_tm_r2(num_input, num_output, 4);
                for (int k = 0; k < 4; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);
                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 4 + k];
                        }
                    }
                }

                const int blocks_n = (num_output + winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N - 1) / (winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N);
                const int kk = (num_input + winograd_coopmat_K - 1) / winograd_coopmat_K;

                weight_winograd23_data_packed.create(winograd_coopmat_N * winograd_coopmat_K * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N * kk, blocks_n, 4);
                for (int b = 0; b < 4; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd23_data_packed.channel(b).row(bn);
                        int k = 0;
                        for (; k + winograd_UNROLL_SG_K - 1 < kk; k += winograd_UNROLL_SG_K)
                        {
                            for (int wn = 0; wn < winograd_UNROLL_WG_N; wn++)
                            {
                                for (int zk = 0; zk < winograd_UNROLL_SG_K; zk++)
                                {
                                    for (int zn = 0; zn < winograd_UNROLL_SG_N; zn++)
                                    {
                                        for (int i = 0; i < winograd_coopmat_K; i++)
                                        {
                                            for (int j = 0; j < winograd_coopmat_N; j++)
                                            {
                                                const int gni = ((bn * winograd_UNROLL_WG_N + wn) * winograd_UNROLL_SG_N + zn) * winograd_coopmat_N + j;
                                                const int gki = (k + zk) * winograd_coopmat_K + i;
                                                if (gni < num_output && gki < num_input)
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                                else
                                                    *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        for (; k < kk; k++)
                        {
                            for (int wn = 0; wn < winograd_UNROLL_WG_N; wn++)
                            {
                                for (int zn = 0; zn < winograd_UNROLL_SG_N; zn++)
                                {
                                    for (int i = 0; i < winograd_coopmat_K; i++)
                                    {
                                        for (int j = 0; j < winograd_coopmat_N; j++)
                                        {
                                            const int gni = ((bn * winograd_UNROLL_WG_N + wn) * winograd_UNROLL_SG_N + zn) * winograd_coopmat_N + j;
                                            const int gki = k * winograd_coopmat_K + i;
                                            if (gni < num_output && gki < num_input)
                                                *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
                                            else
                                                *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                weight_winograd23_data_packed.create(num_input / elempack, num_output / out_elempack, 4, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
                for (int k = 0; k < 4; k++)
                {
                    float* g00 = weight_winograd23_data_packed.channel(k);
                    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                    {
                        for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                        {
                            for (int i = 0; i < out_elempack; i++)
                            {
                                const Mat k0 = weight_data_tm.channel(q + i);
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

            // Create F(2,3) pipelines (same pattern as F(4,3) but with 4 instead of 6, and winograd23 shader names)
            // transform_input
            {
                std::vector<vk_specialization_type> specializations(1 + 4);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = 0;
                specializations[1 + 1].i = 0;
                specializations[1 + 2].i = 0;
                specializations[1 + 3].i = 0;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd23_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution1d_packed_3s1d1_winograd23_transform_input;

                pipeline_convolution1d_3s1d1_winograd23_transform_input = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd23_transform_input->set_local_size_xyz(8, 1, 8);
                pipeline_convolution1d_3s1d1_winograd23_transform_input->create(shader_type_index, opt, specializations);
            }

            // gemm
            if (winograd_use_cooperative_matrix)
            {
                Mat weight_winograd23_data_packed_fp16 = Mat(weight_winograd23_data_packed.w, weight_winograd23_data_packed.h, weight_winograd23_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(15 + 3);
                specializations[0].u32 = 4; // batch
                specializations[1].u32 = winograd_coopmat_M;
                specializations[2].u32 = winograd_coopmat_N;
                specializations[3].u32 = winograd_coopmat_K;
                specializations[4].u32 = winograd_UNROLL_SG_M;
                specializations[5].u32 = winograd_UNROLL_SG_N;
                specializations[6].u32 = winograd_UNROLL_SG_K;
                specializations[7].u32 = winograd_UNROLL_WG_M;
                specializations[8].u32 = winograd_UNROLL_WG_N;
                specializations[9].u32 = winograd_coopmat_subgroup_size;
                specializations[10].u32 = num_input;
                specializations[11].u32 = num_output;
                specializations[12].u32 = elempack;
                specializations[13].u32 = out_elempack;
                specializations[14].u32 = weight_winograd23_data_packed_fp16.cstep;
                specializations[15 + 0].u32 = 0;
                specializations[15 + 1].u32 = 0;
                specializations[15 + 2].u32 = 0;

                pipeline_convolution1d_3s1d1_winograd23_gemm = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd23_gemm->set_subgroup_size(winograd_coopmat_subgroup_size);
                pipeline_convolution1d_3s1d1_winograd23_gemm->set_local_size_xyz(winograd_coopmat_subgroup_size * winograd_UNROLL_WG_M * winograd_UNROLL_WG_N, 1, 1);
                pipeline_convolution1d_3s1d1_winograd23_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 4;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = 0;
                specializations[3 + 1].i = 0;
                specializations[3 + 2].i = 0;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_pack4_3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_pack1to4_3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_pack4to1_3s1d1_winograd_gemm;

                pipeline_convolution1d_3s1d1_winograd23_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                    pipeline_convolution1d_3s1d1_winograd23_gemm->set_local_size_xyz(8, 8, 1);
                else
                    pipeline_convolution1d_3s1d1_winograd23_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                pipeline_convolution1d_3s1d1_winograd23_gemm->create(shader_type_index, opt, specializations);
            }

            // transform_output
            {
                std::vector<vk_specialization_type> specializations(5 + 4);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = 0;
                specializations[5 + 1].i = 0;
                specializations[5 + 2].i = 0;
                specializations[5 + 3].i = 0;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution1d_3s1d1_winograd23_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution1d_packed_3s1d1_winograd23_transform_output;

                pipeline_convolution1d_3s1d1_winograd23_transform_output = new Pipeline(vkdev);
                pipeline_convolution1d_3s1d1_winograd23_transform_output->set_local_size_xyz(8, 1, 8);
                pipeline_convolution1d_3s1d1_winograd23_transform_output->create(shader_type_index, opt, specializations);
            }
        }
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

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (num_input * maxk + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
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

                                        if (gni < num_output && gki < num_input * maxk)
                                        {
                                            *p++ = weight_data_r2[gni * num_input * maxk + gki];
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
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                        {
                            for (int i = 0; i < coopmat_K; i++)
                            {
                                for (int j = 0; j < coopmat_N; j++)
                                {
                                    const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                    const int gki = k * coopmat_K + i;

                                    if (gni < num_output && gki < num_input * maxk)
                                    {
                                        *p++ = weight_data_r2[gni * num_input * maxk + gki];
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

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* p = weight_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
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

                                        if (gni < num_output && gki < num_input)
                                        {
                                            *p++ = weight_data[gni * num_input + gki];
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
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                        {
                            for (int i = 0; i < coopmat_K; i++)
                            {
                                for (int j = 0; j < coopmat_N; j++)
                                {
                                    const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                    const int gki = k * coopmat_K + i;

                                    if (gni < num_output && gki < num_input)
                                    {
                                        *p++ = weight_data[gni * num_input + gki];
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

        if (use_subgroup_ops && opt.use_fp16_arithmetic)
        {
            std::vector<vk_specialization_type> specializations(9 + 5 + 4);
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
            specializations[14].i = 0;
            specializations[15].u32 = UNROLL_SG_M;
            specializations[16].u32 = UNROLL_SG_N;
            specializations[17].u32 = UNROLL_SG_K;

            pipeline_convolution1d = new Pipeline(vkdev);
            pipeline_convolution1d->set_subgroup_size(subgroup_size);
            pipeline_convolution1d->set_local_size_xyz(subgroup_size, 1, 1);
            pipeline_convolution1d->create(LayerShaderType::convolution1d_packed_sg, opt, specializations);
        }
        else
        {
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

    delete pipeline_convolution1d_3s1d1_winograd23_transform_input;
    delete pipeline_convolution1d_3s1d1_winograd23_gemm;
    delete pipeline_convolution1d_3s1d1_winograd23_transform_output;
    pipeline_convolution1d_3s1d1_winograd23_transform_input = 0;
    pipeline_convolution1d_3s1d1_winograd23_gemm = 0;
    pipeline_convolution1d_3s1d1_winograd23_transform_output = 0;

    delete pipeline_convolution1d_3s1d1_winograd43_transform_input;
    delete pipeline_convolution1d_3s1d1_winograd43_gemm;
    delete pipeline_convolution1d_3s1d1_winograd43_transform_output;
    pipeline_convolution1d_3s1d1_winograd43_transform_input = 0;
    pipeline_convolution1d_3s1d1_winograd43_gemm = 0;
    pipeline_convolution1d_3s1d1_winograd43_transform_output = 0;

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

    use_subgroup_ops = false;

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

    if (pipeline_convolution1d_3s1d1_winograd43_gemm)
    {
        cmd.record_upload(weight_winograd43_data_packed, weight_data_gpu_tm_winograd43, opt);
        weight_winograd43_data_packed.release();
    }

    if (pipeline_convolution1d_3s1d1_winograd23_gemm)
    {
        cmd.record_upload(weight_winograd23_data_packed, weight_data_gpu_tm_winograd23, opt);
        weight_winograd23_data_packed.release();
    }

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

    const int num_input = bottom_blob_bordered.h * elempack;

    bool is_conv3s1d1 = kernel_w == 3 && stride_w == 1 && dilation_w == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3s1d1 && num_input >= 16 && num_output >= 16)
    {
        bool pre_winograd43 = opt.use_winograd43_convolution;
        if (opt.use_winograd23_convolution)
        {
            if (vkdev->info.type() == 0 && w <= 18)
                pre_winograd43 = false;
            if (vkdev->info.type() != 0 && w <= 12)
                pre_winograd43 = false;

            if (winograd_use_cooperative_matrix && w <= 18)
                pre_winograd43 = false;
        }

        if (pre_winograd43)
        {
            // winograd43
            int block_x = (outw + 3) / 4;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x, 1, num_input / elempack * 6, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(4);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.w;
                constants[2].i = bottom_tm_blob.cstep;
                constants[3].i = block_x;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = 1;
                dispatcher.c = num_input / elempack;

                cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd43_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x, 1, num_output / out_elempack * 6, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (winograd_use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + winograd_coopmat_M * winograd_UNROLL_SG_M * winograd_UNROLL_WG_M - 1) / (winograd_coopmat_M * winograd_UNROLL_SG_M * winograd_UNROLL_WG_M);
                    const int blocks_y = (num_output + winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N - 1) / (winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N);

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (winograd_coopmat_subgroup_size * winograd_UNROLL_WG_M * winograd_UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 6;

                    cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(4);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;
                    constants[3].i = weight_data_gpu_tm_winograd43.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 6;

                    cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(4);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = top_blob.w;
                constants[3].i = top_blob.w;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = 1;
                dispatcher.c = num_output / out_elempack;

                cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd43_transform_output, bindings, constants, dispatcher);
            }
        }
        else
        {
            // winograd23
            int block_x = (outw + 1) / 2;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x, 1, num_input / elempack * 4, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(4);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.w;
                constants[2].i = bottom_tm_blob.cstep;
                constants[3].i = block_x;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = 1;
                dispatcher.c = num_input / elempack;

                cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd23_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x, 1, num_output / out_elempack * 4, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (winograd_use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + winograd_coopmat_M * winograd_UNROLL_SG_M * winograd_UNROLL_WG_M - 1) / (winograd_coopmat_M * winograd_UNROLL_SG_M * winograd_UNROLL_WG_M);
                    const int blocks_y = (num_output + winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N - 1) / (winograd_coopmat_N * winograd_UNROLL_SG_N * winograd_UNROLL_WG_N);

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (winograd_coopmat_subgroup_size * winograd_UNROLL_WG_M * winograd_UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 4;

                    cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(4);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;
                    constants[3].i = weight_data_gpu_tm_winograd23.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 4;

                    cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(4);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = top_blob.w;
                constants[3].i = top_blob.w;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = 1;
                dispatcher.c = num_output / out_elempack;

                cmd.record_pipeline(pipeline_convolution1d_3s1d1_winograd23_transform_output, bindings, constants, dispatcher);
            }
        }

        return 0;
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && stride_w == 1 && dilation_w == 1;

    const int maxk = kernel_w;

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

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;
            bindings[4] = bottom_blob_bordered;
            bindings[5] = top_blob;

            std::vector<vk_constant_type> constants(4);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.w;
            constants[2].u32 = top_blob.w;
            constants[3].u32 = top_blob.w;

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

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;
            bindings[4] = bottom_blob_bordered;
            bindings[5] = top_blob;

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.w;
            constants[2].u32 = top_blob.w;

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
    if (use_subgroup_ops && opt.use_fp16_arithmetic)
    {
        const int blocks_x = (top_blob.w + UNROLL_SG_M * 4 - 1) / (UNROLL_SG_M * 4);
        const int blocks_y = (outh_pack4 + UNROLL_SG_N - 1) / UNROLL_SG_N;

        dispatcher.w = (blocks_x * blocks_y) * vkdev->info.subgroup_size();
        dispatcher.h = 1;
        dispatcher.c = 1;
    }
    else
    {
        dispatcher.w = (top_blob.w + 1) / 2;
        dispatcher.h = (outh_pack4 + 1) / 2;
        dispatcher.c = 1;
    }

    cmd.record_pipeline(pipeline_convolution1d, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
