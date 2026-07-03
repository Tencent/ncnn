// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"
#include "modelbin.h"

namespace ncnn {

Convolution_vulkan::Convolution_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolution = 0;
    pipeline_convolution_1x1s1d1 = 0;
    pipeline_convolution_gemm = 0;

    pipeline_convolution_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd23_transform_output = 0;

    pipeline_convolution_3x3s1d1_winograd43_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd43_transform_output = 0;

    reshape_1x1xw = 0;
    reshape_w = 0;

    use_cooperative_matrix = false;
#if NCNN_INT8
    quantize = 0;
    pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm = 0;
#endif
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

int Convolution_vulkan::load_param(const ParamDict& pd)
{
    int ret = Convolution::load_param(pd);

    if (dynamic_weight)
    {
        support_vulkan = false;
    }

#if NCNN_INT8
    if (int8_scale_term && pad_value != 0.f)
    {
        NCNN_LOGE("Convolution_vulkan int8 nonzero pad value is not supported");
        support_vulkan = false;
    }
#endif

    return ret;
}

int Convolution_vulkan::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return create_pipeline_int8(opt);
    }
#endif

    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // skip fc like hint
    if (shape.dims != 3) shape = Mat();
    if (out_shape.dims != 3) out_shape = Mat();

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c, (void*)0, shape.elemsize, shape.elempack);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h + hpad, shape.c, (void*)0, shape.elemsize, shape.elempack);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    int elempack = num_input % 4 == 0 ? 4 : 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    // fc
    if (kernel_w == 1 && kernel_h == 1)
    {
        {
            reshape_1x1xw = ncnn::create_layer_vulkan(ncnn::LayerType::Reshape);
            reshape_1x1xw->vkdev = vkdev;

            reshape_1x1xw->bottom_shapes.resize(1);
            reshape_1x1xw->bottom_shapes[0] = Mat(num_input / elempack, (void*)0, elemsize, elempack);
            reshape_1x1xw->top_shapes.resize(1);
            reshape_1x1xw->top_shapes[0] = Mat(1, 1, num_input / elempack, (void*)0, elemsize, elempack);

            ncnn::ParamDict pd;
            pd.set(0, 1);         // w
            pd.set(1, 1);         // h
            pd.set(2, num_input); // c

            reshape_1x1xw->load_param(pd);

            reshape_1x1xw->create_pipeline(opt);
        }

        {
            reshape_w = ncnn::create_layer_vulkan(ncnn::LayerType::Reshape);
            reshape_w->vkdev = vkdev;

            reshape_w->bottom_shapes.resize(1);
            reshape_w->bottom_shapes[0] = Mat(1, 1, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);
            reshape_w->top_shapes.resize(1);
            reshape_w->top_shapes[0] = Mat(num_output / out_elempack, (void*)0, out_elemsize, out_elempack);

            ncnn::ParamDict pd;
            pd.set(0, num_output); // w

            reshape_w->load_param(pd);

            reshape_w->create_pipeline(opt);
        }
    }

    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            // f43 and f23 share the same size parameter, set zero for dynamic dispatch
            // if (out_shape.dims != 0)
            // {
            //     int block_x = (out_shape.w + 3) / 4;
            //     int block_y = (out_shape.h + 3) / 4;
            //     int block_x = (out_shape.w + 1) / 2;
            //     int block_y = (out_shape.h + 1) / 2;
            //     size = block_x * block_y;
            // }

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);
        }

        // winograd43 transform kernel
        if (opt.use_winograd43_convolution)
        {
            Mat weight_data_tm;
            weight_data_tm.create(6 * 6, num_input, num_output);

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
                    const float* kernel0 = (const float*)weight_data + p * num_input * 9 + q * 9;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // transform kernel
                    const float* k0 = kernel0;
                    const float* k1 = kernel0 + 3;
                    const float* k2 = kernel0 + 6;

                    // h
                    float tmp[6][3];
                    for (int i = 0; i < 6; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j < 6; j++)
                    {
                        float* tmpp = &tmp[j][0];

                        for (int i = 0; i < 6; i++)
                        {
                            kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }

            if (use_cooperative_matrix)
            {
                // from 36-inch-outch to inch-outch-36
                Mat weight_data_tm_r2(num_input, num_output, 36);
                for (int k = 0; k < 36; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);

                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 36 + k];
                        }
                    }
                }

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

                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
                const int kk = (num_input + coopmat_K - 1) / coopmat_K;

                weight_winograd43_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n, 36);
                for (int b = 0; b < 36; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd43_data_packed.channel(b).row(bn);

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

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
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

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
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
                }
            }
            else
            {
                // src = 36-inch-outch
                // dst = 8a-8b-inch/8a-outch/8b-36
                weight_winograd43_data_packed.create(num_input / elempack, num_output / out_elempack, 36, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

                for (int k = 0; k < 36; k++)
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
        }

        // winograd43
        if (opt.use_winograd43_convolution)
        {
            int block_x = 0;
            int block_y = 0;
            Mat shape_winograd_input_transformed;
            Mat shape_winograd_gemm;
            Mat shape_winograd_input_transformed_packed;
            Mat shape_winograd_gemm_packed;

            if (out_shape.dims != 0)
            {
                block_x = (out_shape.w + 3) / 4;
                block_y = (out_shape.h + 3) / 4;

                shape_winograd_input_transformed = Mat(block_x * block_y, 1, shape.c * 36, (void*)0);
                shape_winograd_gemm = Mat(block_x * block_y, 1, out_shape.c * 36, (void*)0);
            }

            if (shape_winograd_input_transformed.dims == 3) shape_winograd_input_transformed_packed = Mat(shape_winograd_input_transformed.w, 1, shape_winograd_input_transformed.h / elempack * 36, (void*)0, elemsize, elempack);

            if (shape_winograd_gemm.dims == 3) shape_winograd_gemm_packed = Mat(shape_winograd_gemm.w, 1, shape_winograd_gemm.h / out_elempack * 36, (void*)0, out_elemsize, out_elempack);

            {
                std::vector<vk_specialization_type> specializations(1 + 6);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = shape_bordered.w;
                specializations[1 + 1].i = shape_bordered.h;
                specializations[1 + 2].i = shape_bordered.cstep;
                specializations[1 + 3].i = shape_winograd_input_transformed_packed.cstep;
                specializations[1 + 4].i = block_x;
                specializations[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_input;

                pipeline_convolution_3x3s1d1_winograd43_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->create(shader_type_index, opt, specializations);
            }

            if (use_cooperative_matrix)
            {
                Mat weight_winograd43_data_packed_fp16 = Mat(weight_winograd43_data_packed.w, weight_winograd43_data_packed.h, weight_winograd43_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(15 + 3);
                specializations[0].u32 = 36; //batch
                specializations[1].u32 = coopmat_M;
                specializations[2].u32 = coopmat_N;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_M;
                specializations[5].u32 = UNROLL_SG_N;
                specializations[6].u32 = UNROLL_SG_K;
                specializations[7].u32 = UNROLL_WG_M;
                specializations[8].u32 = UNROLL_WG_N;
                specializations[9].u32 = coopmat_subgroup_size;
                specializations[10].u32 = num_input;
                specializations[11].u32 = num_output;
                specializations[12].u32 = elempack;
                specializations[13].u32 = out_elempack;
                specializations[14].u32 = weight_winograd43_data_packed_fp16.cstep;
                specializations[15 + 0].u32 = shape_winograd_input_transformed_packed.w;
                specializations[15 + 1].u32 = shape_winograd_input_transformed_packed.cstep;
                specializations[15 + 2].u32 = shape_winograd_gemm_packed.cstep;

                pipeline_convolution_3x3s1d1_winograd43_gemm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_gemm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd43_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 36;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = shape_winograd_input_transformed_packed.cstep;
                specializations[3 + 1].i = shape_winograd_gemm_packed.w;
                specializations[3 + 2].i = shape_winograd_gemm_packed.cstep;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_3x3s1d1_winograd_gemm;

                pipeline_convolution_3x3s1d1_winograd43_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                }
                pipeline_convolution_3x3s1d1_winograd43_gemm->create(shader_type_index, opt, specializations);
            }

            {
                std::vector<vk_specialization_type> specializations(5 + 6);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations[5 + 1].i = block_x;
                specializations[5 + 2].i = block_y;
                specializations[5 + 3].i = out_shape.w;
                specializations[5 + 4].i = out_shape.h;
                specializations[5 + 5].i = out_shape.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_output;

                pipeline_convolution_3x3s1d1_winograd43_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->create(shader_type_index, opt, specializations);
            }
        }

        // winograd23 transform kernel
        if (opt.use_winograd23_convolution)
        {
            Mat weight_data_tm;
            weight_data_tm.create(4 * 4, num_input, num_output);

            // G
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
                    const float* kernel0 = (const float*)weight_data + p * num_input * 9 + q * 9;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // transform kernel
                    const float* k0 = kernel0;
                    const float* k1 = kernel0 + 3;
                    const float* k2 = kernel0 + 6;

                    // h
                    float tmp[4][3];
                    for (int i = 0; i < 4; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j = 0; j < 4; j++)
                    {
                        float* tmpp = &tmp[j][0];

                        for (int i = 0; i < 4; i++)
                        {
                            kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }

            if (use_cooperative_matrix)
            {
                // from 16-inch-outch to inch-outch-16
                Mat weight_data_tm_r2(num_input, num_output, 16);
                for (int k = 0; k < 16; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);

                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 16 + k];
                        }
                    }
                }

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

                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
                const int kk = (num_input + coopmat_K - 1) / coopmat_K;

                weight_winograd23_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n, 16);
                for (int b = 0; b < 16; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd23_data_packed.channel(b).row(bn);

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

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
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

                                                if (gni < num_output && gki < num_input)
                                                {
                                                    *p++ = weight_data_tm_r2.channel(b)[gni * num_input + gki];
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
                }
            }
            else
            {
                // src = 16-inch-outch
                // dst = 8a-8b-inch/8a-outch/8b-16
                weight_winograd23_data_packed.create(num_input / elempack, num_output / out_elempack, 16, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

                for (int k = 0; k < 16; k++)
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
        }

        // winograd23
        if (opt.use_winograd23_convolution)
        {
            int block_x = 0;
            int block_y = 0;
            Mat shape_winograd_input_transformed;
            Mat shape_winograd_gemm;
            Mat shape_winograd_input_transformed_packed;
            Mat shape_winograd_gemm_packed;

            if (out_shape.dims != 0)
            {
                block_x = (out_shape.w + 1) / 2;
                block_y = (out_shape.h + 1) / 2;

                shape_winograd_input_transformed = Mat(block_x * block_y, 1, shape.c * 16, (void*)0);
                shape_winograd_gemm = Mat(block_x * block_y, 1, out_shape.c * 16, (void*)0);
            }

            if (shape_winograd_input_transformed.dims == 3) shape_winograd_input_transformed_packed = Mat(shape_winograd_input_transformed.w, 1, shape_winograd_input_transformed.h / elempack * 16, (void*)0, elemsize, elempack);

            if (shape_winograd_gemm.dims == 3) shape_winograd_gemm_packed = Mat(shape_winograd_gemm.w, 1, shape_winograd_gemm.h / out_elempack * 16, (void*)0, out_elemsize, out_elempack);

            {
                std::vector<vk_specialization_type> specializations(1 + 6);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = shape_bordered.w;
                specializations[1 + 1].i = shape_bordered.h;
                specializations[1 + 2].i = shape_bordered.cstep;
                specializations[1 + 3].i = shape_winograd_input_transformed_packed.cstep;
                specializations[1 + 4].i = block_x;
                specializations[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_input;

                pipeline_convolution_3x3s1d1_winograd23_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->create(shader_type_index, opt, specializations);
            }

            if (use_cooperative_matrix)
            {
                Mat weight_winograd23_data_packed_fp16 = Mat(weight_winograd23_data_packed.w, weight_winograd23_data_packed.h, weight_winograd23_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(15 + 3);
                specializations[0].u32 = 16; //batch
                specializations[1].u32 = coopmat_M;
                specializations[2].u32 = coopmat_N;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_M;
                specializations[5].u32 = UNROLL_SG_N;
                specializations[6].u32 = UNROLL_SG_K;
                specializations[7].u32 = UNROLL_WG_M;
                specializations[8].u32 = UNROLL_WG_N;
                specializations[9].u32 = coopmat_subgroup_size;
                specializations[10].u32 = num_input;
                specializations[11].u32 = num_output;
                specializations[12].u32 = elempack;
                specializations[13].u32 = out_elempack;
                specializations[14].u32 = weight_winograd23_data_packed_fp16.cstep;
                specializations[15 + 0].u32 = shape_winograd_input_transformed_packed.w;
                specializations[15 + 1].u32 = shape_winograd_input_transformed_packed.cstep;
                specializations[15 + 2].u32 = shape_winograd_gemm_packed.cstep;

                pipeline_convolution_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_gemm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd23_gemm->create(LayerShaderType::convolution_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 16;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = shape_winograd_input_transformed_packed.cstep;
                specializations[3 + 1].i = shape_winograd_gemm_packed.w;
                specializations[3 + 2].i = shape_winograd_gemm_packed.cstep;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_3x3s1d1_winograd_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack4to1_3x3s1d1_winograd_gemm;

                pipeline_convolution_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                }
                pipeline_convolution_3x3s1d1_winograd23_gemm->create(shader_type_index, opt, specializations);
            }

            {
                std::vector<vk_specialization_type> specializations(5 + 6);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations[5 + 1].i = block_x;
                specializations[5 + 2].i = block_y;
                specializations[5 + 3].i = out_shape.w;
                specializations[5 + 4].i = out_shape.h;
                specializations[5 + 5].i = out_shape.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_output;

                pipeline_convolution_3x3s1d1_winograd23_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->create(shader_type_index, opt, specializations);
            }
        }
    }
    else if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (out_shape.dims == 3)
                size = out_shape.w * out_shape.h;

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
            // const int blocks_k = (num_input / 4 * maxk + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
            const int kk = (num_input * maxk + coopmat_K - 1) / coopmat_K;

            weight_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);
            for (int bn = 0; bn < blocks_n; bn++)
            {
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
            }
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            // unified pack4x4 weight layout: both input and output always packed by 4
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
        }
    }
    else if (is_conv1x1s1d1)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed) && num_input >= 8 && num_output >= 8;

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (shape_bordered.dims == 3)
                size = shape_bordered.w * shape_bordered.h;

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

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            // const int blocks_k = (num_input + coopmat_K * UNROLL_SG_K - 1) / (coopmat_K * UNROLL_SG_K);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;

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
            }
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            // unified pack4x4 weight layout: both input and output always packed by 4
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(maxk, num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.channel(q / 4);

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
        }
    }
    else
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        // unified pack4 weight layout: output channels always packed by 4
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
    }

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        // pass
    }
    else if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations(23 + 6);
            specializations[0].u32 = kernel_w;
            specializations[1].u32 = kernel_h;
            specializations[2].u32 = dilation_w;
            specializations[3].u32 = dilation_h;
            specializations[4].u32 = stride_w;
            specializations[5].u32 = stride_h;
            specializations[6].i = bias_term;
            specializations[7].i = activation_type;
            specializations[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[10].u32 = coopmat_M;
            specializations[11].u32 = coopmat_N;
            specializations[12].u32 = coopmat_K;
            specializations[13].u32 = coopmat_subgroup_size;
            specializations[14].u32 = UNROLL_SG_M;
            specializations[15].u32 = UNROLL_SG_N;
            specializations[16].u32 = UNROLL_SG_K;
            specializations[17].u32 = UNROLL_WG_M;
            specializations[18].u32 = UNROLL_WG_N;
            specializations[19].u32 = num_input;
            specializations[20].u32 = num_output;
            specializations[21].u32 = elempack;
            specializations[22].u32 = out_elempack;
            specializations[23 + 0].u32 = shape_bordered.w;
            specializations[23 + 1].u32 = shape_bordered.h;
            specializations[23 + 2].u32 = shape_bordered.cstep;
            specializations[23 + 3].u32 = out_shape.w;
            specializations[23 + 4].u32 = out_shape.h;
            specializations[23 + 5].u32 = out_shape.cstep;

            pipeline_convolution_gemm = new Pipeline(vkdev);
            pipeline_convolution_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_gemm->create(LayerShaderType::convolution_gemm_cm, opt, specializations);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            std::vector<vk_specialization_type> specializations(12 + 10);
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
            specializations[10].i = elempack;
            specializations[11].i = out_elempack;
            specializations[12 + 0].i = shape_bordered.w;
            specializations[12 + 1].i = shape_bordered.h;
            specializations[12 + 2].i = num_input_packed / 4;
            specializations[12 + 3].i = shape_bordered.cstep;
            specializations[12 + 4].i = out_shape.w;
            specializations[12 + 5].i = out_shape.h;
            specializations[12 + 6].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
            specializations[12 + 7].i = out_shape.dims != 0 ? out_shape.cstep : 0;
            specializations[12 + 8].i = num_output;
            specializations[12 + 9].i = num_input;

            Mat local_size_xyz(16, std::min(4, num_output_packed / 4), 1, (void*)0);
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(16, out_shape.w * out_shape.h);
                local_size_xyz.h = std::min(4, num_output_packed / 4);
            }

            pipeline_convolution_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_convolution_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_gemm->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_convolution_gemm->create(LayerShaderType::convolution_packed_gemm, opt, specializations);
        }
    }
    else if (is_conv1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
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
            specializations[17 + 0].u32 = shape_bordered.w * shape_bordered.h;
            specializations[17 + 1].u32 = shape_bordered.cstep;
            specializations[17 + 2].u32 = out_shape.cstep;

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_1x1s1d1->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution_1x1s1d1->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_1x1s1d1_cm, opt, specializations);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            // c = loop iterations = num_input_packed/4 for all elempacks
            // cstep = vec4 stride between channels
            //   elempack=4: shape_bordered.cstep (already vec4 units)
            //   elempack=1: shape_bordered.cstep (scalar units), but sfpvec4 stride = cstep_scalar/4
            const int c_packed = num_input_packed / 4;
            const int cstep_vec4 = (elempack == 4) ? (shape_bordered.dims != 0 ? shape_bordered.cstep : 0)
                                   : (shape_bordered.dims != 0 ? shape_bordered.cstep / 4 : 0);

            std::vector<vk_specialization_type> specializations(6 + 8);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[4].i = elempack;
            specializations[5].i = out_elempack;
            specializations[6 + 0].i = c_packed;
            specializations[6 + 1].i = cstep_vec4;
            specializations[6 + 2].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
            specializations[6 + 3].i = out_shape.dims != 0 ? (out_elempack == 4 ? out_shape.cstep : out_shape.cstep / 4) : 0;
            specializations[6 + 4].i = out_shape.dims != 0 ? out_shape.cstep / 4 : 0;
            specializations[6 + 5].i = out_shape.dims != 0 ? (out_shape.w * out_shape.h + 3) / 4 : 0;
            specializations[6 + 6].i = num_output;
            specializations[6 + 7].i = num_input;

            const int outc_pack4 = num_output_packed / 4;

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_convolution_1x1s1d1->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_1x1s1d1->set_local_size_xyz(8, std::min(8, outc_pack4), 1);
            }
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_packed_1x1s1d1, opt, specializations);
        }
    }
    else
    {
        const int num_output_packed = (num_output + 3) / 4 * 4;

        std::vector<vk_specialization_type> specializations(12 + 11);
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
        specializations[10].i = elempack;
        specializations[11].i = out_elempack;
        specializations[12 + 0].i = shape_bordered.dims;
        specializations[12 + 1].i = shape_bordered.w;
        specializations[12 + 2].i = shape_bordered.h;
        specializations[12 + 3].i = shape_bordered.c;
        specializations[12 + 4].i = shape_bordered.cstep;
        specializations[12 + 5].i = out_shape.dims;
        specializations[12 + 6].i = out_shape.w;
        specializations[12 + 7].i = out_shape.h;
        specializations[12 + 8].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
        specializations[12 + 9].i = out_shape.dims != 0 ? (out_elempack == 4 ? out_shape.cstep : out_shape.cstep * 4) : 0;
        specializations[12 + 10].i = num_output;

        const int outc_pack4 = num_output_packed / 4;

        Mat local_size_xyz(8, 8, std::min(4, (outc_pack4 + 1) / 2), (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(8, out_shape.w);
            local_size_xyz.h = std::min(8, out_shape.h);
            local_size_xyz.c = std::min(4, (outc_pack4 + 1) / 2);
        }

        pipeline_convolution = new Pipeline(vkdev);
        pipeline_convolution->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution->create(LayerShaderType::convolution_packed, opt, specializations);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Convolution_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution;
    pipeline_convolution = 0;

    delete pipeline_convolution_1x1s1d1;
    pipeline_convolution_1x1s1d1 = 0;

    delete pipeline_convolution_gemm;
    pipeline_convolution_gemm = 0;

    delete pipeline_convolution_3x3s1d1_winograd23_transform_input;
    delete pipeline_convolution_3x3s1d1_winograd23_gemm;
    delete pipeline_convolution_3x3s1d1_winograd23_transform_output;
    pipeline_convolution_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd23_transform_output = 0;

    delete pipeline_convolution_3x3s1d1_winograd43_transform_input;
    delete pipeline_convolution_3x3s1d1_winograd43_gemm;
    delete pipeline_convolution_3x3s1d1_winograd43_transform_output;
    pipeline_convolution_3x3s1d1_winograd43_transform_input = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm = 0;
    pipeline_convolution_3x3s1d1_winograd43_transform_output = 0;

#if NCNN_INT8
    delete pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm;
    delete pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm;
    pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm = 0;
    pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm = 0;
#endif

    // fc
    if (reshape_1x1xw)
    {
        reshape_1x1xw->destroy_pipeline(opt);
        delete reshape_1x1xw;
        reshape_1x1xw = 0;
    }

    if (reshape_w)
    {
        reshape_w->destroy_pipeline(opt);
        delete reshape_w;
        reshape_w = 0;
    }

    use_cooperative_matrix = false;
#if NCNN_INT8
    if (quantize)
    {
        quantize->destroy_pipeline(opt);
        delete quantize;
        quantize = 0;
    }
#endif
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

int Convolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return upload_model_int8(cmd, opt);
    }
#endif

    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        // winograd43
        if (opt.use_winograd43_convolution)
        {
            cmd.record_upload(weight_winograd43_data_packed, weight_data_gpu_tm_winograd43, opt);

            weight_winograd43_data_packed.release();
        }

        // winograd23
        if (opt.use_winograd23_convolution)
        {
            cmd.record_upload(weight_winograd23_data_packed, weight_data_gpu_tm_winograd23, opt);

            weight_winograd23_data_packed.release();
        }
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

        weight_data_packed.release();
    }

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);

        bias_data.release();
    }

    return 0;
}

int Convolution_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blob, top_blob, cmd, opt);
    }
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        NCNN_LOGE("Convolution 1d input compatibility path is deprecated and will be removed, please replace this layer with InnerProduct");
        NCNN_LOGE("ncnn param suggestion: Convolution ... 0=%d 1=1 11=1 5=%d 6=%d 8=%d 9=%d 10=... -> InnerProduct ... 0=%d 1=%d 2=%d 8=%d 9=%d 10=...", num_output, bias_term, weight_data_size, int8_scale_term, activation_type, num_output, bias_term, weight_data_size, int8_scale_term, activation_type);

        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            VkMat bottom_blob_1x1xw;
            {
                Option opt_reshape = opt;
                opt_reshape.blob_vkallocator = opt.workspace_vkallocator;
                reshape_1x1xw->forward(bottom_blob, bottom_blob_1x1xw, cmd, opt_reshape);
            }

            if (bottom_blob_1x1xw.empty())
                return -100;

            VkMat top_blob_1x1xw;
            int ret = forward(bottom_blob_1x1xw, top_blob_1x1xw, cmd, opt);
            if (ret != 0)
                return ret;

            return reshape_w->forward(top_blob_1x1xw, top_blob, cmd, opt);
        }
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
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
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
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
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int maxk = kernel_w * kernel_h;
    const int num_input = channels * elempack;

    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;

    if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        bool pre_winograd43 = opt.use_winograd43_convolution;
        if (opt.use_winograd23_convolution)
        {
            if (vkdev->info.type() == 0 && ((w <= 18 && h <= 18) || ((w >= 23 && w <= 24) && (h >= 23 && h <= 24))))
                pre_winograd43 = false;
            if (vkdev->info.type() != 0 && (w <= 12 && h <= 12))
                pre_winograd43 = false;

            if (use_cooperative_matrix && (w <= 18 && h <= 18))
                pre_winograd43 = false;
        }

        if (pre_winograd43)
        {
            // winograd43
            int block_x = (outw + 3) / 4;
            int block_y = (outh + 3) / 4;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x * block_y, 1, channels * 36, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.h;
                constants[2].i = bottom_blob_bordered.cstep;
                constants[3].i = bottom_tm_blob.cstep;
                constants[4].i = block_x;
                constants[5].i = block_y;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = channels;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x * block_y, 1, num_output / out_elempack * 36, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                    const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 36;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd43;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 36;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = block_y;
                constants[3].i = top_blob.w;
                constants[4].i = top_blob.h;
                constants[5].i = top_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = top_blob.c;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd43_transform_output, bindings, constants, dispatcher);
            }
        }
        else
        {
            // winograd23
            int block_x = (outw + 1) / 2;
            int block_y = (outh + 1) / 2;

            // transform input
            VkMat bottom_tm_blob;
            {
                bottom_tm_blob.create(block_x * block_y, 1, channels * 16, elemsize, elempack, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(2);
                bindings[0] = bottom_blob_bordered;
                bindings[1] = bottom_tm_blob;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = bottom_blob_bordered.w;
                constants[1].i = bottom_blob_bordered.h;
                constants[2].i = bottom_blob_bordered.cstep;
                constants[3].i = bottom_tm_blob.cstep;
                constants[4].i = block_x;
                constants[5].i = block_y;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = channels;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_transform_input, bindings, constants, dispatcher);
            }

            // gemm
            VkMat top_tm_blob;
            {
                top_tm_blob.create(block_x * block_y, 1, num_output / out_elempack * 16, out_elemsize, out_elempack, opt.workspace_vkallocator);
                if (top_tm_blob.empty())
                    return -100;

                if (use_cooperative_matrix)
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.w;
                    constants[1].i = bottom_tm_blob.cstep;
                    constants[2].i = top_tm_blob.cstep;

                    const int blocks_x = (bottom_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                    const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                    VkMat dispatcher;
                    dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                    dispatcher.h = 1;
                    dispatcher.c = 16;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<VkMat> bindings(3);
                    bindings[0] = bottom_tm_blob;
                    bindings[1] = top_tm_blob;
                    bindings[2] = weight_data_gpu_tm_winograd23;

                    std::vector<vk_constant_type> constants(3);
                    constants[0].i = bottom_tm_blob.cstep;
                    constants[1].i = top_tm_blob.w;
                    constants[2].i = top_tm_blob.cstep;

                    VkMat dispatcher;
                    dispatcher.w = (top_tm_blob.w + 3) / 4;
                    dispatcher.h = num_output / out_elempack;
                    dispatcher.c = 16;

                    cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_gemm, bindings, constants, dispatcher);
                }
            }

            // transform output
            {
                top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
                if (top_blob.empty())
                    return -100;

                std::vector<VkMat> bindings(3);
                bindings[0] = top_tm_blob;
                bindings[1] = top_blob;
                bindings[2] = bias_data_gpu;

                std::vector<vk_constant_type> constants(6);
                constants[0].i = top_tm_blob.cstep;
                constants[1].i = block_x;
                constants[2].i = block_y;
                constants[3].i = top_blob.w;
                constants[4].i = top_blob.h;
                constants[5].i = top_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = block_x;
                dispatcher.h = block_y;
                dispatcher.c = top_blob.c;

                cmd.record_pipeline(pipeline_convolution_3x3s1d1_winograd23_transform_output, bindings, constants, dispatcher);
            }
        }

        return 0;
    }
    if (opt.use_sgemm_convolution && !is_conv1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        // gemm
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.h;
            constants[2].u32 = bottom_blob_bordered.cstep;
            constants[3].u32 = top_blob.w;
            constants[4].u32 = top_blob.h;
            constants[5].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = bottom_blob_bordered;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;

            std::vector<vk_constant_type> constants(10);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = num_input_packed / 4;
            constants[3].i = bottom_blob_bordered.cstep;
            constants[4].i = top_blob.w;
            constants[5].i = top_blob.h;
            constants[6].i = num_output_packed / 4;
            constants[7].i = top_blob.cstep;
            constants[8].i = num_output;
            constants[9].i = num_input;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
            dispatcher.h = num_output_packed / 4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }

        return 0;
    }
    else if (is_conv1x1s1d1)
    {
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(3);
            constants[0].u32 = bottom_blob_bordered.w * bottom_blob_bordered.h;
            constants[1].u32 = bottom_blob_bordered.cstep;
            constants[2].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;
            const int outc_pack4 = num_output_packed / 4;
            const int c_packed = num_input_packed / 4;
            const int cstep_vec4 = (elempack == 4) ? bottom_blob_bordered.cstep : (bottom_blob_bordered.cstep / 4);
            const int size = (top_blob.w * top_blob.h + 3) / 4;

            // outcstep: for out_elempack=4, vec4 cstep; for out_elempack=1, scalar cstep as vec4 count
            const int outcstep_vec4 = (out_elempack == 4) ? top_blob.cstep : (top_blob.cstep / 4);
            const int outcstep_native = top_blob.cstep / 4;

            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = c_packed;
            constants[1].i = cstep_vec4;
            constants[2].i = outc_pack4;
            constants[3].i = outcstep_vec4;
            constants[4].i = outcstep_native;
            constants[5].i = size;
            constants[6].i = num_output;
            constants[7].i = num_input;

            VkMat dispatcher;
            dispatcher.w = size;
            dispatcher.h = outc_pack4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
        }

        return 0;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // for the unified shader, outc and outcstep are in pack4 units
    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int outc_pack4 = num_output_packed / 4;
    const int outcstep_pack4 = (out_elempack == 4) ? top_blob.cstep : (top_blob.cstep * 4);

    std::vector<VkMat> bindings(6);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob_bordered;
    bindings[3] = top_blob;
    bindings[4] = weight_data_gpu;
    bindings[5] = bias_data_gpu;

    std::vector<vk_constant_type> constants(11);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = outc_pack4;
    constants[9].i = outcstep_pack4;
    constants[10].i = num_output;

    VkMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (top_blob.h + 1) / 2;
    dispatcher.c = (outc_pack4 + 1) / 2;

    cmd.record_pipeline(pipeline_convolution, bindings, constants, dispatcher);

    return 0;
}

#if NCNN_INT8
int Convolution_vulkan::create_pipeline_int8(const Option& opt)
{
    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // skip fc like hint
    if (shape.dims != 3) shape = Mat();
    if (out_shape.dims != 3) out_shape = Mat();

    if (weight_data.elemsize != (size_t)1u)
    {
        NCNN_LOGE("Convolution_vulkan int8 weight data is not int8");
        return -1;
    }

    Option opt_int8 = opt;
    opt_int8.use_fp16_arithmetic = false;
    opt_int8.use_int16_packed = false;
    opt_int8.use_int16_storage = false;
    const bool use_int8_requantize = int8_scale_term > 100;
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;
    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    bool use_winograd = opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16;
    bool use_winograd43 = use_winograd && opt.use_winograd43_convolution;
    bool use_winograd23 = use_winograd && opt.use_winograd23_convolution;
    bool use_gemm = opt.use_sgemm_convolution && !is_conv1x1s1d1 && !use_winograd && num_input * maxk >= 8 && num_output >= 8;
    const int elempack = opt.use_packing_layout && num_input % 4 == 0 ? 4 : 1;
    const int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;

    Mat shape_int8;
    if (shape.dims == 3)
    {
        shape_int8 = Mat(shape.w, shape.h, num_input / elempack, (void*)0, (size_t)elempack, elempack);
    }

    Mat shape_int8_bordered;
    if (shape_int8.dims == 3)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_int8_bordered = Mat(shape_int8.w + pad_left + pad_right, shape_int8.h + pad_top + pad_bottom, shape_int8.c, (void*)0, shape_int8.elemsize, shape_int8.elempack);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape_int8.w - 1) / stride_w * stride_w - shape_int8.w;
            int hpad = kernel_extent_h + (shape_int8.h - 1) / stride_h * stride_h - shape_int8.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_int8_bordered = Mat(shape_int8.w + wpad, shape_int8.h + hpad, shape_int8.c, (void*)0, shape_int8.elemsize, shape_int8.elempack);
            }
            else
            {
                shape_int8_bordered = shape_int8;
            }
        }
        else
        {
            shape_int8_bordered = shape_int8;
        }
    }

    if (shape_int8_bordered.dims == 3 && use_winograd43 && use_winograd23)
    {
        const int w_bordered = shape_int8_bordered.w;
        const int h_bordered = shape_int8_bordered.h;

        bool prefer_winograd43 = true;
        if (vkdev->info.type() == 0 && ((w_bordered <= 18 && h_bordered <= 18) || ((w_bordered >= 23 && w_bordered <= 24) && (h_bordered >= 23 && h_bordered <= 24))))
            prefer_winograd43 = false;
        if (vkdev->info.type() != 0 && (w_bordered <= 12 && h_bordered <= 12))
            prefer_winograd43 = false;

        use_winograd43 = prefer_winograd43;
        use_winograd23 = !prefer_winograd43;
    }

    Mat shape_padding_int8_bordered;
    if (shape_int8_bordered.dims == 3)
    {
        const int padding_outc = shape_int8_bordered.c * shape_int8_bordered.elempack;
        const int padding_out_elempack = padding_outc % 4 == 0 ? 4 : 1;
        const size_t padding_out_elemsize = shape_int8_bordered.elemsize / shape_int8_bordered.elempack * padding_out_elempack;
        shape_padding_int8_bordered = Mat(shape_int8_bordered.w, shape_int8_bordered.h, padding_outc / padding_out_elempack, (void*)0, padding_out_elemsize, padding_out_elempack);
    }

    Mat out_shape_blob;
    if (out_shape.dims == 3)
    {
        size_t out_elemsize;
        if (use_int8_requantize)
        {
            out_elemsize = out_elempack;
        }
        else if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
        {
            out_elemsize = (size_t)2u * out_elempack;
        }
        else
        {
            out_elemsize = (size_t)4u * out_elempack;
        }

        out_shape_blob = Mat(out_shape.w, out_shape.h, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);
    }

    {
        quantize = ncnn::create_layer_vulkan(ncnn::LayerType::Quantize);
        quantize->vkdev = vkdev;

        Mat shape_quantize;
        Mat out_shape_quantize;
        if (shape.dims == 3)
        {
            const size_t shape_elemsize = shape.elemsize / shape.elempack * elempack;
            shape_quantize = Mat(shape.w, shape.h, num_input / elempack, (void*)0, shape_elemsize, elempack);
            out_shape_quantize = shape_int8;
        }

        quantize->bottom_shapes.resize(1);
        quantize->bottom_shapes[0] = shape_quantize;
        quantize->top_shapes.resize(1);
        quantize->top_shapes[0] = out_shape_quantize;

        ncnn::ParamDict pd;
        pd.set(0, 1);
        quantize->load_param(pd);

        Mat weights[1];
        weights[0] = bottom_blob_int8_scales;
        quantize->load_model(ModelBinFromMatArray(weights));

        Option opt_quantize = opt;
        opt_quantize.use_fp16_arithmetic = false;

        quantize->create_pipeline(opt_quantize);
    }

    {
        padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape_int8;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_padding_int8_bordered;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, 0.f);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    const int num_input_packed = (num_input + 3) / 4 * 4;
    const int num_output_packed = (num_output + 3) / 4 * 4;
    const int c_packed = num_input_packed / 4;
    const int outc_pack4 = num_output_packed / 4;
    const int c_shader = use_gemm || elempack == 4 ? c_packed : num_input;

    std::vector<vk_specialization_type> specializations(13 + 10);
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
    specializations[10].i = use_int8_requantize ? 1 : 0;
    specializations[11].i = elempack;
    specializations[12].i = out_elempack;
    specializations[13 + 0].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.w : 0;
    specializations[13 + 1].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.h : 0;
    specializations[13 + 2].i = shape_int8_bordered.dims != 0 ? c_shader : 0;
    specializations[13 + 3].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.cstep : 0;
    specializations[13 + 4].i = out_shape_blob.dims != 0 ? out_shape_blob.w : 0;
    specializations[13 + 5].i = out_shape_blob.dims != 0 ? out_shape_blob.h : 0;
    specializations[13 + 6].i = out_shape_blob.dims != 0 ? outc_pack4 : 0;
    specializations[13 + 7].i = out_shape_blob.dims != 0 ? (out_elempack == 4 ? out_shape_blob.cstep : out_shape_blob.cstep * 4) : 0;
    specializations[13 + 8].i = num_output;
    specializations[13 + 9].i = num_input;

    const bool use_int8_winograd_int16_storage = use_winograd && opt.use_int16_storage && vkdev->info.support_int16_storage() && vkdev->info.support_int16_arithmetic();
    if (use_winograd)
    {
        if (use_int8_winograd_int16_storage)
            opt_int8.use_int16_storage = true;
        else
            opt_int8.use_int16_packed = true;
    }

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;

    if (use_winograd && opt.use_cooperative_matrix && opt.use_int8_arithmetic && vkdev->info.support_int8_cooperative_matrix())
    {
        int M = 1024;
        if (out_shape.dims == 3)
        {
            const int block_x = use_winograd43 ? (out_shape.w + 3) / 4 : (out_shape.w + 1) / 2;
            const int block_y = use_winograd43 ? (out_shape.h + 3) / 4 : (out_shape.h + 1) / 2;
            M = block_x * block_y;
        }

        const int N = num_output;
        const int K = num_input;

        coopmat_subgroup_size = vkdev->info.querySubgroupProperties().subgroupSize;

        double min_cost = 1e300;

        if (vkdev->info.support_VK_KHR_cooperative_matrix() && vkdev->info.queryCooperativeMatrixFeatures().cooperativeMatrix)
        {
            const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = vkdev->info.queryCooperativeMatrixSubProperties();
            for (size_t i = 0; i < properties.size(); i++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = properties[i];
                if (cmp.AType == VK_COMPONENT_TYPE_SINT8_KHR && cmp.BType == VK_COMPONENT_TYPE_SINT8_KHR
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_SINT32_KHR
                        && cmp.scope == VK_SCOPE_SUBGROUP_KHR
                        && cmp.MSize % 4 == 0 && cmp.NSize % 4 == 0 && cmp.KSize % 4 == 0)
                {
                    const int M_pad = (M + cmp.MSize - 1) / cmp.MSize * cmp.MSize;
                    const int N_pad = (N + cmp.NSize - 1) / cmp.NSize * cmp.NSize;
                    const int K_pad = (K + cmp.KSize - 1) / cmp.KSize * cmp.KSize;

                    double cost = (double)M_pad * N_pad * K_pad - (double)M * N * K;
                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        coopmat_M = cmp.MSize;
                        coopmat_N = cmp.NSize;
                        coopmat_K = cmp.KSize;
                    }
                }
            }
        }
        else if (vkdev->info.support_VK_NV_cooperative_matrix() && vkdev->info.queryCooperativeMatrixFeaturesNV().cooperativeMatrix)
        {
            const std::vector<VkCooperativeMatrixPropertiesNV>& properties = vkdev->info.queryCooperativeMatrixSubPropertiesNV();
            for (size_t i = 0; i < properties.size(); i++)
            {
                const VkCooperativeMatrixPropertiesNV& cmp = properties[i];
                if (cmp.AType == VK_COMPONENT_TYPE_SINT8_NV && cmp.BType == VK_COMPONENT_TYPE_SINT8_NV
                        && cmp.CType == VK_COMPONENT_TYPE_SINT32_NV && cmp.DType == VK_COMPONENT_TYPE_SINT32_NV
                        && cmp.scope == VK_SCOPE_SUBGROUP_NV
                        && cmp.MSize % 4 == 0 && cmp.NSize % 4 == 0 && cmp.KSize % 4 == 0)
                {
                    const int M_pad = (M + cmp.MSize - 1) / cmp.MSize * cmp.MSize;
                    const int N_pad = (N + cmp.NSize - 1) / cmp.NSize * cmp.NSize;
                    const int K_pad = (K + cmp.KSize - 1) / cmp.KSize * cmp.KSize;

                    double cost = (double)M_pad * N_pad * K_pad - (double)M * N * K;
                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        coopmat_M = cmp.MSize;
                        coopmat_N = cmp.NSize;
                        coopmat_K = cmp.KSize;
                    }
                }
            }
        }

        if (coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0)
        {
            use_cooperative_matrix = true;

            UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);
        }
    }
    else if ((is_conv1x1s1d1 || use_gemm) && opt.use_cooperative_matrix && opt.use_int8_arithmetic && vkdev->info.support_int8_cooperative_matrix())
    {
        const int M = out_shape.dims == 3 ? out_shape.w * out_shape.h : 1024;
        const int N = num_output;
        const int K = is_conv1x1s1d1 ? num_input : num_input * maxk;

        if (N >= 8 && K >= 8)
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }

        if (coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0)
        {
            use_cooperative_matrix = true;

            UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);
        }
    }

    if (is_conv1x1s1d1)
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        if (use_cooperative_matrix)
        {
            const signed char* weight_data_ptr = weight_data;

            const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (num_input + coopmat_K - 1) / coopmat_K;
            const int kk_padded = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K * UNROLL_SG_K;
            const int coopmat_Nd4 = coopmat_N / 4;
            const int coopmat_Nd4p = coopmat_Nd4 + (vkdev->info.support_VK_KHR_cooperative_matrix() ? 1 : 0);

            const int weight_data_int8_packed_size = coopmat_Nd4p * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk_padded;
            weight_data_int8_packed.create(weight_data_int8_packed_size, blocks_n, (size_t)4u, 4);
            if (weight_data_int8_packed.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bn = 0; bn < blocks_n; bn++)
            {
                signed char* p = weight_data_int8_packed.row<signed char>(bn);

                for (int k = 0; k < kk_padded; k += UNROLL_SG_K)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_Nd4p; j++)
                                    {
                                        for (int jj = 0; jj < 4; jj++)
                                        {
                                            const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j * 4 + jj;
                                            const int gki = (k + zk) * coopmat_K + i;

                                            *p++ = j < coopmat_Nd4 && gni < num_output && gki < num_input ? weight_data_ptr[gni * num_input + gki] : 0;
                                        }
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
            const int num_output_pack4_aligned = ((num_output + 3) / 4 + 7) / 8 * 8;
            const int num_output_packed = num_output_pack4_aligned * 4;

            weight_data_int8_packed.create(maxk, num_input_packed / 4, num_output_pack4_aligned, (size_t)4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                signed char* g00 = weight_data_int8_packed.channel(q / 4);

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
                                    const signed char* k00 = weight_data_r2.channel(q + i).row<const signed char>(p + j);
                                    g00[0] = k00[k];
                                }
                                else
                                {
                                    g00[0] = 0;
                                }
                                g00++;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        if (use_gemm)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            if (use_cooperative_matrix)
            {
                const int K = num_input * maxk;
                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                const int kk = (K + coopmat_K - 1) / coopmat_K;
                const int kk_padded = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K * UNROLL_SG_K;
                const int coopmat_Nd4 = coopmat_N / 4;
                const int coopmat_Nd4p = coopmat_Nd4 + (vkdev->info.support_VK_KHR_cooperative_matrix() ? 1 : 0);

                const int weight_data_int8_packed_size = coopmat_Nd4p * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk_padded;
                weight_data_int8_packed.create(weight_data_int8_packed_size, blocks_n, (size_t)4u, 4);
                if (weight_data_int8_packed.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bn = 0; bn < blocks_n; bn++)
                {
                    signed char* g00 = weight_data_int8_packed.row<signed char>(bn);

                    for (int k = 0; k < kk_padded; k += UNROLL_SG_K)
                    {
                        for (int wn = 0; wn < UNROLL_WG_N; wn++)
                        {
                            for (int zk = 0; zk < UNROLL_SG_K; zk++)
                            {
                                for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                {
                                    for (int i = 0; i < coopmat_K; i++)
                                    {
                                        for (int j = 0; j < coopmat_Nd4p; j++)
                                        {
                                            for (int jj = 0; jj < 4; jj++)
                                            {
                                                const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j * 4 + jj;
                                                const int gki = (k + zk) * coopmat_K + i;

                                                if (j < coopmat_Nd4 && gni < num_output && gki < K)
                                                {
                                                    int p = gki / maxk;
                                                    int k0 = gki - p * maxk;

                                                    if (elempack == 4)
                                                    {
                                                        const int gki4 = gki / 4;
                                                        p = (gki4 / maxk) * 4 + gki % 4;
                                                        k0 = gki4 % maxk;
                                                    }

                                                    const signed char* k00 = weight_data_r2.channel(gni).row<const signed char>(p);
                                                    g00[0] = k00[k0];
                                                }
                                                else
                                                {
                                                    g00[0] = 0;
                                                }
                                                g00++;
                                            }
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
                const int num_output_pack4_aligned = ((num_output + 3) / 4 + 7) / 8 * 8;

                weight_data_int8_packed.create(maxk, c_packed, num_output_pack4_aligned, (size_t)4 * 4, 4 * 4);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < num_output_pack4_aligned; q++)
                {
                    signed char* g00 = weight_data_int8_packed.channel(q);

                    for (int p = 0; p < num_input_packed; p += 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            for (int i = 0; i < 4; i++)
                            {
                                for (int j = 0; j < 4; j++)
                                {
                                    if (q * 4 + i < num_output && p + j < num_input)
                                    {
                                        const signed char* k00 = weight_data_r2.channel(q * 4 + i).row<const signed char>(p + j);
                                        g00[0] = k00[k];
                                    }
                                    else
                                    {
                                        g00[0] = 0;
                                    }
                                    g00++;
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (!use_winograd)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            const int num_output_pack4_aligned = ((num_output + 3) / 4 + 7) / 8 * 8;

            if (elempack == 4)
            {
                weight_data_int8_packed.create(maxk, c_packed, num_output_pack4_aligned, (size_t)4 * 4, 4 * 4);
            }
            else
            {
                weight_data_int8_packed.create(maxk * num_input, num_output_pack4_aligned, (size_t)4u, 4);
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < num_output_pack4_aligned; q++)
            {
                signed char* g00 = elempack == 4 ? weight_data_int8_packed.channel(q) : weight_data_int8_packed.row<signed char>(q);

                const int num_input_loop = elempack == 4 ? num_input_packed : num_input;
                for (int p = 0; p < num_input_loop; p += elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < elempack; j++)
                            {
                                if (q * 4 + i < num_output && p + j < num_input)
                                {
                                    const signed char* k00 = weight_data_r2.channel(q * 4 + i).row<const signed char>(p + j);
                                    g00[0] = k00[k];
                                }
                                else
                                {
                                    g00[0] = 0;
                                }
                                g00++;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            weight_data_int8_packed = weight_data.reshape(weight_data_size);
        }
    }

    {
        int num_output_packed = (num_output + 3) / 4 * 4;
        if (!is_conv1x1s1d1 && !use_winograd && !use_gemm)
            num_output_packed = (num_output + 7) / 8 * 8;

        const float bottom_blob_int8_scale = bottom_blob_int8_scales.empty() ? 1.f : bottom_blob_int8_scales[0];
        const float bottom_blob_int8_descale = bottom_blob_int8_scale == 0.f ? 0.f : 1.f / bottom_blob_int8_scale;

        if (is_conv1x1s1d1)
            weight_data_int8_descales.create(num_output_packed / 4, (size_t)4u * 4, 4);
        else
            weight_data_int8_descales.create(num_output_packed, (size_t)4u, 1);
        if (weight_data_int8_descales.empty())
            return -100;

        float* outptr = weight_data_int8_descales;
        for (int q = 0; q < num_output_packed; q += 4)
        {
            float scale0 = q + 0 < num_output ? weight_data_int8_scales[q + 0] : 0.f;
            float scale1 = q + 1 < num_output ? weight_data_int8_scales[q + 1] : 0.f;
            float scale2 = q + 2 < num_output ? weight_data_int8_scales[q + 2] : 0.f;
            float scale3 = q + 3 < num_output ? weight_data_int8_scales[q + 3] : 0.f;
            outptr[0] = scale0 == 0.f ? 0.f : bottom_blob_int8_descale / scale0;
            outptr[1] = scale1 == 0.f ? 0.f : bottom_blob_int8_descale / scale1;
            outptr[2] = scale2 == 0.f ? 0.f : bottom_blob_int8_descale / scale2;
            outptr[3] = scale3 == 0.f ? 0.f : bottom_blob_int8_descale / scale3;
            outptr += 4;
        }
    }

    if (bias_term)
    {
        int num_output_packed = (num_output + 3) / 4 * 4;
        if (!is_conv1x1s1d1 && !use_winograd && !use_gemm)
            num_output_packed = (num_output + 7) / 8 * 8;

        bias_data_int8_packed.create(num_output_packed, (size_t)4u, 1);
        if (bias_data_int8_packed.empty())
            return -100;

        float* outptr = bias_data_int8_packed;
        for (int q = 0; q < num_output_packed; q += 4)
        {
            outptr[0] = q + 0 < num_output ? bias_data[q + 0] : 0.f;
            outptr[1] = q + 1 < num_output ? bias_data[q + 1] : 0.f;
            outptr[2] = q + 2 < num_output ? bias_data[q + 2] : 0.f;
            outptr[3] = q + 3 < num_output ? bias_data[q + 3] : 0.f;
            outptr += 4;
        }
    }

    if (use_winograd)
    {
        if (use_winograd43)
        {
            Mat weight_data_tm;
            weight_data_tm.create(36, num_input, num_output);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const signed char* kernel0 = (const signed char*)weight_data + p * num_input * 9 + q * 9;
                    int* kernel_tm0 = weight_data_tm.channel(p).row<int>(q);

                    int tmp[6][3];
                    for (int m = 0; m < 3; m++)
                    {
                        const int r0 = kernel0[0];
                        const int r1 = kernel0[1];
                        const int r2 = kernel0[2];

                        tmp[0][m] = r0 * 6;
                        tmp[1][m] = -r0 * 4 - r1 * 4 - r2 * 4;
                        tmp[2][m] = -r0 * 4 + r1 * 4 - r2 * 4;
                        tmp[3][m] = r0 + r1 * 2 + r2 * 4;
                        tmp[4][m] = r0 - r1 * 2 + r2 * 4;
                        tmp[5][m] = r2 * 6;

                        kernel0 += 3;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        const int r0 = tmp[m][0];
                        const int r1 = tmp[m][1];
                        const int r2 = tmp[m][2];

                        kernel_tm0[m * 6 + 0] = r0 * 6;
                        kernel_tm0[m * 6 + 1] = -r0 * 4 - r1 * 4 - r2 * 4;
                        kernel_tm0[m * 6 + 2] = -r0 * 4 + r1 * 4 - r2 * 4;
                        kernel_tm0[m * 6 + 3] = r0 + r1 * 2 + r2 * 4;
                        kernel_tm0[m * 6 + 4] = r0 - r1 * 2 + r2 * 4;
                        kernel_tm0[m * 6 + 5] = r2 * 6;
                    }
                }
            }

            {
                if (use_cooperative_matrix)
                {
                    const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                    const int kk = (num_input + coopmat_K - 1) / coopmat_K;
                    const int kk_padded = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K * UNROLL_SG_K;
                    const int coopmat_Nd4 = coopmat_N / 4;
                    const int coopmat_Nd4p = coopmat_Nd4 + (vkdev->info.support_VK_KHR_cooperative_matrix() ? 1 : 0);

                    const int weight_data_int8_packed_size = coopmat_Nd4p * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk_padded;
                    weight_winograd43_data_int8_packed_cm.create(weight_data_int8_packed_size, blocks_n, 36, (size_t)8u, 1);
                    if (weight_winograd43_data_int8_packed_cm.empty())
                        return -100;

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < 36; b++)
                    {
                        for (int bn = 0; bn < blocks_n; bn++)
                        {
                            signed char* p0 = weight_winograd43_data_int8_packed_cm.channel(b).row<signed char>(bn);

                            for (int k = 0; k < kk_padded; k += UNROLL_SG_K)
                            {
                                for (int wn = 0; wn < UNROLL_WG_N; wn++)
                                {
                                    for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                    {
                                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                        {
                                            for (int i = 0; i < coopmat_K; i++)
                                            {
                                                for (int j = 0; j < coopmat_Nd4p; j++)
                                                {
                                                    for (int jj = 0; jj < 4; jj++)
                                                    {
                                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j * 4 + jj;
                                                        const int gki = (k + zk) * coopmat_K + i;

                                                        const int v = j < coopmat_Nd4 && gni < num_output && gki < num_input ? weight_data_tm.channel(gni).row<const int>(gki)[b] : 0;
                                                        int vlow = v & 255;
                                                        if (vlow >= 128) vlow -= 256;
                                                        p0[jj] = (signed char)vlow;
                                                        p0[4 + jj] = (signed char)((v - vlow) >> 8);
                                                    }
                                                    p0 += 8;
                                                }
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
                    const int num_input_packed = (num_input + 3) / 4 * 4;
                    const int num_output_packed = (num_output + 3) / 4 * 4;
                    const int c4 = num_input_packed / 4;

                    weight_winograd43_data_int8_packed.create(c4, num_output_packed, 36, (size_t)8u, 1);

                    for (int k = 0; k < 36; k++)
                    {
                        int* g00 = weight_winograd43_data_int8_packed.channel(k);

                        for (int p = 0; p < num_output_packed; p++)
                        {
                            const int* k0 = p < num_output ? weight_data_tm.channel(p) : 0;

                            for (int q = 0; q < num_input_packed; q += 4)
                            {
                                const int v0 = k0 && q + 0 < num_input ? k0[(q + 0) * 36 + k] : 0;
                                const int v1 = k0 && q + 1 < num_input ? k0[(q + 1) * 36 + k] : 0;
                                const int v2 = k0 && q + 2 < num_input ? k0[(q + 2) * 36 + k] : 0;
                                const int v3 = k0 && q + 3 < num_input ? k0[(q + 3) * 36 + k] : 0;

                                g00[0] = (int)(((unsigned int)(unsigned short)v0) | ((unsigned int)(unsigned short)v1 << 16));
                                g00[1] = (int)(((unsigned int)(unsigned short)v2) | ((unsigned int)(unsigned short)v3 << 16));
                                g00 += 2;
                            }
                        }
                    }
                }
            }

            {
                int block_x = 0;
                int block_y = 0;
                Mat shape_winograd_input_transformed;

                if (shape_int8_bordered.dims == 3 && out_shape_blob.dims == 3)
                {
                    block_x = (out_shape_blob.w + 3) / 4;
                    block_y = (out_shape_blob.h + 3) / 4;
                    if (use_cooperative_matrix)
                    {
                        if (elempack == 4)
                            shape_winograd_input_transformed = Mat(block_x * block_y * 2, 1, c_packed * 36, (void*)0, (size_t)4u, 4);
                        else
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, num_input * 36, (void*)0, (size_t)1u, 1);
                    }
                    else
                    {
                        if (elempack == 4)
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, c_packed * 36, (void*)0, (size_t)8u, 4);
                        else
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, num_input * 36, (void*)0, (size_t)2u, 1);
                    }
                }

                std::vector<vk_specialization_type> specializations_winograd_input(1 + 6);
                specializations_winograd_input[0].i = elempack == 4 ? c_packed : num_input;
                specializations_winograd_input[1 + 0].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.w : 0;
                specializations_winograd_input[1 + 1].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.h : 0;
                specializations_winograd_input[1 + 2].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.cstep : 0;
                specializations_winograd_input[1 + 3].i = shape_winograd_input_transformed.dims != 0 ? shape_winograd_input_transformed.cstep : 0;
                specializations_winograd_input[1 + 4].i = block_x;
                specializations_winograd_input[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (use_cooperative_matrix)
                {
                    if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_input_int8_cm;
                    if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_input_int8_cm;
                }
                else
                {
                    if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_input_int8;
                    if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_input_int8;
                }

                pipeline_convolution_3x3s1d1_winograd43_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_input->create(shader_type_index, opt_int8, specializations_winograd_input);
            }
            if (!use_cooperative_matrix)
            {
                // winograd23/43 share gemm shader, transform count is set by dispatcher.c
                pipeline_convolution_3x3s1d1_winograd43_gemm = new Pipeline(vkdev);
                {
                    std::vector<vk_specialization_type> specializations_winograd_gemm(5 + 3);
                    specializations_winograd_gemm[0].i = 36;
                    specializations_winograd_gemm[1].i = num_input;
                    specializations_winograd_gemm[2].i = num_output;
                    specializations_winograd_gemm[3].i = elempack;
                    specializations_winograd_gemm[4].i = out_elempack;
                    specializations_winograd_gemm[5 + 0].i = 0;
                    specializations_winograd_gemm[5 + 1].i = 0;
                    specializations_winograd_gemm[5 + 2].i = 0;

                    pipeline_convolution_3x3s1d1_winograd43_gemm->set_local_size_xyz(opt_int8.use_shader_local_memory ? 8 : 4, opt_int8.use_shader_local_memory ? 8 : std::min(4, (num_output + 3) / 4), opt_int8.use_shader_local_memory ? 1 : 4);
                    pipeline_convolution_3x3s1d1_winograd43_gemm->create(LayerShaderType::convolution_3x3s1d1_winograd_gemm_int8, opt_int8, specializations_winograd_gemm);
                }
            }
            else
            {
                std::vector<vk_specialization_type> specializations_winograd_gemm(15 + 3);
                specializations_winograd_gemm[0].u32 = 36;
                specializations_winograd_gemm[1].u32 = coopmat_M;
                specializations_winograd_gemm[2].u32 = coopmat_N;
                specializations_winograd_gemm[3].u32 = coopmat_K;
                specializations_winograd_gemm[4].u32 = UNROLL_SG_M;
                specializations_winograd_gemm[5].u32 = UNROLL_SG_N;
                specializations_winograd_gemm[6].u32 = UNROLL_SG_K;
                specializations_winograd_gemm[7].u32 = UNROLL_WG_M;
                specializations_winograd_gemm[8].u32 = UNROLL_WG_N;
                specializations_winograd_gemm[9].u32 = coopmat_subgroup_size;
                specializations_winograd_gemm[10].u32 = num_input;
                specializations_winograd_gemm[11].u32 = num_output;
                specializations_winograd_gemm[12].u32 = elempack;
                specializations_winograd_gemm[13].u32 = out_elempack;
                specializations_winograd_gemm[14].u32 = weight_winograd43_data_int8_packed_cm.cstep;
                specializations_winograd_gemm[15 + 0].u32 = 0;
                specializations_winograd_gemm[15 + 1].u32 = 0;
                specializations_winograd_gemm[15 + 2].u32 = 0;

                pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm->create(LayerShaderType::convolution_3x3s1d1_winograd_gemm_int8_cm, opt_int8, specializations_winograd_gemm);
            }
            {
                int block_x = 0;
                int block_y = 0;
                Mat shape_winograd_gemm_packed;

                if (out_shape_blob.dims == 3)
                {
                    block_x = (out_shape_blob.w + 3) / 4;
                    block_y = (out_shape_blob.h + 3) / 4;
                    shape_winograd_gemm_packed = Mat(block_x * block_y, 1, num_output / out_elempack * 36, (void*)0, (size_t)4u * out_elempack, out_elempack);
                }

                std::vector<vk_specialization_type> specializations_winograd_output(6 + 6);
                specializations_winograd_output[0].i = bias_term;
                specializations_winograd_output[1].i = activation_type;
                specializations_winograd_output[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations_winograd_output[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations_winograd_output[4].i = use_int8_requantize ? 1 : 0;
                specializations_winograd_output[5].i = num_output;
                specializations_winograd_output[6 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations_winograd_output[6 + 1].i = block_x;
                specializations_winograd_output[6 + 2].i = block_y;
                specializations_winograd_output[6 + 3].i = out_shape_blob.w;
                specializations_winograd_output[6 + 4].i = out_shape_blob.h;
                specializations_winograd_output[6 + 5].i = out_shape_blob.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd43_transform_output_int8;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd43_transform_output_int8;

                pipeline_convolution_3x3s1d1_winograd43_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->set_local_size_xyz(4, 4, 1);
                pipeline_convolution_3x3s1d1_winograd43_transform_output->create(shader_type_index, opt_int8, specializations_winograd_output);
            }
        }

        if (use_winograd23)
        {
            Mat weight_data_tm;
            weight_data_tm.create(16, num_input, num_output);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    const signed char* kernel0 = (const signed char*)weight_data + p * num_input * 9 + q * 9;
                    int* kernel_tm0 = weight_data_tm.channel(p).row<int>(q);

                    int tmp[4][3];
                    for (int m = 0; m < 3; m++)
                    {
                        const int r0 = kernel0[0];
                        const int r1 = kernel0[1];
                        const int r2 = kernel0[2];

                        tmp[0][m] = r0 * 2;
                        tmp[1][m] = r0 + r1 + r2;
                        tmp[2][m] = r0 - r1 + r2;
                        tmp[3][m] = r2 * 2;

                        kernel0 += 3;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        const int r0 = tmp[m][0];
                        const int r1 = tmp[m][1];
                        const int r2 = tmp[m][2];

                        kernel_tm0[m * 4 + 0] = r0 * 2;
                        kernel_tm0[m * 4 + 1] = r0 + r1 + r2;
                        kernel_tm0[m * 4 + 2] = r0 - r1 + r2;
                        kernel_tm0[m * 4 + 3] = r2 * 2;
                    }
                }
            }

            {
                if (use_cooperative_matrix)
                {
                    const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                    const int kk = (num_input + coopmat_K - 1) / coopmat_K;
                    const int kk_padded = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K * UNROLL_SG_K;
                    const int coopmat_Nd4 = coopmat_N / 4;
                    const int coopmat_Nd4p = coopmat_Nd4 + (vkdev->info.support_VK_KHR_cooperative_matrix() ? 1 : 0);

                    const int weight_data_int8_packed_size = coopmat_Nd4p * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk_padded;
                    weight_winograd23_data_int8_packed_cm.create(weight_data_int8_packed_size, blocks_n, 16, (size_t)8u, 1);
                    if (weight_winograd23_data_int8_packed_cm.empty())
                        return -100;

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < 16; b++)
                    {
                        for (int bn = 0; bn < blocks_n; bn++)
                        {
                            signed char* p0 = weight_winograd23_data_int8_packed_cm.channel(b).row<signed char>(bn);

                            for (int k = 0; k < kk_padded; k += UNROLL_SG_K)
                            {
                                for (int wn = 0; wn < UNROLL_WG_N; wn++)
                                {
                                    for (int zk = 0; zk < UNROLL_SG_K; zk++)
                                    {
                                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                        {
                                            for (int i = 0; i < coopmat_K; i++)
                                            {
                                                for (int j = 0; j < coopmat_Nd4p; j++)
                                                {
                                                    for (int jj = 0; jj < 4; jj++)
                                                    {
                                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j * 4 + jj;
                                                        const int gki = (k + zk) * coopmat_K + i;

                                                        const int v = j < coopmat_Nd4 && gni < num_output && gki < num_input ? weight_data_tm.channel(gni).row<const int>(gki)[b] : 0;
                                                        int vlow = v & 255;
                                                        if (vlow >= 128) vlow -= 256;
                                                        p0[jj] = (signed char)vlow;
                                                        p0[4 + jj] = (signed char)((v - vlow) >> 8);
                                                    }
                                                    p0 += 8;
                                                }
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
                    const int num_input_packed = (num_input + 3) / 4 * 4;
                    const int num_output_packed = (num_output + 3) / 4 * 4;
                    const int c4 = num_input_packed / 4;

                    weight_winograd23_data_int8_packed.create(c4, num_output_packed, 16, (size_t)8u, 1);

                    for (int k = 0; k < 16; k++)
                    {
                        int* g00 = weight_winograd23_data_int8_packed.channel(k);

                        for (int p = 0; p < num_output_packed; p++)
                        {
                            const int* k0 = p < num_output ? weight_data_tm.channel(p) : 0;

                            for (int q = 0; q < num_input_packed; q += 4)
                            {
                                const int v0 = k0 && q + 0 < num_input ? k0[(q + 0) * 16 + k] : 0;
                                const int v1 = k0 && q + 1 < num_input ? k0[(q + 1) * 16 + k] : 0;
                                const int v2 = k0 && q + 2 < num_input ? k0[(q + 2) * 16 + k] : 0;
                                const int v3 = k0 && q + 3 < num_input ? k0[(q + 3) * 16 + k] : 0;

                                g00[0] = (int)(((unsigned int)(unsigned short)v0) | ((unsigned int)(unsigned short)v1 << 16));
                                g00[1] = (int)(((unsigned int)(unsigned short)v2) | ((unsigned int)(unsigned short)v3 << 16));
                                g00 += 2;
                            }
                        }
                    }
                }
            }

            {
                int block_x = 0;
                int block_y = 0;
                Mat shape_winograd_input_transformed;

                if (shape_int8_bordered.dims == 3 && out_shape_blob.dims == 3)
                {
                    block_x = (out_shape_blob.w + 1) / 2;
                    block_y = (out_shape_blob.h + 1) / 2;
                    if (use_cooperative_matrix)
                    {
                        if (elempack == 4)
                            shape_winograd_input_transformed = Mat(block_x * block_y * 2, 1, c_packed * 16, (void*)0, (size_t)4u, 4);
                        else
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, num_input * 16, (void*)0, (size_t)1u, 1);
                    }
                    else
                    {
                        if (elempack == 4)
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, c_packed * 16, (void*)0, (size_t)8u, 4);
                        else
                            shape_winograd_input_transformed = Mat(block_x * block_y, 1, num_input * 16, (void*)0, (size_t)2u, 1);
                    }
                }

                std::vector<vk_specialization_type> specializations_winograd_input(1 + 6);
                specializations_winograd_input[0].i = elempack == 4 ? c_packed : num_input;
                specializations_winograd_input[1 + 0].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.w : 0;
                specializations_winograd_input[1 + 1].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.h : 0;
                specializations_winograd_input[1 + 2].i = shape_int8_bordered.dims != 0 ? shape_int8_bordered.cstep : 0;
                specializations_winograd_input[1 + 3].i = shape_winograd_input_transformed.dims != 0 ? shape_winograd_input_transformed.cstep : 0;
                specializations_winograd_input[1 + 4].i = block_x;
                specializations_winograd_input[1 + 5].i = block_y;

                int shader_type_index = -1;
                if (use_cooperative_matrix)
                {
                    if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_input_int8_cm;
                    if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_input_int8_cm;
                }
                else
                {
                    if (elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_input_int8;
                    if (elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_input_int8;
                }

                pipeline_convolution_3x3s1d1_winograd23_transform_input = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_input->create(shader_type_index, opt_int8, specializations_winograd_input);
            }
            if (!use_cooperative_matrix)
            {
                // winograd23/43 share gemm shader, transform count is set by dispatcher.c
                pipeline_convolution_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                {
                    std::vector<vk_specialization_type> specializations_winograd_gemm(5 + 3);
                    specializations_winograd_gemm[0].i = 16;
                    specializations_winograd_gemm[1].i = num_input;
                    specializations_winograd_gemm[2].i = num_output;
                    specializations_winograd_gemm[3].i = elempack;
                    specializations_winograd_gemm[4].i = out_elempack;
                    specializations_winograd_gemm[5 + 0].i = 0;
                    specializations_winograd_gemm[5 + 1].i = 0;
                    specializations_winograd_gemm[5 + 2].i = 0;

                    pipeline_convolution_3x3s1d1_winograd23_gemm->set_local_size_xyz(opt_int8.use_shader_local_memory ? 8 : 4, opt_int8.use_shader_local_memory ? 8 : std::min(4, (num_output + 3) / 4), opt_int8.use_shader_local_memory ? 1 : 4);
                    pipeline_convolution_3x3s1d1_winograd23_gemm->create(LayerShaderType::convolution_3x3s1d1_winograd_gemm_int8, opt_int8, specializations_winograd_gemm);
                }
            }
            else
            {
                std::vector<vk_specialization_type> specializations_winograd_gemm(15 + 3);
                specializations_winograd_gemm[0].u32 = 16;
                specializations_winograd_gemm[1].u32 = coopmat_M;
                specializations_winograd_gemm[2].u32 = coopmat_N;
                specializations_winograd_gemm[3].u32 = coopmat_K;
                specializations_winograd_gemm[4].u32 = UNROLL_SG_M;
                specializations_winograd_gemm[5].u32 = UNROLL_SG_N;
                specializations_winograd_gemm[6].u32 = UNROLL_SG_K;
                specializations_winograd_gemm[7].u32 = UNROLL_WG_M;
                specializations_winograd_gemm[8].u32 = UNROLL_WG_N;
                specializations_winograd_gemm[9].u32 = coopmat_subgroup_size;
                specializations_winograd_gemm[10].u32 = num_input;
                specializations_winograd_gemm[11].u32 = num_output;
                specializations_winograd_gemm[12].u32 = elempack;
                specializations_winograd_gemm[13].u32 = out_elempack;
                specializations_winograd_gemm[14].u32 = weight_winograd23_data_int8_packed_cm.cstep;
                specializations_winograd_gemm[15 + 0].u32 = 0;
                specializations_winograd_gemm[15 + 1].u32 = 0;
                specializations_winograd_gemm[15 + 2].u32 = 0;

                pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm->create(LayerShaderType::convolution_3x3s1d1_winograd_gemm_int8_cm, opt_int8, specializations_winograd_gemm);
            }
            {
                int block_x = 0;
                int block_y = 0;
                Mat shape_winograd_gemm_packed;

                if (out_shape_blob.dims == 3)
                {
                    block_x = (out_shape_blob.w + 1) / 2;
                    block_y = (out_shape_blob.h + 1) / 2;
                    shape_winograd_gemm_packed = Mat(block_x * block_y, 1, num_output / out_elempack * 16, (void*)0, (size_t)4u * out_elempack, out_elempack);
                }

                std::vector<vk_specialization_type> specializations_winograd_output(6 + 6);
                specializations_winograd_output[0].i = bias_term;
                specializations_winograd_output[1].i = activation_type;
                specializations_winograd_output[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations_winograd_output[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations_winograd_output[4].i = use_int8_requantize ? 1 : 0;
                specializations_winograd_output[5].i = num_output;
                specializations_winograd_output[6 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations_winograd_output[6 + 1].i = block_x;
                specializations_winograd_output[6 + 2].i = block_y;
                specializations_winograd_output[6 + 3].i = out_shape_blob.w;
                specializations_winograd_output[6 + 4].i = out_shape_blob.h;
                specializations_winograd_output[6 + 5].i = out_shape_blob.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution_3x3s1d1_winograd23_transform_output_int8;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack4_3x3s1d1_winograd23_transform_output_int8;

                pipeline_convolution_3x3s1d1_winograd23_transform_output = new Pipeline(vkdev);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_3x3s1d1_winograd23_transform_output->create(shader_type_index, opt_int8, specializations_winograd_output);
            }
        }
    }
    else if (is_conv1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations_1x1(7 + 5 + 9);
            specializations_1x1[0].i = bias_term;
            specializations_1x1[1].i = activation_type;
            specializations_1x1[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations_1x1[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations_1x1[4].i = use_int8_requantize ? 1 : 0;
            specializations_1x1[5].u32 = elempack;
            specializations_1x1[6].u32 = out_elempack;
            specializations_1x1[7 + 0].u32 = shape_int8_bordered.dims != 0 ? shape_int8_bordered.cstep : 0;
            specializations_1x1[7 + 1].u32 = out_shape_blob.dims != 0 ? out_shape_blob.cstep : 0;
            specializations_1x1[7 + 2].u32 = out_shape.dims != 0 ? out_shape.w * out_shape.h : 0;
            specializations_1x1[7 + 3].u32 = num_output;
            specializations_1x1[7 + 4].u32 = num_input;
            specializations_1x1[12 + 0].u32 = coopmat_M;
            specializations_1x1[12 + 1].u32 = coopmat_N;
            specializations_1x1[12 + 2].u32 = coopmat_K;
            specializations_1x1[12 + 3].u32 = coopmat_subgroup_size;
            specializations_1x1[12 + 4].u32 = UNROLL_SG_M;
            specializations_1x1[12 + 5].u32 = UNROLL_SG_N;
            specializations_1x1[12 + 6].u32 = UNROLL_SG_K;
            specializations_1x1[12 + 7].u32 = UNROLL_WG_M;
            specializations_1x1[12 + 8].u32 = UNROLL_WG_N;

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_1x1s1d1->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution_1x1s1d1->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_1x1s1d1_int8_cm, opt_int8, specializations_1x1);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            const int c_packed = num_input_packed / 4;
            const int cstep_vec4 = shape_int8_bordered.dims != 0 ? (elempack == 4 ? shape_int8_bordered.cstep : shape_int8_bordered.cstep / 4) : 0;
            const int outc_pack4 = num_output_packed / 4;

            std::vector<vk_specialization_type> specializations_1x1(7 + 8);
            specializations_1x1[0].i = bias_term;
            specializations_1x1[1].i = activation_type;
            specializations_1x1[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations_1x1[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations_1x1[4].i = use_int8_requantize ? 1 : 0;
            specializations_1x1[5].i = elempack;
            specializations_1x1[6].i = out_elempack;
            specializations_1x1[7 + 0].i = c_packed;
            specializations_1x1[7 + 1].i = cstep_vec4;
            specializations_1x1[7 + 2].i = out_shape.dims != 0 ? outc_pack4 : 0;
            specializations_1x1[7 + 3].i = out_shape_blob.dims != 0 ? (out_elempack == 4 ? out_shape_blob.cstep : out_shape_blob.cstep / 4) : 0;
            specializations_1x1[7 + 4].i = out_shape_blob.dims != 0 ? out_shape_blob.cstep / 4 : 0;
            specializations_1x1[7 + 5].i = out_shape.dims != 0 ? (out_shape.w * out_shape.h + 3) / 4 : 0;
            specializations_1x1[7 + 6].i = num_output;
            specializations_1x1[7 + 7].i = num_input;

            Mat local_size_xyz(8, std::min(8, outc_pack4), 1, (void*)0);

            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            if (opt_int8.use_shader_local_memory)
            {
                pipeline_convolution_1x1s1d1->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_1x1s1d1->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_packed_1x1s1d1_int8, opt_int8, specializations_1x1);
        }
    }
    else if (use_gemm)
    {
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations_gemm(13 + 8 + 9);
            specializations_gemm[0].u32 = kernel_w;
            specializations_gemm[1].u32 = kernel_h;
            specializations_gemm[2].u32 = dilation_w;
            specializations_gemm[3].u32 = dilation_h;
            specializations_gemm[4].u32 = stride_w;
            specializations_gemm[5].u32 = stride_h;
            specializations_gemm[6].i = bias_term;
            specializations_gemm[7].i = activation_type;
            specializations_gemm[8].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations_gemm[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations_gemm[10].i = use_int8_requantize ? 1 : 0;
            specializations_gemm[11].u32 = elempack;
            specializations_gemm[12].u32 = out_elempack;
            specializations_gemm[13 + 0].u32 = shape_int8_bordered.dims != 0 ? shape_int8_bordered.w : 0;
            specializations_gemm[13 + 1].u32 = shape_int8_bordered.dims != 0 ? shape_int8_bordered.h : 0;
            specializations_gemm[13 + 2].u32 = shape_int8_bordered.dims != 0 ? shape_int8_bordered.cstep : 0;
            specializations_gemm[13 + 3].u32 = out_shape_blob.dims != 0 ? out_shape_blob.w : 0;
            specializations_gemm[13 + 4].u32 = out_shape_blob.dims != 0 ? out_shape_blob.h : 0;
            specializations_gemm[13 + 5].u32 = out_shape_blob.dims != 0 ? out_shape_blob.cstep : 0;
            specializations_gemm[13 + 6].u32 = num_output;
            specializations_gemm[13 + 7].u32 = num_input;
            specializations_gemm[21 + 0].u32 = coopmat_M;
            specializations_gemm[21 + 1].u32 = coopmat_N;
            specializations_gemm[21 + 2].u32 = coopmat_K;
            specializations_gemm[21 + 3].u32 = coopmat_subgroup_size;
            specializations_gemm[21 + 4].u32 = UNROLL_SG_M;
            specializations_gemm[21 + 5].u32 = UNROLL_SG_N;
            specializations_gemm[21 + 6].u32 = UNROLL_SG_K;
            specializations_gemm[21 + 7].u32 = UNROLL_WG_M;
            specializations_gemm[21 + 8].u32 = UNROLL_WG_N;

            pipeline_convolution_gemm = new Pipeline(vkdev);
            pipeline_convolution_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_convolution_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_convolution_gemm->create(LayerShaderType::convolution_gemm_int8_cm, opt_int8, specializations_gemm);
        }
        else
        {
            const int outc_pack4 = (num_output + 3) / 4;
            const int outsize = shape.dims == 3 ? (shape.w * shape.h + 3) / 4 : 16;
            Mat local_size_xyz(std::min(8, outsize), std::min(8, outc_pack4), 1, (void*)0);

            pipeline_convolution_gemm = new Pipeline(vkdev);
            if (opt_int8.use_shader_local_memory)
            {
                pipeline_convolution_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_convolution_gemm->set_optimal_local_size_xyz(local_size_xyz);
            }
            pipeline_convolution_gemm->create(LayerShaderType::convolution_packed_gemm_int8, opt_int8, specializations);
        }
    }
    else
    {
        const int outc_pack4 = (num_output + 3) / 4;
        Mat local_size_xyz(8, 8, std::min(4, (outc_pack4 + 1) / 2), (void*)0);

        std::vector<vk_specialization_type> specializations_direct = specializations;
        for (int i = 0; i < 8; i++)
        {
            specializations_direct[13 + i].i = 0;
        }

        pipeline_convolution = new Pipeline(vkdev);
        pipeline_convolution->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution->create(LayerShaderType::convolution_packed_int8, opt_int8, specializations_direct);
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Convolution_vulkan::upload_model_int8(VkTransfer& cmd, const Option& opt)
{
    Option opt_fp32 = opt;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    const bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    const bool use_winograd = opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16;

    if (use_winograd)
    {
        if (use_cooperative_matrix)
        {
            if (!weight_winograd43_data_int8_packed_cm.empty())
            {
                cmd.record_upload(weight_winograd43_data_int8_packed_cm, weight_data_gpu_tm_winograd43_int8_cm, opt);

                weight_winograd43_data_int8_packed_cm.release();
            }

            if (!weight_winograd23_data_int8_packed_cm.empty())
            {
                cmd.record_upload(weight_winograd23_data_int8_packed_cm, weight_data_gpu_tm_winograd23_int8_cm, opt);

                weight_winograd23_data_int8_packed_cm.release();
            }

            weight_winograd43_data_int8_packed.release();
            weight_winograd23_data_int8_packed.release();
        }
        else
        {
            weight_winograd43_data_int8_packed_cm.release();
            weight_winograd23_data_int8_packed_cm.release();

            if (!weight_winograd43_data_int8_packed.empty())
            {
                cmd.record_upload(weight_winograd43_data_int8_packed, weight_data_gpu_tm_winograd43, opt_fp32);

                weight_winograd43_data_int8_packed.release();
            }

            if (!weight_winograd23_data_int8_packed.empty())
            {
                cmd.record_upload(weight_winograd23_data_int8_packed, weight_data_gpu_tm_winograd23, opt_fp32);

                weight_winograd23_data_int8_packed.release();
            }
        }

        weight_data_int8_packed.release();
    }
    else
    {
        cmd.record_upload(weight_data_int8_packed, weight_data_gpu, opt);

        weight_data_int8_packed.release();
    }

    cmd.record_upload(weight_data_int8_descales, weight_data_int8_descales_gpu, opt_fp32);

    weight_data_int8_descales.release();

    const bool use_int8_requantize = int8_scale_term > 100;
    if (use_int8_requantize)
    {
        cmd.record_upload(top_blob_int8_scales, top_blob_int8_scales_gpu, opt);
    }

    if (bias_term)
    {
        cmd.record_upload(bias_data_int8_packed, bias_data_gpu, opt_fp32);

        bias_data_int8_packed.release();
        bias_data.release();
    }

    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    quantize->upload_model(cmd, opt);

    return 0;
}

int Convolution_vulkan::forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;
    const bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    const bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    const bool use_winograd = opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && is_conv3x3s1d1 && num_input >= 16 && num_output >= 16;
    const bool use_gemm = opt.use_sgemm_convolution && !is_conv1x1s1d1 && !use_winograd && num_input * maxk >= 8 && num_output >= 8;

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        NCNN_LOGE("Convolution int8 1d input is not supported, please replace this layer with InnerProduct");
        NCNN_LOGE("ncnn param suggestion: Convolution ... 0=%d 1=1 11=1 5=%d 6=%d 8=%d 9=%d 10=... -> InnerProduct ... 0=%d 1=%d 2=%d 8=%d 9=%d 10=...", num_output, bias_term, weight_data_size, int8_scale_term, activation_type, num_output, bias_term, weight_data_size, int8_scale_term, activation_type);
        return -1;
    }

    if (bottom_blob.dims != 3)
    {
        NCNN_LOGE("Convolution_vulkan int8 only supports 3d input for now");
        return -1;
    }

    const int elempack = opt.use_packing_layout && num_input % 4 == 0 ? 4 : 1;

    VkMat bottom = bottom_blob;
    if (bottom.elembits() != 8)
    {
        Option opt_quantize = opt;
        opt_quantize.blob_vkallocator = opt.workspace_vkallocator;
        opt_quantize.use_fp16_arithmetic = false;

        VkMat bottom_int8;
        quantize->forward(bottom, bottom_int8, cmd, opt_quantize);

        bottom = bottom_int8;
    }

    int w = bottom.w;
    int h = bottom.h;
    const int channels = bottom.c * bottom.elempack;

    if (channels != num_input)
    {
        NCNN_LOGE("Convolution_vulkan int8 input channels mismatch");
        return -1;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        VkMat bottom_bordered;
        padding->forward(bottom, bottom_bordered, cmd, opt_pad);

        bottom = bottom_bordered;
        w = bottom.w;
        h = bottom.h;
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);

            bottom = padding_outputs[0];
            w = bottom.w;
            h = bottom.h;
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);

            bottom = padding_outputs[0];
            w = bottom.w;
            h = bottom.h;
        }
    }

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const bool use_int8_requantize = int8_scale_term > 100;
    const int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize;
    if (use_int8_requantize)
    {
        out_elemsize = out_elempack;
    }
    else if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        out_elemsize = (size_t)2u * out_elempack;
    }
    else
    {
        out_elemsize = (size_t)4u * out_elempack;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    if (use_winograd)
    {
        bool pre_winograd43 = opt.use_winograd43_convolution;
        const int w_bordered = w;
        const int h_bordered = h;
        if (opt.use_winograd23_convolution)
        {
            if (vkdev->info.type() == 0 && ((w_bordered <= 18 && h_bordered <= 18) || ((w_bordered >= 23 && w_bordered <= 24) && (h_bordered >= 23 && h_bordered <= 24))))
                pre_winograd43 = false;
            if (vkdev->info.type() != 0 && (w_bordered <= 12 && h_bordered <= 12))
                pre_winograd43 = false;
        }

        const int B = pre_winograd43 ? 36 : 16;
        const int c4 = (channels + 3) / 4;
        const int block_x = pre_winograd43 ? (outw + 3) / 4 : (outw + 1) / 2;
        const int block_y = pre_winograd43 ? (outh + 3) / 4 : (outh + 1) / 2;

        VkMat bottom_tm_blob;
        VkMat bottom_tm_blob_low;
        VkMat bottom_tm_blob_high;
        {
            if (use_cooperative_matrix)
            {
                if (elempack == 4)
                {
                    bottom_tm_blob_low.create(block_x * block_y * 2, 1, c4 * B, (size_t)4u, 4, opt.workspace_vkallocator);
                    bottom_tm_blob_high = bottom_tm_blob_low;
                }
                else
                {
                    bottom_tm_blob_low.create(block_x * block_y, 1, channels * B, (size_t)1u, 1, opt.workspace_vkallocator);
                    bottom_tm_blob_high.create(block_x * block_y, 1, channels * B, (size_t)1u, 1, opt.workspace_vkallocator);
                }
                if (bottom_tm_blob_low.empty() || bottom_tm_blob_high.empty())
                    return -100;
            }
            else
            {
                if (elempack == 4)
                    bottom_tm_blob.create(block_x * block_y, 1, c4 * B, (size_t)8u, 4, opt.workspace_vkallocator);
                else
                    bottom_tm_blob.create(block_x * block_y, 1, channels * B, (size_t)2u, 1, opt.workspace_vkallocator);
                if (bottom_tm_blob.empty())
                    return -100;
            }

            std::vector<VkMat> bindings(use_cooperative_matrix && elempack == 1 ? 3 : 2);
            bindings[0] = bottom;
            if (use_cooperative_matrix)
            {
                bindings[1] = bottom_tm_blob_low;
                if (elempack == 1)
                    bindings[2] = bottom_tm_blob_high;
            }
            else
            {
                bindings[1] = bottom_tm_blob;
            }

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom.w;
            constants[1].i = bottom.h;
            constants[2].i = bottom.cstep;
            constants[3].i = use_cooperative_matrix ? bottom_tm_blob_low.cstep : bottom_tm_blob.cstep;
            constants[4].i = block_x;
            constants[5].i = block_y;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y;
            dispatcher.c = use_cooperative_matrix ? bottom_tm_blob_low.c / B : bottom_tm_blob.c / B;

            const Pipeline* pipeline = pre_winograd43 ? pipeline_convolution_3x3s1d1_winograd43_transform_input : pipeline_convolution_3x3s1d1_winograd23_transform_input;
            cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        }

        VkMat top_tm_blob;
        {
            top_tm_blob.create(block_x * block_y, 1, num_output / out_elempack * B, (size_t)4u * out_elempack, out_elempack, opt.workspace_vkallocator);
            if (top_tm_blob.empty())
                return -100;

            if (use_cooperative_matrix)
            {
                std::vector<VkMat> bindings(6);
                bindings[0] = bottom_tm_blob_low;
                bindings[1] = bottom_tm_blob_high;
                bindings[2] = bottom_tm_blob_low;
                bindings[3] = top_tm_blob;
                bindings[4] = top_tm_blob;
                bindings[5] = pre_winograd43 ? weight_data_gpu_tm_winograd43_int8_cm : weight_data_gpu_tm_winograd23_int8_cm;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = top_tm_blob.w;
                constants[1].i = bottom_tm_blob_low.cstep;
                constants[2].i = top_tm_blob.cstep;

                const int blocks_x = (top_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                VkMat dispatcher;
                dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                dispatcher.h = 1;
                dispatcher.c = B;

                const Pipeline* pipeline = pre_winograd43 ? pipeline_convolution_3x3s1d1_winograd43_gemm_int8_cm : pipeline_convolution_3x3s1d1_winograd23_gemm_int8_cm;
                cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
            }
            else
            {
                std::vector<VkMat> bindings(5);
                bindings[0] = bottom_tm_blob;
                bindings[1] = bottom_tm_blob;
                bindings[2] = top_tm_blob;
                bindings[3] = top_tm_blob;
                bindings[4] = pre_winograd43 ? weight_data_gpu_tm_winograd43 : weight_data_gpu_tm_winograd23;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = bottom_tm_blob.cstep;
                constants[1].i = top_tm_blob.w;
                constants[2].i = top_tm_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = (top_tm_blob.w + 3) / 4;
                dispatcher.h = (num_output + 3) / 4;
                dispatcher.c = B;

                const Pipeline* pipeline = pre_winograd43 ? pipeline_convolution_3x3s1d1_winograd43_gemm : pipeline_convolution_3x3s1d1_winograd23_gemm;
                cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
            }
        }

        {
            std::vector<VkMat> bindings(6);
            bindings[0] = top_tm_blob;
            bindings[1] = top_blob;
            bindings[2] = bias_data_gpu;
            bindings[3] = weight_data_int8_descales_gpu;
            bindings[4] = top_blob_int8_scales_gpu;
            bindings[5] = top_blob;

            std::vector<vk_constant_type> constants(7);
            constants[0].i = top_tm_blob.cstep;
            constants[1].i = block_x;
            constants[2].i = block_y;
            constants[3].i = top_blob.w;
            constants[4].i = top_blob.h;
            constants[5].i = top_blob.cstep;
            constants[6].i = num_output;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y;
            dispatcher.c = top_blob.c;

            const Pipeline* pipeline = pre_winograd43 ? pipeline_convolution_3x3s1d1_winograd43_transform_output : pipeline_convolution_3x3s1d1_winograd23_transform_output;
            cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        }
    }
    else if (is_conv1x1s1d1)
    {
        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings_1x1(7);
            bindings_1x1[0] = bottom;
            bindings_1x1[1] = top_blob;
            bindings_1x1[2] = top_blob;
            bindings_1x1[3] = weight_data_gpu;
            bindings_1x1[4] = bias_data_gpu;
            bindings_1x1[5] = weight_data_int8_descales_gpu;
            bindings_1x1[6] = top_blob_int8_scales_gpu;

            const int size = top_blob.w * top_blob.h;

            std::vector<vk_constant_type> constants_1x1(5);
            constants_1x1[0].u32 = bottom.cstep;
            constants_1x1[1].u32 = top_blob.cstep;
            constants_1x1[2].u32 = size;
            constants_1x1[3].u32 = num_output;
            constants_1x1[4].u32 = num_input;

            const int blocks_x = (size + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings_1x1, constants_1x1, dispatcher);
        }
        else
        {
            std::vector<VkMat> bindings_1x1(7);
            bindings_1x1[0] = bottom;
            bindings_1x1[1] = top_blob;
            bindings_1x1[2] = weight_data_gpu;
            bindings_1x1[3] = bias_data_gpu;
            bindings_1x1[4] = weight_data_int8_descales_gpu;
            bindings_1x1[5] = top_blob_int8_scales_gpu;
            bindings_1x1[6] = top_blob;

            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;
            const int outc_pack4 = num_output_packed / 4;
            const int c_packed = num_input_packed / 4;
            const int cstep_vec4 = bottom.elempack == 4 ? bottom.cstep : bottom.cstep / 4;
            const int size = (top_blob.w * top_blob.h + 3) / 4;
            const int outcstep_vec4 = out_elempack == 4 ? top_blob.cstep : top_blob.cstep / 4;
            const int outcstep_native = top_blob.cstep / 4;

            std::vector<vk_constant_type> constants_1x1(6);
            constants_1x1[0].i = c_packed;
            constants_1x1[1].i = cstep_vec4;
            constants_1x1[2].i = outc_pack4;
            constants_1x1[3].i = outcstep_vec4;
            constants_1x1[4].i = outcstep_native;
            constants_1x1[5].i = size;

            VkMat dispatcher;
            dispatcher.w = size;
            dispatcher.h = outc_pack4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings_1x1, constants_1x1, dispatcher);
        }
    }
    else if (use_gemm)
    {
        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(7);
            bindings[0] = bottom;
            bindings[1] = top_blob;
            bindings[2] = top_blob;
            bindings[3] = weight_data_gpu;
            bindings[4] = bias_data_gpu;
            bindings[5] = weight_data_int8_descales_gpu;
            bindings[6] = top_blob_int8_scales_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].u32 = bottom.w;
            constants[1].u32 = bottom.h;
            constants[2].u32 = bottom.cstep;
            constants[3].u32 = top_blob.w;
            constants[4].u32 = top_blob.h;
            constants[5].u32 = top_blob.cstep;
            constants[6].u32 = num_output;
            constants[7].u32 = num_input;

            const int size = top_blob.w * top_blob.h;
            const int blocks_x = (size + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }
        else
        {
            const int c_packed = (num_input + 3) / 4;
            const int outc_pack4 = (num_output + 3) / 4;

            std::vector<VkMat> bindings(10);
            bindings[0] = bottom;
            bindings[1] = top_blob;
            bindings[2] = bottom;
            bindings[3] = top_blob;
            bindings[4] = weight_data_gpu;
            bindings[5] = bias_data_gpu;
            bindings[6] = weight_data_int8_descales_gpu;
            bindings[7] = top_blob_int8_scales_gpu;
            bindings[8] = top_blob;
            bindings[9] = top_blob;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = bottom.w;
            constants[1].i = bottom.h;
            constants[2].i = c_packed;
            constants[3].i = bottom.cstep;
            constants[4].i = top_blob.w;
            constants[5].i = top_blob.h;
            constants[6].i = outc_pack4;
            constants[7].i = out_elempack == 4 ? top_blob.cstep : top_blob.cstep * 4;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
            dispatcher.h = outc_pack4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution_gemm, bindings, constants, dispatcher);
        }
    }
    else
    {
        const int c_packed = (num_input + 3) / 4;
        const int c_shader = bottom.elempack == 4 ? c_packed : num_input;
        const int outc_pack4 = (num_output + 3) / 4;

        std::vector<VkMat> bindings(10);
        bindings[0] = bottom;
        bindings[1] = top_blob;
        bindings[2] = bottom;
        bindings[3] = top_blob;
        bindings[4] = weight_data_gpu;
        bindings[5] = bias_data_gpu;
        bindings[6] = weight_data_int8_descales_gpu;
        bindings[7] = top_blob_int8_scales_gpu;
        bindings[8] = top_blob;
        bindings[9] = top_blob;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = bottom.w;
        constants[1].i = bottom.h;
        constants[2].i = c_shader;
        constants[3].i = bottom.cstep;
        constants[4].i = top_blob.w;
        constants[5].i = top_blob.h;
        constants[6].i = outc_pack4;
        constants[7].i = out_elempack == 4 ? top_blob.cstep : top_blob.cstep * 4;

        VkMat dispatcher;
        dispatcher.w = (top_blob.w + 1) / 2;
        dispatcher.h = (top_blob.h + 1) / 2;
        dispatcher.c = (outc_pack4 + 1) / 2;

        cmd.record_pipeline(pipeline_convolution, bindings, constants, dispatcher);
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
