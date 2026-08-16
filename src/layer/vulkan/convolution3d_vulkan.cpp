// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution3d_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"
#include "modelbin.h"

namespace ncnn {

Convolution3D_vulkan::Convolution3D_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    padding = 0;

    pipeline_convolution3d = 0;
    pipeline_convolution3d_1x1x1 = 0;
    pipeline_convolution3d_gemm = 0;

    pipeline_convolution3d_3x3x3_winograd222_transform_input = 0;
    pipeline_convolution3d_3x3x3_winograd222_gemm = 0;
    pipeline_convolution3d_3x3x3_winograd222_transform_output = 0;

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

int Convolution3D_vulkan::load_param(const ParamDict& pd)
{
    return Convolution3D::load_param(pd);
}

int Convolution3D_vulkan::create_pipeline(const Option& opt)
{
    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // skip fc like hint
    if (shape.dims != 4) shape = Mat();
    if (out_shape.dims != 4) out_shape = Mat();

    const int maxk = kernel_w * kernel_h * kernel_d;
    int num_input = weight_data_size / maxk / num_output;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.d + pad_front + pad_behind, shape.c, (void*)0, shape.elemsize, shape.elempack);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233 && pad_front == -233 && pad_behind == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234 && pad_front == -234 && pad_behind == -234))
        {
            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            int dpad = kernel_extent_d + (shape.d - 1) / stride_d * stride_d - shape.d;
            if (wpad > 0 || hpad > 0 || dpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h + hpad, shape.d + dpad, shape.c, (void*)0, shape.elemsize, shape.elempack);
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

    bool is_conv1x1x1s1d1 = kernel_w == 1 && kernel_h == 1 && kernel_d == 1 && stride_w == 1 && stride_h == 1 && stride_d == 1 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1;
    bool is_conv3x3x3s1d1 = kernel_w == 3 && kernel_h == 3 && kernel_d == 3 && stride_w == 1 && stride_h == 1 && stride_d == 1 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1;

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
        pd.set(7, pad_front);
        pd.set(8, pad_behind);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    if (opt.use_winograd_convolution && opt.use_winograd23_convolution && is_conv3x3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            // f222 shares the same size parameter as conv2d winograd, set zero for dynamic dispatch
            // if (out_shape.dims != 0)
            // {
            //     int block_x = (out_shape.w + 1) / 2;
            //     int block_y = (out_shape.h + 1) / 2;
            //     int block_z = (out_shape.d + 1) / 2;
            //     size = block_x * block_y * block_z;
            // }

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, num_output, num_input, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((num_output + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((num_input + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((num_output + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);
        }
        // winograd222 transform kernel
        {
            Mat weight_data_tm;
            weight_data_tm.create(4 * 4 * 4, num_input, num_output);

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
                    const float* kernel0 = (const float*)weight_data + p * num_input * 27 + q * 27;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // U = G g G^T G^T, separable along x/y/z
                    float U[4][4][4];
                    for (int x = 0; x < 4; x++)
                    {
                        for (int y = 0; y < 4; y++)
                        {
                            for (int z = 0; z < 4; z++)
                            {
                                float s = 0.f;
                                for (int i = 0; i < 3; i++)
                                {
                                    for (int j = 0; j < 3; j++)
                                    {
                                        for (int k = 0; k < 3; k++)
                                        {
                                            s += ktm[x][i] * ktm[y][j] * ktm[z][k] * kernel0[(k * 3 + j) * 3 + i];
                                        }
                                    }
                                }
                                U[x][y][z] = s;
                            }
                        }
                    }

                    // store 64, index = (x*4+y)*4+z
                    for (int x = 0; x < 4; x++)
                    {
                        for (int y = 0; y < 4; y++)
                        {
                            for (int z = 0; z < 4; z++)
                            {
                                kernel_tm0[(x * 4 + y) * 4 + z] = U[x][y][z];
                            }
                        }
                    }
                }
            }

            if (use_cooperative_matrix)
            {
                // from 64-inch-outch to inch-outch-64
                Mat weight_data_tm_r2(num_input, num_output, 64);
                for (int k = 0; k < 64; k++)
                {
                    float* g00 = weight_data_tm_r2.channel(k);

                    for (int q = 0; q < num_output; q++)
                    {
                        for (int p = 0; p < num_input; p++)
                        {
                            *g00++ = weight_data_tm[(q * num_input + p) * 64 + k];
                        }
                    }
                }

                const int blocks_n = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                const int kk = (num_input + coopmat_K - 1) / coopmat_K;

                weight_winograd222_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n, 64);
                for (int b = 0; b < 64; b++)
                {
                    for (int bn = 0; bn < blocks_n; bn++)
                    {
                        float* p = weight_winograd222_data_packed.channel(b).row(bn);

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
            else
            {
                // src = 64-inch-outch
                // dst = 8a-8b-inch/8a-outch/8b-64
                weight_winograd222_data_packed.create(num_input / elempack, num_output / out_elempack, 64, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

                for (int k = 0; k < 64; k++)
                {
                    float* g00 = weight_winograd222_data_packed.channel(k);

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
        // winograd222
        {
            int block_x = 0;
            int block_y = 0;
            int block_z = 0;
            Mat shape_winograd_input_transformed;
            Mat shape_winograd_gemm;
            Mat shape_winograd_input_transformed_packed;
            Mat shape_winograd_gemm_packed;

            if (out_shape.dims != 0)
            {
                block_x = (out_shape.w + 1) / 2;
                block_y = (out_shape.h + 1) / 2;
                block_z = (out_shape.d + 1) / 2;

                shape_winograd_input_transformed = Mat(block_x * block_y * block_z, 1, shape.c * 64, (void*)0);
                shape_winograd_gemm = Mat(block_x * block_y * block_z, 1, out_shape.c * 64, (void*)0);
            }

            if (shape_winograd_input_transformed.dims == 3) shape_winograd_input_transformed_packed = Mat(shape_winograd_input_transformed.w, 1, shape_winograd_input_transformed.h / elempack * 64, (void*)0, elemsize, elempack);

            if (shape_winograd_gemm.dims == 3) shape_winograd_gemm_packed = Mat(shape_winograd_gemm.w, 1, shape_winograd_gemm.h / out_elempack * 64, (void*)0, out_elemsize, out_elempack);

            {
                std::vector<vk_specialization_type> specializations(1 + 8);
                specializations[0].i = num_input / elempack;
                specializations[1 + 0].i = shape_bordered.w;
                specializations[1 + 1].i = shape_bordered.h;
                specializations[1 + 2].i = shape_bordered.d;
                specializations[1 + 3].i = shape_bordered.cstep;
                specializations[1 + 4].i = shape_winograd_input_transformed_packed.cstep;
                specializations[1 + 5].i = block_x;
                specializations[1 + 6].i = block_y;
                specializations[1 + 7].i = block_z;

                int shader_type_index = -1;
                if (elempack == 1) shader_type_index = LayerShaderType::convolution3d_3x3x3_winograd222_transform_input;
                if (elempack == 4) shader_type_index = LayerShaderType::convolution3d_pack4_3x3x3_winograd222_transform_input;

                pipeline_convolution3d_3x3x3_winograd222_transform_input = new Pipeline(vkdev);
                pipeline_convolution3d_3x3x3_winograd222_transform_input->set_local_size_xyz(8, 8, 1);
                pipeline_convolution3d_3x3x3_winograd222_transform_input->create(shader_type_index, opt, specializations);
            }

            if (use_cooperative_matrix)
            {
                Mat weight_winograd222_data_packed_fp16 = Mat(weight_winograd222_data_packed.w, weight_winograd222_data_packed.h, weight_winograd222_data_packed.c, (void*)0, 2u, 1);

                std::vector<vk_specialization_type> specializations(15 + 3);
                specializations[0].u32 = 64; //batch
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
                specializations[14].u32 = weight_winograd222_data_packed_fp16.cstep;
                specializations[15 + 0].u32 = shape_winograd_input_transformed_packed.w;
                specializations[15 + 1].u32 = shape_winograd_input_transformed_packed.cstep;
                specializations[15 + 2].u32 = shape_winograd_gemm_packed.cstep;

                pipeline_convolution3d_3x3x3_winograd222_gemm = new Pipeline(vkdev);
                pipeline_convolution3d_3x3x3_winograd222_gemm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution3d_3x3x3_winograd222_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution3d_3x3x3_winograd222_gemm->create(LayerShaderType::convolution3d_winograd_gemm_cm, opt, specializations);
            }
            else
            {
                std::vector<vk_specialization_type> specializations(3 + 3);
                specializations[0].i = 64;
                specializations[1].i = num_input / elempack;
                specializations[2].i = num_output / out_elempack;
                specializations[3 + 0].i = shape_winograd_input_transformed_packed.cstep;
                specializations[3 + 1].i = shape_winograd_gemm_packed.w;
                specializations[3 + 2].i = shape_winograd_gemm_packed.cstep;

                int shader_type_index = -1;
                if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution3d_3x3x3_winograd222_gemm;
                if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::convolution3d_pack4_3x3x3_winograd222_gemm;
                if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution3d_pack1to4_3x3x3_winograd222_gemm;
                if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::convolution3d_pack4to1_3x3x3_winograd222_gemm;

                pipeline_convolution3d_3x3x3_winograd222_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution3d_3x3x3_winograd222_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution3d_3x3x3_winograd222_gemm->set_local_size_xyz(4, std::min(4, num_output / out_elempack), 4);
                }
                pipeline_convolution3d_3x3x3_winograd222_gemm->create(shader_type_index, opt, specializations);
            }

            {
                std::vector<vk_specialization_type> specializations(5 + 8);
                specializations[0].i = bias_term;
                specializations[1].i = activation_type;
                specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[4].i = num_output / out_elempack;
                specializations[5 + 0].i = shape_winograd_gemm_packed.cstep;
                specializations[5 + 1].i = block_x;
                specializations[5 + 2].i = block_y;
                specializations[5 + 3].i = block_z;
                specializations[5 + 4].i = out_shape.w;
                specializations[5 + 5].i = out_shape.h;
                specializations[5 + 6].i = out_shape.d;
                specializations[5 + 7].i = out_shape.cstep;

                int shader_type_index = -1;
                if (out_elempack == 1) shader_type_index = LayerShaderType::convolution3d_3x3x3_winograd222_transform_output;
                if (out_elempack == 4) shader_type_index = LayerShaderType::convolution3d_pack4_3x3x3_winograd222_transform_output;

                pipeline_convolution3d_3x3x3_winograd222_transform_output = new Pipeline(vkdev);
                pipeline_convolution3d_3x3x3_winograd222_transform_output->set_local_size_xyz(8, 8, 1);
                pipeline_convolution3d_3x3x3_winograd222_transform_output->create(shader_type_index, opt, specializations);
            }
        }
    }
    else if (opt.use_sgemm_convolution && !is_conv1x1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (out_shape.dims == 4)
                size = out_shape.w * out_shape.h * out_shape.d;

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
        {
            if (use_cooperative_matrix)
            {
                std::vector<vk_specialization_type> specializations(26 + 8);
                specializations[0].u32 = kernel_w;
                specializations[1].u32 = kernel_h;
                specializations[2].u32 = kernel_d;
                specializations[3].u32 = dilation_w;
                specializations[4].u32 = dilation_h;
                specializations[5].u32 = dilation_d;
                specializations[6].u32 = stride_w;
                specializations[7].u32 = stride_h;
                specializations[8].u32 = stride_d;
                specializations[9].i = bias_term;
                specializations[10].i = activation_type;
                specializations[11].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[12].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[13].u32 = coopmat_M;
                specializations[14].u32 = coopmat_N;
                specializations[15].u32 = coopmat_K;
                specializations[16].u32 = coopmat_subgroup_size;
                specializations[17].u32 = UNROLL_SG_M;
                specializations[18].u32 = UNROLL_SG_N;
                specializations[19].u32 = UNROLL_SG_K;
                specializations[20].u32 = UNROLL_WG_M;
                specializations[21].u32 = UNROLL_WG_N;
                specializations[22].u32 = num_input;
                specializations[23].u32 = num_output;
                specializations[24].u32 = elempack;
                specializations[25].u32 = out_elempack;
                specializations[26 + 0].u32 = shape_bordered.w;
                specializations[26 + 1].u32 = shape_bordered.h;
                specializations[26 + 2].u32 = shape_bordered.d;
                specializations[26 + 3].u32 = shape_bordered.cstep;
                specializations[26 + 4].u32 = out_shape.w;
                specializations[26 + 5].u32 = out_shape.h;
                specializations[26 + 6].u32 = out_shape.d;
                specializations[26 + 7].u32 = out_shape.cstep;

                pipeline_convolution3d_gemm = new Pipeline(vkdev);
                pipeline_convolution3d_gemm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution3d_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution3d_gemm->create(LayerShaderType::convolution3d_gemm_cm, opt, specializations);
            }
            else
            {
                const int num_input_packed = (num_input + 3) / 4 * 4;
                const int num_output_packed = (num_output + 3) / 4 * 4;

                std::vector<vk_specialization_type> specializations(15 + 12);
                specializations[0].i = kernel_w;
                specializations[1].i = kernel_h;
                specializations[2].i = kernel_d;
                specializations[3].i = dilation_w;
                specializations[4].i = dilation_h;
                specializations[5].i = dilation_d;
                specializations[6].i = stride_w;
                specializations[7].i = stride_h;
                specializations[8].i = stride_d;
                specializations[9].i = bias_term;
                specializations[10].i = activation_type;
                specializations[11].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
                specializations[12].f = activation_params.w == 2 ? activation_params[1] : 0.f;
                specializations[13].i = elempack;
                specializations[14].i = out_elempack;
                specializations[15 + 0].i = shape_bordered.w;
                specializations[15 + 1].i = shape_bordered.h;
                specializations[15 + 2].i = shape_bordered.d;
                specializations[15 + 3].i = num_input_packed / 4;
                specializations[15 + 4].i = shape_bordered.cstep;
                specializations[15 + 5].i = out_shape.w;
                specializations[15 + 6].i = out_shape.h;
                specializations[15 + 7].i = out_shape.d;
                specializations[15 + 8].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
                specializations[15 + 9].i = out_shape.dims != 0 ? out_shape.cstep : 0;
                specializations[15 + 10].i = num_output;
                specializations[15 + 11].i = num_input;

                Mat local_size_xyz(16, std::min(4, num_output_packed / 4), 1, (void*)0);
                if (out_shape.dims != 0)
                {
                    local_size_xyz.w = std::min(16, out_shape.w * out_shape.h * out_shape.d);
                    local_size_xyz.h = std::min(4, num_output_packed / 4);
                }

                pipeline_convolution3d_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution3d_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution3d_gemm->set_optimal_local_size_xyz(local_size_xyz);
                }
                pipeline_convolution3d_gemm->create(LayerShaderType::convolution3d_packed_gemm, opt, specializations);
            }
        }
    }
    else if (is_conv1x1x1s1d1)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

        if (use_cooperative_matrix)
        {
            int size = 1024;
            if (out_shape.dims == 4)
                size = out_shape.w * out_shape.h * out_shape.d;

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
        }
        else
        {
            Mat weight_data_r2 = weight_data.reshape(1, num_input, num_output);

            // unified pack4x4 weight layout: both input and output always packed by 4
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;

            weight_data_packed.create(1, num_input_packed / 4, num_output_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < num_output_packed; q += 4)
            {
                float* g00 = weight_data_packed.channel(q / 4);

                for (int p = 0; p < num_input_packed; p += 4)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (q + i < num_output && p + j < num_input)
                            {
                                const float* k00 = weight_data_r2.channel(q + i).row(p + j);
                                g00[0] = k00[0];
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
                specializations[17 + 0].u32 = out_shape.w * out_shape.h * out_shape.d;
                specializations[17 + 1].u32 = shape_bordered.cstep;
                specializations[17 + 2].u32 = out_shape.cstep;

                pipeline_convolution3d_1x1x1 = new Pipeline(vkdev);
                pipeline_convolution3d_1x1x1->set_subgroup_size(coopmat_subgroup_size);
                pipeline_convolution3d_1x1x1->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_convolution3d_1x1x1->create(LayerShaderType::convolution3d_1x1x1_cm, opt, specializations);
            }
            else
            {
                const int num_input_packed = (num_input + 3) / 4 * 4;
                const int num_output_packed = (num_output + 3) / 4 * 4;

                // c = loop iterations = num_input_packed/4 for all elempacks
                // cstep = vec4 stride between channels
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
                specializations[6 + 5].i = out_shape.dims != 0 ? (out_shape.w * out_shape.h * out_shape.d + 3) / 4 : 0;
                specializations[6 + 6].i = num_output;
                specializations[6 + 7].i = num_input;

                const int outc_pack4 = num_output_packed / 4;

                pipeline_convolution3d_1x1x1 = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_convolution3d_1x1x1->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_convolution3d_1x1x1->set_local_size_xyz(8, std::min(8, outc_pack4), 1);
                }
                pipeline_convolution3d_1x1x1->create(LayerShaderType::convolution3d_packed_1x1x1, opt, specializations);
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

        {
            const int num_output_packed = (num_output + 3) / 4 * 4;

            std::vector<vk_specialization_type> specializations(15 + 13);
            specializations[0].i = kernel_w;
            specializations[1].i = kernel_h;
            specializations[2].i = kernel_d;
            specializations[3].i = dilation_w;
            specializations[4].i = dilation_h;
            specializations[5].i = dilation_d;
            specializations[6].i = stride_w;
            specializations[7].i = stride_h;
            specializations[8].i = stride_d;
            specializations[9].i = bias_term;
            specializations[10].i = activation_type;
            specializations[11].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
            specializations[12].f = activation_params.w == 2 ? activation_params[1] : 0.f;
            specializations[13].i = elempack;
            specializations[14].i = out_elempack;
            specializations[15 + 0].i = shape_bordered.dims;
            specializations[15 + 1].i = shape_bordered.w;
            specializations[15 + 2].i = shape_bordered.h;
            specializations[15 + 3].i = shape_bordered.d;
            specializations[15 + 4].i = shape_bordered.c;
            specializations[15 + 5].i = shape_bordered.cstep;
            specializations[15 + 6].i = out_shape.dims;
            specializations[15 + 7].i = out_shape.w;
            specializations[15 + 8].i = out_shape.h;
            specializations[15 + 9].i = out_shape.d;
            specializations[15 + 10].i = out_shape.dims != 0 ? num_output_packed / 4 : 0;
            specializations[15 + 11].i = out_shape.dims != 0 ? (out_elempack == 4 ? out_shape.cstep : out_shape.cstep * 4) : 0;
            specializations[15 + 12].i = num_output;

            const int outc_pack4 = num_output_packed / 4;

            Mat local_size_xyz(8, 8, std::min(4, (outc_pack4 + 1) / 2), (void*)0);
            if (out_shape.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape.w);
                local_size_xyz.h = std::min(8, out_shape.h * out_shape.d);
                local_size_xyz.c = std::min(4, (outc_pack4 + 1) / 2);
            }

            pipeline_convolution3d = new Pipeline(vkdev);
            pipeline_convolution3d->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolution3d->create(LayerShaderType::convolution3d_packed, opt, specializations);
        }
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int Convolution3D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution3d;
    pipeline_convolution3d = 0;

    delete pipeline_convolution3d_1x1x1;
    pipeline_convolution3d_1x1x1 = 0;

    delete pipeline_convolution3d_gemm;
    pipeline_convolution3d_gemm = 0;

    delete pipeline_convolution3d_3x3x3_winograd222_transform_input;
    pipeline_convolution3d_3x3x3_winograd222_transform_input = 0;

    delete pipeline_convolution3d_3x3x3_winograd222_gemm;
    pipeline_convolution3d_3x3x3_winograd222_gemm = 0;

    delete pipeline_convolution3d_3x3x3_winograd222_transform_output;
    pipeline_convolution3d_3x3x3_winograd222_transform_output = 0;

    return 0;
}

int Convolution3D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    const int maxk = kernel_w * kernel_h * kernel_d;
    int num_input = weight_data_size / maxk / num_output;

    bool is_conv3x3x3s1d1 = kernel_w == 3 && kernel_h == 3 && kernel_d == 3 && stride_w == 1 && stride_h == 1 && stride_d == 1 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1;

    if (opt.use_winograd_convolution && opt.use_winograd23_convolution && is_conv3x3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        cmd.record_upload(weight_winograd222_data_packed, weight_data_gpu_tm_winograd222, opt);

        weight_winograd222_data_packed.release();
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
int Convolution3D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233 && pad_front == -233 && pad_behind == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = dpad / 2;
            padding_params[5] = dpad - dpad / 2;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234 && pad_front == -234 && pad_behind == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = dpad / 2;
            padding_params[5] = dpad - dpad / 2;

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
    d = bottom_blob_bordered.d;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int outd = (d - kernel_extent_d) / stride_d + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int maxk = kernel_w * kernel_h * kernel_d;
    const int num_input = channels * elempack;

    bool is_conv1x1x1s1d1 = kernel_w == 1 && kernel_h == 1 && kernel_d == 1 && stride_w == 1 && stride_h == 1 && stride_d == 1 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1;
    bool is_conv3x3x3s1d1 = kernel_w == 3 && kernel_h == 3 && kernel_d == 3 && stride_w == 1 && stride_h == 1 && stride_d == 1 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1;

    if (opt.use_winograd_convolution && opt.use_winograd23_convolution && is_conv3x3x3s1d1 && num_input >= 16 && num_output >= 16)
    {
        // winograd222
        int block_x = (outw + 1) / 2;
        int block_y = (outh + 1) / 2;
        int block_z = (outd + 1) / 2;

        // transform input
        VkMat bottom_tm_blob;
        {
            bottom_tm_blob.create(block_x * block_y * block_z, 1, channels * 64, elemsize, elempack, opt.workspace_vkallocator);
            if (bottom_tm_blob.empty())
                return -100;

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = bottom_tm_blob;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = bottom_blob_bordered.d;
            constants[3].i = bottom_blob_bordered.cstep;
            constants[4].i = bottom_tm_blob.cstep;
            constants[5].i = block_x;
            constants[6].i = block_y;
            constants[7].i = block_z;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y * block_z;
            dispatcher.c = channels;

            cmd.record_pipeline(pipeline_convolution3d_3x3x3_winograd222_transform_input, bindings, constants, dispatcher);
        }

        // gemm
        VkMat top_tm_blob;
        {
            top_tm_blob.create(block_x * block_y * block_z, 1, num_output / out_elempack * 64, out_elemsize, out_elempack, opt.workspace_vkallocator);
            if (top_tm_blob.empty())
                return -100;

            if (use_cooperative_matrix)
            {
                std::vector<VkMat> bindings(3);
                bindings[0] = bottom_tm_blob;
                bindings[1] = top_tm_blob;
                bindings[2] = weight_data_gpu_tm_winograd222;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = bottom_tm_blob.w;
                constants[1].i = bottom_tm_blob.cstep;
                constants[2].i = top_tm_blob.cstep;

                const int blocks_x = (bottom_tm_blob.w + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
                const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

                VkMat dispatcher;
                dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
                dispatcher.h = 1;
                dispatcher.c = 64;

                cmd.record_pipeline(pipeline_convolution3d_3x3x3_winograd222_gemm, bindings, constants, dispatcher);
            }
            else
            {
                std::vector<VkMat> bindings(3);
                bindings[0] = bottom_tm_blob;
                bindings[1] = top_tm_blob;
                bindings[2] = weight_data_gpu_tm_winograd222;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = bottom_tm_blob.cstep;
                constants[1].i = top_tm_blob.w;
                constants[2].i = top_tm_blob.cstep;

                VkMat dispatcher;
                dispatcher.w = (top_tm_blob.w + 3) / 4;
                dispatcher.h = num_output / out_elempack;
                dispatcher.c = 64;

                cmd.record_pipeline(pipeline_convolution3d_3x3x3_winograd222_gemm, bindings, constants, dispatcher);
            }
        }

        // transform output
        {
            top_blob.create(outw, outh, outd, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = top_tm_blob;
            bindings[1] = top_blob;
            bindings[2] = bias_data_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].i = top_tm_blob.cstep;
            constants[1].i = block_x;
            constants[2].i = block_y;
            constants[3].i = block_z;
            constants[4].i = top_blob.w;
            constants[5].i = top_blob.h;
            constants[6].i = top_blob.d;
            constants[7].i = top_blob.cstep;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y * block_z;
            dispatcher.c = top_blob.c;

            cmd.record_pipeline(pipeline_convolution3d_3x3x3_winograd222_transform_output, bindings, constants, dispatcher);
        }

        return 0;
    }
    if (opt.use_sgemm_convolution && !is_conv1x1x1s1d1 && num_input * maxk >= 8 && num_output >= 8)
    {
        // gemm
        top_blob.create(outw, outh, outd, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = top_blob;
            bindings[2] = weight_data_gpu;
            bindings[3] = bias_data_gpu;

            std::vector<vk_constant_type> constants(8);
            constants[0].u32 = bottom_blob_bordered.w;
            constants[1].u32 = bottom_blob_bordered.h;
            constants[2].u32 = bottom_blob_bordered.d;
            constants[3].u32 = bottom_blob_bordered.cstep;
            constants[4].u32 = top_blob.w;
            constants[5].u32 = top_blob.h;
            constants[6].u32 = top_blob.d;
            constants[7].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h * top_blob.d + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution3d_gemm, bindings, constants, dispatcher);
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

            std::vector<vk_constant_type> constants(12);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = bottom_blob_bordered.d;
            constants[3].i = num_input_packed / 4;
            constants[4].i = bottom_blob_bordered.cstep;
            constants[5].i = top_blob.w;
            constants[6].i = top_blob.h;
            constants[7].i = top_blob.d;
            constants[8].i = num_output_packed / 4;
            constants[9].i = top_blob.cstep;
            constants[10].i = num_output;
            constants[11].i = num_input;

            VkMat dispatcher;
            dispatcher.w = (top_blob.w * top_blob.h * top_blob.d + 3) / 4;
            dispatcher.h = num_output_packed / 4;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution3d_gemm, bindings, constants, dispatcher);
        }

        return 0;
    }
    else if (is_conv1x1x1s1d1)
    {
        top_blob.create(outw, outh, outd, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
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
            constants[0].u32 = bottom_blob_bordered.w * bottom_blob_bordered.h * bottom_blob_bordered.d;
            constants[1].u32 = bottom_blob_bordered.cstep;
            constants[2].u32 = top_blob.cstep;

            const int blocks_x = (top_blob.w * top_blob.h * top_blob.d + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (num_output + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_convolution3d_1x1x1, bindings, constants, dispatcher);
        }
        else
        {
            const int num_input_packed = (num_input + 3) / 4 * 4;
            const int num_output_packed = (num_output + 3) / 4 * 4;
            const int outc_pack4 = num_output_packed / 4;
            const int c_packed = num_input_packed / 4;
            const int cstep_vec4 = (elempack == 4) ? bottom_blob_bordered.cstep : (bottom_blob_bordered.cstep / 4);
            const int size = (top_blob.w * top_blob.h * top_blob.d + 3) / 4;

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

            cmd.record_pipeline(pipeline_convolution3d_1x1x1, bindings, constants, dispatcher);
        }

        return 0;
    }

    top_blob.create(outw, outh, outd, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
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

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.d;
    constants[4].i = bottom_blob_bordered.c;
    constants[5].i = bottom_blob_bordered.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = outc_pack4;
    constants[11].i = outcstep_pack4;
    constants[12].i = num_output;

    VkMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = ((top_blob.h + 1) / 2) * ((top_blob.d + 1) / 2);
    dispatcher.c = outc_pack4;

    cmd.record_pipeline(pipeline_convolution3d, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn



