// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "deconvolution_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Deconvolution_vulkan::Deconvolution_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    crop = 0;
    output_crop = 0;

    pipeline_deconvolution = 0;

    pipeline_deconvolution_gemm = 0;
    pipeline_deconvolution_col2im = 0;
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

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
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

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_bordered_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    // check weight shape
    Mat weight_data_packed(maxk, num_input / elempack, num_output / out_elempack, (void*)0, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
    if (!vkdev->shape_support_image_storage(weight_data_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    {
        crop = ncnn::create_layer(ncnn::LayerType::Crop);
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
        output_crop = ncnn::create_layer(ncnn::LayerType::Crop);
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

    if (opt.use_sgemm_convolution)
    {
        bool use_cooperative_matrix = vkdev->info.support_cooperative_matrix_16_8_8() && opt.use_cooperative_matrix && !opt.use_image_storage && !opt.use_shader_pack8 && opt.use_fp16_storage && num_input % 8 == 0 && num_output % 8 == 0;

        Mat out_shape_col;
        if (shape.dims != 0 && out_shape.dims != 0)
        {
            out_shape_col = Mat(shape.w * shape.h, maxk * out_shape.c, (void*)0);
        }

        Mat out_shape_col_packed;
        if (out_shape_col.dims == 2) out_shape_col_packed = Mat(out_shape_col.w, out_shape_col.h / out_elempack, (void*)0, out_elemsize, out_elempack);

        // check blob shape
        if (!vkdev->shape_support_image_storage(out_shape_col_packed))
        {
            support_image_storage = false;
            opt.use_image_storage = false;
        }

        {
            std::vector<vk_specialization_type> specializations(1 + 6);
            specializations[0].i = maxk;
            specializations[1 + 0].i = shape_packed.w;
            specializations[1 + 1].i = shape_packed.h;
            specializations[1 + 2].i = shape_packed.c;
            specializations[1 + 3].i = shape_packed.cstep;
            specializations[1 + 4].i = out_shape_col_packed.w;
            specializations[1 + 5].i = out_shape_col_packed.h;

            Mat local_size_xyz(8, std::min(4, num_output / out_elempack), 1, (void*)0);
            if (out_shape_col_packed.dims != 0)
            {
                local_size_xyz.w = std::min(8, out_shape_col_packed.w);
                local_size_xyz.h = std::min(4, out_shape_col_packed.h);
            }

            int shader_type_index = -1;
            if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_gemm;
            if (elempack == 4 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack4_gemm;
            if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack1to4_gemm;
            if (elempack == 4 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_pack4to1_gemm;
            if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack8_gemm;
            if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack1to8_gemm;
            if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_pack8to1_gemm;
            if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack4to8_gemm;
            if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack8to4_gemm;

            if (use_cooperative_matrix)
            {
                shader_type_index = LayerShaderType::deconvolution_pack4_gemm_cm_16_8_8;
            }

            pipeline_deconvolution_gemm = new Pipeline(vkdev);
            if (use_cooperative_matrix)
            {
                // TODO proper unroll y
                pipeline_deconvolution_gemm->set_local_size_xyz(32, 4, 1); // 16_8_8 ly*4
            }
            else if (opt.use_shader_local_memory)
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
            std::vector<vk_specialization_type> specializations(10 + 6);
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
            specializations[10 + 0].i = shape_packed.w;
            specializations[10 + 1].i = shape_packed.h;
            specializations[10 + 2].i = out_shape_bordered_packed.w;
            specializations[10 + 3].i = out_shape_bordered_packed.h;
            specializations[10 + 4].i = out_shape_bordered_packed.c;
            specializations[10 + 5].i = out_shape_bordered_packed.cstep;

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
            if (out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack8_col2im;

            pipeline_deconvolution_col2im = new Pipeline(vkdev);
            pipeline_deconvolution_col2im->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_deconvolution_col2im->create(shader_type_index, opt, specializations);
        }

        return 0;
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
    if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack8;
    if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack1to8;
    if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::deconvolution_pack8to1;
    if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::deconvolution_pack4to8;
    if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::deconvolution_pack8to4;

    pipeline_deconvolution = new Pipeline(vkdev);
    pipeline_deconvolution->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_deconvolution->create(shader_type_index, opt, specializations);

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

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    Mat weight_data_transposed(weight_data.w);
    if (opt.use_sgemm_convolution)
    {
        weight_data_transposed = weight_data;
    }
    else
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
    // dst = pa-pb-inch/pa-kw-kh-outch/pb (sgemm)
    Mat weight_data_packed;
    if (opt.use_sgemm_convolution)
    {
        bool use_cooperative_matrix = vkdev->info.support_cooperative_matrix_16_8_8() && opt.use_cooperative_matrix && !opt.use_image_storage && !opt.use_shader_pack8 && opt.use_fp16_storage && num_input % 8 == 0 && num_output % 8 == 0;
        if (use_cooperative_matrix)
        {
            // dst = 8a-8b-inch/8a-maxk-outch/8b
            // dst = 16a-16b-inch/16a-maxk-outch/16b
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

            weight_data_packed.create(num_input / 8, maxk * num_output / 8, (size_t)4 * 8 * 8, 8 * 8);

            for (int q = 0; q + 7 < num_output; q += 8)
            {
                for (int k = 0; k < maxk; k++)
                {
                    float* g00 = weight_data_packed.row(q / 8 * maxk + k);

                    for (int p = 0; p + 7 < num_input; p += 8)
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            for (int j = 0; j < 8; j++)
                            {
                                const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                                g00[0] = k00[k];

                                g00++;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

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
        }
    }
    else
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

    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

    if (bias_term)
    {
        Mat bias_data_packed;
        convert_packing(bias_data, bias_data_packed, out_elempack, opt);

        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }
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
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    VkMat top_blob_bordered;
    if (opt.use_sgemm_convolution)
    {
        bool use_cooperative_matrix = vkdev->info.support_cooperative_matrix_16_8_8() && opt.use_cooperative_matrix && !opt.use_image_storage && !opt.use_shader_pack8 && opt.use_fp16_storage && channels * elempack % 8 == 0 && num_output % 8 == 0;

        const int maxk = kernel_w * kernel_h;

        // gemm
        VkMat top_blob_col;
        {
            top_blob_col.create(w * h, maxk * num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
            if (top_blob_col.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_col;
            bindings[2] = weight_data_gpu;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob.w;
            constants[1].i = bottom_blob.h;
            constants[2].i = bottom_blob.c;
            constants[3].i = bottom_blob.cstep;
            constants[4].i = top_blob_col.w;
            constants[5].i = top_blob_col.h;

            VkMat dispatcher;
            dispatcher.w = (top_blob_col.w + 3) / 4;
            dispatcher.h = top_blob_col.h;
            dispatcher.c = 1;

            if (use_cooperative_matrix)
            {
                dispatcher.w = ((top_blob_col.w + 15) / 16 + 3) / 4 * 32;
                dispatcher.h = (top_blob_col.h + 1) / 2;
                dispatcher.c = 1;
            }

            cmd.record_pipeline(pipeline_deconvolution_gemm, bindings, constants, dispatcher);
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
            constants[2].i = top_blob_bordered.w;
            constants[3].i = top_blob_bordered.h;
            constants[4].i = top_blob_bordered.c;
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

int Deconvolution_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    VkImageMat top_blob_bordered;
    if (opt.use_sgemm_convolution)
    {
        const int maxk = kernel_w * kernel_h;

        // gemm
        VkImageMat top_blob_col;
        {
            top_blob_col.create(w * h, maxk * num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_vkallocator);
            if (top_blob_col.empty())
                return -100;

            std::vector<VkImageMat> bindings(3);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_col;
            bindings[2] = weight_data_gpu_image;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_blob.w;
            constants[1].i = bottom_blob.h;
            constants[2].i = bottom_blob.c;
            constants[3].i = 0; // bottom_blob.cstep;
            constants[4].i = top_blob_col.w;
            constants[5].i = top_blob_col.h;

            VkImageMat dispatcher;
            dispatcher.w = (top_blob_col.w + 3) / 4;
            dispatcher.h = top_blob_col.h;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_deconvolution_gemm, bindings, constants, dispatcher);
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

            std::vector<VkImageMat> bindings(3);
            bindings[0] = top_blob_col;
            bindings[1] = top_blob_bordered;
            bindings[2] = bias_data_gpu_image;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = w;
            constants[1].i = h;
            constants[2].i = top_blob_bordered.w;
            constants[3].i = top_blob_bordered.h;
            constants[4].i = top_blob_bordered.c;
            constants[5].i = 0; //top_blob_bordered.cstep;

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

        std::vector<VkImageMat> bindings(4);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob_bordered;
        bindings[2] = weight_data_gpu_image;
        bindings[3] = bias_data_gpu_image;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = 0; //bottom_blob.cstep;
        constants[5].i = top_blob_bordered.dims;
        constants[6].i = top_blob_bordered.w;
        constants[7].i = top_blob_bordered.h;
        constants[8].i = top_blob_bordered.c;
        constants[9].i = 0; //top_blob_bordered.cstep;

        cmd.record_pipeline(pipeline_deconvolution, bindings, constants, top_blob_bordered);
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        {
            VkImageMat reference_blob;
            reference_blob.dims = 2;
            reference_blob.w = top_blob_bordered.w - pad_left - pad_right;
            reference_blob.h = top_blob_bordered.h - pad_top - pad_bottom;
            reference_blob.elempack = 1;

            std::vector<VkImageMat> crop_bottom_blobs(2);
            crop_bottom_blobs[0] = top_blob_bordered;
            crop_bottom_blobs[1] = reference_blob;
            std::vector<VkImageMat> crop_top_blobs(1);
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

        VkImageMat crop_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator);
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

        std::vector<VkImageMat> crop_inputs(2);
        crop_inputs[0] = top_blob_bordered;
        crop_inputs[1] = crop_param_blob;

        std::vector<VkImageMat> crop_outputs(1);
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
