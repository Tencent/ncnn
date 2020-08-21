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

#include "innerproduct_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

InnerProduct_vulkan::InnerProduct_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    flatten = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_pack4 = 0;
    pipeline_innerproduct_pack1to4 = 0;
    pipeline_innerproduct_pack4to1 = 0;
    pipeline_innerproduct_pack8 = 0;
    pipeline_innerproduct_pack1to8 = 0;
    pipeline_innerproduct_pack4to8 = 0;
    pipeline_innerproduct_pack8to4 = 0;
    pipeline_innerproduct_pack8to1 = 0;
}

int InnerProduct_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    Mat shape_flatten;
    if (shape.dims != 0)
    {
        shape_flatten = Mat(shape.w * shape.h * shape.c, (void*)0);
    }

    int num_input = weight_data_size / num_output;

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

    Mat shape_flatten_packed;
    if (shape_flatten.dims == 1) shape_flatten_packed = Mat(shape_flatten.w / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_flatten_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    // check weight shape
    Mat weight_data_packed(num_input / elempack, num_output / out_elempack, (void*)0, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
    if (!vkdev->shape_support_image_storage(weight_data_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);
        flatten->vkdev = vkdev;

        flatten->bottom_shapes.resize(1);
        flatten->bottom_shapes[0] = shape;
        flatten->top_shapes.resize(1);
        flatten->top_shapes[0] = shape_flatten;

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(4 + 10);
    specializations[0].i = bias_term;
    specializations[1].i = activation_type;
    specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[4 + 0].i = shape_flatten_packed.dims;
    specializations[4 + 1].i = shape_flatten_packed.w;
    specializations[4 + 2].i = shape_flatten_packed.h;
    specializations[4 + 3].i = shape_flatten_packed.c;
    specializations[4 + 4].i = shape_flatten_packed.cstep;
    specializations[4 + 5].i = out_shape_packed.dims;
    specializations[4 + 6].i = out_shape_packed.w;
    specializations[4 + 7].i = out_shape_packed.h;
    specializations[4 + 8].i = out_shape_packed.c;
    specializations[4 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz(std::min(64, num_output / out_elempack), 1, 1, (void*)0);
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline_innerproduct = new Pipeline(vkdev);
        pipeline_innerproduct->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct->create(LayerShaderType::innerproduct, opt, specializations);
    }

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        pipeline_innerproduct_pack4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4->create(LayerShaderType::innerproduct_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        pipeline_innerproduct_pack1to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack1to4->create(LayerShaderType::innerproduct_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        pipeline_innerproduct_pack4to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4to1->create(LayerShaderType::innerproduct_pack4to1, opt, specializations);
    }

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        pipeline_innerproduct_pack8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8->create(LayerShaderType::innerproduct_pack8, opt, specializations);
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        pipeline_innerproduct_pack1to8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack1to8->create(LayerShaderType::innerproduct_pack1to8, opt, specializations);
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        pipeline_innerproduct_pack4to8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4to8->create(LayerShaderType::innerproduct_pack4to8, opt, specializations);
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        pipeline_innerproduct_pack8to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8to4->create(LayerShaderType::innerproduct_pack8to4, opt, specializations);
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        pipeline_innerproduct_pack8to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8to1->create(LayerShaderType::innerproduct_pack8to1, opt, specializations);
    }

    return 0;
}

int InnerProduct_vulkan::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_pack4;
    pipeline_innerproduct_pack4 = 0;

    delete pipeline_innerproduct_pack1to4;
    pipeline_innerproduct_pack1to4 = 0;

    delete pipeline_innerproduct_pack4to1;
    pipeline_innerproduct_pack4to1 = 0;

    delete pipeline_innerproduct_pack8;
    pipeline_innerproduct_pack8 = 0;

    delete pipeline_innerproduct_pack1to8;
    pipeline_innerproduct_pack1to8 = 0;

    delete pipeline_innerproduct_pack4to8;
    pipeline_innerproduct_pack4to8 = 0;

    delete pipeline_innerproduct_pack8to4;
    pipeline_innerproduct_pack8to4 = 0;

    delete pipeline_innerproduct_pack8to1;
    pipeline_innerproduct_pack8to1 = 0;

    return 0;
}

int InnerProduct_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    int num_input = weight_data_size / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    // src = inch-outch
    // dst = pa-pb-inch/pa-outch/pb
    Mat weight_data_packed;
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_packed.create(num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.row(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int i = 0; i < out_elempack; i++)
                {
                    const float* k0 = weight_data_r2.row(q + i);
                    k0 += p;

                    for (int j = 0; j < elempack; j++)
                    {
                        g00[0] = k0[j];

                        g00++;
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
        convert_packing(bias_data, bias_data_packed, out_elempack);

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

int InnerProduct_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // flatten
    VkMat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_flattened.dims;
    constants[1].i = bottom_blob_flattened.w;
    constants[2].i = bottom_blob_flattened.h;
    constants[3].i = bottom_blob_flattened.c;
    constants[4].i = bottom_blob_flattened.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack4to1;
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int InnerProduct_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // flatten
    VkImageMat bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(4);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu_image;
    bindings[3] = bias_data_gpu_image;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_flattened.dims;
    constants[1].i = bottom_blob_flattened.w;
    constants[2].i = bottom_blob_flattened.h;
    constants[3].i = bottom_blob_flattened.c;
    constants[4].i = 0; //bottom_blob_flattened.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = 0; //top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack4to1;
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
