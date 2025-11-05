// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Packing_vulkan::Packing_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_packing = 0;
    pipeline_packing_pack1to4 = 0;
    pipeline_packing_pack4to1 = 0;
}

int Packing_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int dims = shape.dims;

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

    const int local_size_x = vkdev->info.subgroup_size();

    bool use_int8_shader = cast_type_from == 4 || cast_type_to == 4;

    std::vector<vk_specialization_type> specializations(2 + 3);
    specializations[0].i = cast_type_from;
    specializations[1].i = cast_type_to;

    if (shape.dims == 0 || elempack == out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (cast_type_from == 1)
        {
            if (dims == 1 || dims == 2)
            {
                n = shape.cstep;
                c = 1;
                stride = out_shape.cstep;
            }
            if (dims == 3 || dims == 4)
            {
                n = shape.cstep;
                c = shape.c;
                stride = out_shape.cstep;
            }
        }
        else // if (cast_type_to == 1)
        {
            if (dims == 1 || dims == 2)
            {
                n = out_shape.cstep;
                c = 1;
                stride = shape.cstep;
            }
            if (dims == 3 || dims == 4)
            {
                n = out_shape.cstep;
                c = out_shape.c;
                stride = shape.cstep;
            }
        }

        specializations[2 + 0].u32 = n / 4;
        specializations[2 + 1].u32 = c;
        specializations[2 + 2].u32 = stride / 4;

        pipeline_packing = new Pipeline(vkdev);
        pipeline_packing->set_optimal_local_size_xyz(local_size_x, 1, 1);
        if (use_int8_shader)
        {
            pipeline_packing->create(LayerShaderType::packing_int8, opt, specializations);
        }
        else
        {
            pipeline_packing->create(LayerShaderType::packing, opt, specializations);
        }
    }
    if (shape.dims == 0 || elempack < out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (dims == 1)
        {
            n = 1;
            c = out_shape.w;
            stride = 1;
        }
        if (dims == 2)
        {
            n = out_shape.w;
            c = out_shape.h;
            stride = shape.w;
        }
        if (dims == 3 || dims == 4)
        {
            n = out_shape.cstep;
            c = out_shape.c;
            stride = shape.cstep;
        }

        if (shape.dims == 0 || (elempack == 1 && out_elempack == 4))
        {
            // pack1to4
            specializations[2 + 0].u32 = n;
            specializations[2 + 1].u32 = c / 4;
            specializations[2 + 2].u32 = stride;

            pipeline_packing_pack1to4 = new Pipeline(vkdev);
            pipeline_packing_pack1to4->set_optimal_local_size_xyz(local_size_x, 1, 1);
            if (use_int8_shader)
            {
                pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4_int8, opt, specializations);
            }
            else
            {
                pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4, opt, specializations);
            }
        }
    }
    if (shape.dims == 0 || elempack > out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (dims == 1)
        {
            n = 1;
            c = shape.w;
            stride = 1;
        }
        if (dims == 2)
        {
            n = shape.w;
            c = shape.h;
            stride = out_shape.w;
        }
        if (dims == 3 || dims == 4)
        {
            n = shape.cstep;
            c = shape.c;
            stride = out_shape.cstep;
        }

        if (shape.dims == 0 || (elempack == 4 && out_elempack == 1))
        {
            // pack4to1
            specializations[2 + 0].u32 = n;
            specializations[2 + 1].u32 = c / 4;
            specializations[2 + 2].u32 = stride;

            pipeline_packing_pack4to1 = new Pipeline(vkdev);
            pipeline_packing_pack4to1->set_optimal_local_size_xyz(local_size_x, 1, 1);
            if (use_int8_shader)
            {
                pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1_int8, opt, specializations);
            }
            else
            {
                pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1, opt, specializations);
            }
        }
    }

    return 0;
}

int Packing_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_packing;
    pipeline_packing = 0;

    delete pipeline_packing_pack1to4;
    pipeline_packing_pack1to4 = 0;

    delete pipeline_packing_pack4to1;
    pipeline_packing_pack4to1 = 0;

    return 0;
}

int Packing_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int elempack = bottom_blob.elempack;
    // NCNN_LOGE("Packing_vulkan b2b %d %d   %d %d", elempack, out_elempack, cast_type_from, cast_type_to);

    if (elempack == out_elempack && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    size_t out_elemsize;
    if (cast_type_to == 0)
    {
        if (opt.use_fp16_storage || opt.use_fp16_packed)
        {
            out_elemsize = out_elempack * 2u;
        }
        else
        {
            out_elemsize = out_elempack * 4u;
        }
    }
    else if (cast_type_to == 1)
    {
        out_elemsize = out_elempack * 4u;
    }
    else if (cast_type_to == 2)
    {
        out_elemsize = out_elempack * 2u;
    }
    else // if (cast_type_to == 3)
    {
        out_elemsize = out_elempack * 1u;
    }

    if (dims == 1)
    {
        if (out_elempack == 1 && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = bottom_blob.cstep * elempack;
            top_blob.elemsize = bottom_blob.elemsize / elempack;
            top_blob.elempack = out_elempack;
            return 0;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 4)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    std::vector<VkMat> buffer_bindings(4);
    buffer_bindings[0] = bottom_blob;
    buffer_bindings[1] = bottom_blob;
    buffer_bindings[2] = top_blob;
    buffer_bindings[3] = top_blob;

    if (elempack == out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (cast_type_from == 1)
        {
            if (dims == 1 || dims == 2)
            {
                n = bottom_blob.cstep * elempack;
                c = 1;
                stride = top_blob.cstep * out_elempack;
            }
            if (dims == 3 || dims == 4)
            {
                n = bottom_blob.cstep * elempack;
                c = bottom_blob.c;
                stride = top_blob.cstep * out_elempack;
            }
        }
        else // if (cast_type_to == 1)
        {
            if (dims == 1 || dims == 2)
            {
                n = top_blob.cstep * out_elempack;
                c = 1;
                stride = bottom_blob.cstep * elempack;
            }
            if (dims == 3 || dims == 4)
            {
                n = top_blob.cstep * out_elempack;
                c = top_blob.c;
                stride = bottom_blob.cstep * elempack;
            }
        }

        std::vector<vk_constant_type> constants(3);
        constants[0].u32 = n / 4;
        constants[1].u32 = c;
        constants[2].u32 = stride / 4;

        VkMat dispatcher;
        dispatcher.w = n / 4;
        dispatcher.h = c;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_packing, buffer_bindings, constants, dispatcher);
    }
    if (elempack < out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (dims == 1)
        {
            n = 1;
            c = top_blob.w;
            stride = 1;
        }
        if (dims == 2)
        {
            n = top_blob.w;
            c = top_blob.h;
            stride = bottom_blob.w;
        }
        if (dims == 3 || dims == 4)
        {
            n = top_blob.cstep;
            c = top_blob.c;
            stride = bottom_blob.cstep;
        }

        std::vector<vk_constant_type> constants(3);
        constants[0].u32 = n;
        constants[1].u32 = c;
        constants[2].u32 = stride;

        // NCNN_LOGE("n = %u   c = %u  stride = %u", n, c, stride);

        VkMat dispatcher;
        dispatcher.w = n;
        dispatcher.h = c;
        dispatcher.c = 1;

        if (elempack == 1 && out_elempack == 4)
        {
            cmd.record_pipeline(pipeline_packing_pack1to4, buffer_bindings, constants, dispatcher);
        }
    }
    if (elempack > out_elempack)
    {
        size_t n = 0;
        size_t c = 0;
        size_t stride = 0;
        if (dims == 1)
        {
            n = 1;
            c = bottom_blob.w;
            stride = 1;
        }
        if (dims == 2)
        {
            n = bottom_blob.w;
            c = bottom_blob.h;
            stride = top_blob.w;
        }
        if (dims == 3 || dims == 4)
        {
            n = bottom_blob.cstep;
            c = bottom_blob.c;
            stride = top_blob.cstep;
        }

        std::vector<vk_constant_type> constants(3);
        constants[0].u32 = n;
        constants[1].u32 = c;
        constants[2].u32 = stride;

        // NCNN_LOGE("n = %u   c = %u  stride = %u", n, c, stride);

        VkMat dispatcher;
        dispatcher.w = n;
        dispatcher.h = c;
        dispatcher.c = 1;

        if (elempack == 4 && out_elempack == 1)
        {
            cmd.record_pipeline(pipeline_packing_pack4to1, buffer_bindings, constants, dispatcher);
        }
    }

    return 0;
}

} // namespace ncnn
