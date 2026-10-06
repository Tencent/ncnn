// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tile_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

Tile_vulkan::Tile_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_tile = 0;
    pipeline_tile_pack4 = 0;
}

int Tile_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const bool shapes_known = shape.dims != 0 && out_shape.dims != 0;

    // numpy style repeats may promote the output dims, in which case the
    // packed axis semantics change and the gather is done on elempack 1 blobs
    const bool dims_promotion = shapes_known && out_shape.dims > shape.dims;

    const int gather_elempack = dims_promotion ? 1 : shape.elempack;

    // the specialization constants describe the blobs the shader actually
    // reads from and writes to, which may be unpacked relative to the blob
    // shapes when a convert_packing happens before or after the gather
    Mat shape_gathered = shape;
    Mat out_shape_gathered = out_shape;
    if (shapes_known)
    {
        {
            const int elempack1 = std::min(shape.elempack, gather_elempack);
            if (shape.elempack != elempack1)
            {
                size_t elemsize1 = shape.elemsize / shape.elempack * elempack1;
                if (shape.dims == 1) shape_gathered = Mat(shape.w * shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (shape.dims == 2) shape_gathered = Mat(shape.w, shape.h * shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (shape.dims == 3) shape_gathered = Mat(shape.w, shape.h, shape.c * shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (shape.dims == 4) shape_gathered = Mat(shape.w, shape.h, shape.d, shape.c * shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
            }
        }
        {
            const int elempack1 = std::min(out_shape.elempack, gather_elempack);
            if (out_shape.elempack != elempack1)
            {
                size_t elemsize1 = out_shape.elemsize / out_shape.elempack * elempack1;
                if (out_shape.dims == 1) out_shape_gathered = Mat(out_shape.w * out_shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (out_shape.dims == 2) out_shape_gathered = Mat(out_shape.w, out_shape.h * out_shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (out_shape.dims == 3) out_shape_gathered = Mat(out_shape.w, out_shape.h, out_shape.c * out_shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
                if (out_shape.dims == 4) out_shape_gathered = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c * out_shape.elempack / elempack1, (void*)0, elemsize1, elempack1);
            }
        }
    }

    std::vector<vk_specialization_type> specializations(1 + 12);
    specializations[0].i = vkdev->info.bug_implicit_fp16_arithmetic();
    specializations[1 + 0].i = shape_gathered.dims;
    specializations[1 + 1].i = shape_gathered.w;
    specializations[1 + 2].i = shape_gathered.h;
    specializations[1 + 3].i = shape_gathered.d;
    specializations[1 + 4].i = shape_gathered.c;
    specializations[1 + 5].i = shape_gathered.cstep;
    specializations[1 + 6].i = out_shape_gathered.dims;
    specializations[1 + 7].i = out_shape_gathered.w;
    specializations[1 + 8].i = out_shape_gathered.h;
    specializations[1 + 9].i = out_shape_gathered.d;
    specializations[1 + 10].i = out_shape_gathered.c;
    specializations[1 + 11].i = out_shape_gathered.cstep;

    Mat local_size_xyz;
    if (out_shape_gathered.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape_gathered.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_gathered.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape_gathered.w);
        local_size_xyz.h = std::min(8, out_shape_gathered.h);
        local_size_xyz.c = 1;
    }
    if (out_shape_gathered.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_gathered.w);
        local_size_xyz.h = std::min(4, out_shape_gathered.h);
        local_size_xyz.c = std::min(4, out_shape_gathered.c);
    }
    if (out_shape_gathered.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape_gathered.w);
        local_size_xyz.h = std::min(4, out_shape_gathered.h * out_shape_gathered.d);
        local_size_xyz.c = std::min(4, out_shape_gathered.c);
    }

    // pack1
    if (!shapes_known || shape.elempack == 1 || dims_promotion)
    {
        pipeline_tile = new Pipeline(vkdev);
        pipeline_tile->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_tile->create(LayerShaderType::tile, opt, specializations);
    }

    // pack4
    if (!shapes_known || (shape.elempack == 4 && !dims_promotion))
    {
        pipeline_tile_pack4 = new Pipeline(vkdev);
        pipeline_tile_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_tile_pack4->create(LayerShaderType::tile_pack4, opt, specializations);
    }

    return 0;
}

int Tile_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_tile;
    pipeline_tile = 0;

    delete pipeline_tile_pack4;
    pipeline_tile_pack4 = 0;

    return 0;
}

int Tile_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;

    int repeat_w = 1;
    int repeat_h = 1;
    int repeat_d = 1;
    int repeat_c = 1;

    const int repeats_num = repeats.w;

    if (repeats.empty())
    {
        if (dims == 1) // axis == 0
        {
            repeat_w = tiles;
        }
        else if (dims == 2)
        {
            if (axis == 0) repeat_h = tiles;
            if (axis == 1) repeat_w = tiles;
        }
        else if (dims == 3)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_h = tiles;
            if (axis == 2) repeat_w = tiles;
        }
        else if (dims == 4)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_d = tiles;
            if (axis == 2) repeat_h = tiles;
            if (axis == 3) repeat_w = tiles;
        }
    }
    else
    {
        // numpy style tile
        const int* repeats_ptr = repeats;

        if (repeats_num == 1)
        {
            repeat_w = repeats_ptr[0];
        }
        if (repeats_num == 2)
        {
            repeat_h = repeats_ptr[0];
            repeat_w = repeats_ptr[1];
        }
        if (repeats_num == 3)
        {
            if (dims == 4)
            {
                repeat_d = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
            else
            {
                repeat_c = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
        }
        if (repeats_num == 4)
        {
            repeat_c = repeats_ptr[0];
            repeat_d = repeats_ptr[1];
            repeat_h = repeats_ptr[2];
            repeat_w = repeats_ptr[3];
        }
    }

    const int outdims = std::max(dims, repeats_num);

    if (repeat_w == 1 && repeat_h == 1 && repeat_d == 1 && repeat_c == 1)
    {
        // all ones
        if (repeats_num == 0 || dims == repeats_num)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    // output sizes in unpacked element counts
    const int outw = w * (dims == 1 ? elempack : 1) * repeat_w;
    const int outh = h * (dims == 2 ? elempack : 1) * repeat_h;
    const int outd = d * repeat_d;
    const int outc = channels * (dims == 3 || dims == 4 ? elempack : 1) * repeat_c;

    // packed gather is only possible when the packed axis semantics is preserved
    const int gather_elempack = outdims == dims ? elempack : 1;

    int out_elempack;
    if (outdims == 1)
        out_elempack = outw % 4 == 0 ? 4 : 1;
    else if (outdims == 2)
        out_elempack = outh % 4 == 0 ? 4 : 1;
    else // if (outdims == 3 || outdims == 4)
        out_elempack = outc % 4 == 0 ? 4 : 1;

    // unpacking
    VkMat bottom_blob_gathered = bottom_blob;
    if (elempack > gather_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_gathered, gather_elempack, cmd, opt_pack1);
    }

    const size_t gather_elemsize = elemsize / elempack * gather_elempack;
    const size_t out_elemsize = elemsize / elempack * out_elempack;

    VkMat top_blob_gathered;
    {
        VkMat& dst_blob = gather_elempack == out_elempack ? top_blob : top_blob_gathered;
        VkAllocator* dst_allocator = gather_elempack == out_elempack ? opt.blob_vkallocator : opt.workspace_vkallocator;

        if (outdims == 1)
        {
            dst_blob.create(outw / gather_elempack, gather_elemsize, gather_elempack, dst_allocator);
        }
        if (outdims == 2)
        {
            dst_blob.create(outw, outh / gather_elempack, gather_elemsize, gather_elempack, dst_allocator);
        }
        if (outdims == 3)
        {
            dst_blob.create(outw, outh, outc / gather_elempack, gather_elemsize, gather_elempack, dst_allocator);
        }
        if (outdims == 4)
        {
            dst_blob.create(outw, outh, outd, outc / gather_elempack, gather_elemsize, gather_elempack, dst_allocator);
        }
        if (dst_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob_gathered;
        bindings[1] = dst_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob_gathered.dims;
        constants[1].i = bottom_blob_gathered.w;
        constants[2].i = bottom_blob_gathered.h;
        constants[3].i = bottom_blob_gathered.d;
        constants[4].i = bottom_blob_gathered.c;
        constants[5].i = bottom_blob_gathered.cstep;
        constants[6].i = dst_blob.dims;
        constants[7].i = dst_blob.w;
        constants[8].i = dst_blob.h;
        constants[9].i = dst_blob.d;
        constants[10].i = dst_blob.c;
        constants[11].i = dst_blob.cstep;

        const Pipeline* pipeline = gather_elempack == 4 ? pipeline_tile_pack4 : pipeline_tile;

        cmd.record_pipeline(pipeline, bindings, constants, dst_blob);
    }

    // packing
    if (gather_elempack != out_elempack)
    {
        if (outdims == 1)
        {
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 2)
        {
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (outdims == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        }
        if (top_blob.empty())
            return -100;

        vkdev->convert_packing(top_blob_gathered, top_blob, out_elempack, cmd, opt);
    }

    return 0;
}

} // namespace ncnn
