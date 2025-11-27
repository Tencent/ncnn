// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "padding_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Padding_vulkan::Padding_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_padding = 0;
    pipeline_padding_pack4 = 0;
    pipeline_padding_pack1to4 = 0;
    pipeline_padding_pack4to1 = 0;

    pipeline_padding_3d = 0;
    pipeline_padding_3d_pack4 = 0;
}

int Padding_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = out_shape.c % 4 == 0 ? 4 : 1;

    int offset_elempack = 1;
    if (shape.dims == 1)
    {
        if (left == 0)
            offset_elempack = elempack;
        else
            offset_elempack = left % 4 == 0 ? 4 : 1;
    }
    else if (shape.dims == 2)
    {
        if (top == 0)
            offset_elempack = elempack;
        else
            offset_elempack = top % 4 == 0 ? 4 : 1;
    }
    else if (shape.dims == 3)
    {
        if (front == 0)
            offset_elempack = elempack;
        else
            offset_elempack = front % 4 == 0 ? 4 : 1;
    }
    else // if (shape.dims == 4)
    {
        offset_elempack = elempack;
    }

    offset_elempack = std::min(offset_elempack, elempack);

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
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    Mat shape_unpacked = shape_packed;
    if (one_blob_only && shape.dims != 0 && elempack > offset_elempack)
    {
        size_t offset_elemsize;
        if (opt.use_fp16_storage || opt.use_fp16_packed)
        {
            offset_elemsize = offset_elempack * 2u;
        }
        else
        {
            offset_elemsize = offset_elempack * 4u;
        }

        if (shape.dims == 1) shape_unpacked = Mat(shape.w / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 2) shape_unpacked = Mat(shape.w, shape.h / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 3) shape_unpacked = Mat(shape.w, shape.h, shape.c / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        // if (shape.dims == 4) should never reach here
    }

    std::vector<vk_specialization_type> specializations(3 + 10);
    specializations[0].i = type;
    specializations[1].f = value;
    specializations[2].i = per_channel_pad_data_size ? 1 : 0;
    specializations[3 + 0].i = shape_unpacked.dims;
    specializations[3 + 1].i = shape_unpacked.w;
    specializations[3 + 2].i = shape_unpacked.h;
    specializations[3 + 3].i = shape_unpacked.c;
    specializations[3 + 4].i = shape_unpacked.cstep;
    specializations[3 + 5].i = out_shape_packed.dims;
    specializations[3 + 6].i = out_shape_packed.w;
    specializations[3 + 7].i = out_shape_packed.h;
    specializations[3 + 8].i = out_shape_packed.c;
    specializations[3 + 9].i = out_shape_packed.cstep;

    std::vector<vk_specialization_type> specializations_3d(3 + 12);
    specializations_3d[0].i = type;
    specializations_3d[1].f = value;
    specializations_3d[2].i = per_channel_pad_data_size ? 1 : 0;
    specializations_3d[3 + 0].i = shape_unpacked.dims;
    specializations_3d[3 + 1].i = shape_unpacked.w;
    specializations_3d[3 + 2].i = shape_unpacked.h;
    specializations_3d[3 + 3].i = shape_unpacked.d;
    specializations_3d[3 + 4].i = shape_unpacked.c;
    specializations_3d[3 + 5].i = shape_unpacked.cstep;
    specializations_3d[3 + 6].i = out_shape_packed.dims;
    specializations_3d[3 + 7].i = out_shape_packed.w;
    specializations_3d[3 + 8].i = out_shape_packed.h;
    specializations_3d[3 + 9].i = out_shape_packed.d;
    specializations_3d[3 + 10].i = out_shape_packed.c;
    specializations_3d[3 + 11].i = out_shape_packed.cstep;

    Mat local_size_xyz;
    if (out_shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape_packed.w);
        local_size_xyz.h = std::min(8, out_shape_packed.h);
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }
    if (out_shape_packed.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h * out_shape_packed.d);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (out_shape.dims == 0 || (offset_elempack == 1 && out_elempack == 1))
    {
        pipeline_padding = new Pipeline(vkdev);
        pipeline_padding->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding->create(LayerShaderType::padding, opt, specializations);

        pipeline_padding_3d = new Pipeline(vkdev);
        pipeline_padding_3d->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_3d->create(LayerShaderType::padding_3d, opt, specializations_3d);
    }

    // pack4
    if (out_shape.dims == 0 || (offset_elempack == 4 && out_elempack == 4))
    {
        pipeline_padding_pack4 = new Pipeline(vkdev);
        pipeline_padding_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_pack4->create(LayerShaderType::padding_pack4, opt, specializations);

        pipeline_padding_3d_pack4 = new Pipeline(vkdev);
        pipeline_padding_3d_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_3d_pack4->create(LayerShaderType::padding_3d_pack4, opt, specializations_3d);
    }

    // pack1to4
    if (out_shape.dims == 0 || (offset_elempack == 1 && out_elempack == 4))
    {
        pipeline_padding_pack1to4 = new Pipeline(vkdev);
        pipeline_padding_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_pack1to4->create(LayerShaderType::padding_pack1to4, opt, specializations);
    }

    // pack4to1
    if (out_shape.dims == 0 || (offset_elempack == 4 && out_elempack == 1))
    {
        pipeline_padding_pack4to1 = new Pipeline(vkdev);
        pipeline_padding_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_pack4to1->create(LayerShaderType::padding_pack4to1, opt, specializations);
    }

    return 0;
}

int Padding_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_padding;
    pipeline_padding = 0;

    delete pipeline_padding_pack4;
    pipeline_padding_pack4 = 0;

    delete pipeline_padding_pack1to4;
    pipeline_padding_pack1to4 = 0;

    delete pipeline_padding_pack4to1;
    pipeline_padding_pack4to1 = 0;

    delete pipeline_padding_3d;
    pipeline_padding_3d = 0;

    delete pipeline_padding_3d_pack4;
    pipeline_padding_3d_pack4 = 0;

    return 0;
}

int Padding_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (per_channel_pad_data_size == 0)
        return 0;

    cmd.record_upload(per_channel_pad_data, per_channel_pad_data_gpu, opt);

    if (opt.lightmode)
    {
        per_channel_pad_data.release();
    }

    return 0;
}

int Padding_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = 0;
    int outh = 0;
    int outd = 0;
    int outc = 0;

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (left == 0 && right == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w * elempack + left + right;
        out_elempack = outw % 4 == 0 ? 4 : 1;
        offset_elempack = left == 0 ? elempack : left % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (top == 0 && bottom == 0 && left == 0 && right == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + left + right;
        outh = h * elempack + top + bottom;
        out_elempack = outh % 4 == 0 ? 4 : 1;
        offset_elempack = top == 0 ? elempack : top % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + left + right;
        outh = h + top + bottom;
        outc = channels * elempack + front + behind;
        out_elempack = outc % 4 == 0 ? 4 : 1;
        offset_elempack = front == 0 ? elempack : front % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + left + right;
        outh = h + top + bottom;
        outd = d + front + behind;
        outc = channels * elempack;
        out_elempack = elempack;
        offset_elempack = elempack;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    // unpacking
    VkMat bottom_blob_unpacked = bottom_blob;
    if (elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;
    bindings[2] = per_channel_pad_data_gpu;

    if (dims == 4)
    {
        std::vector<vk_constant_type> constants(15);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.d;
        constants[4].i = bottom_blob_unpacked.c;
        constants[5].i = bottom_blob_unpacked.cstep;
        constants[6].i = top_blob.dims;
        constants[7].i = top_blob.w;
        constants[8].i = top_blob.h;
        constants[9].i = top_blob.d;
        constants[10].i = top_blob.c;
        constants[11].i = top_blob.cstep;
        constants[12].i = left;
        constants[13].i = top;
        constants[14].i = front;

        const Pipeline* pipeline = out_elempack == 4 ? pipeline_padding_3d_pack4 : pipeline_padding_3d;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.c;
    constants[4].i = bottom_blob_unpacked.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].i = left;
    constants[11].i = top;
    constants[12].i = front;

    const Pipeline* pipeline = 0;
    if (offset_elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_padding;
    }
    else if (offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_padding_pack4;
    }
    else if (offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_padding_pack1to4;
    }
    else if (offset_elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_padding_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Padding_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];
    int _top;
    int _bottom;
    int _left;
    int _right;
    int _front;
    int _behind;
    {
        const int* param_data = reference_blob.mapped();

        _top = param_data[0];
        _bottom = param_data[1];
        _left = param_data[2];
        _right = param_data[3];
        _front = param_data[4];
        _behind = param_data[5];
    }

    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = 0;
    int outh = 0;
    int outd = 0;
    int outc = 0;

    int offset_elempack;
    int out_elempack;

    if (dims == 1)
    {
        if (_left == 0 && _right == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w * elempack + _left + _right;
        out_elempack = outw % 4 == 0 ? 4 : 1;
        offset_elempack = _left == 0 ? elempack : _left % 4 == 0 ? 4 : 1;
    }
    else if (dims == 2)
    {
        if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + _left + _right;
        outh = h * elempack + _top + _bottom;
        out_elempack = outh % 4 == 0 ? 4 : 1;
        offset_elempack = _top == 0 ? elempack : _top % 4 == 0 ? 4 : 1;
    }
    else if (dims == 3)
    {
        if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0 && _front == 0 && _behind == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + _left + _right;
        outh = h + _top + _bottom;
        outc = channels * elempack + _front + _behind;
        out_elempack = outc % 4 == 0 ? 4 : 1;
        offset_elempack = _front == 0 ? elempack : _front % 4 == 0 ? 4 : 1;
    }
    else // if (dims == 4)
    {
        if (_top == 0 && _bottom == 0 && _left == 0 && _right == 0 && _front == 0 && _behind == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        outw = w + _left + _right;
        outh = h + _top + _bottom;
        outd = d + _front + _behind;
        outc = channels * elempack;
        out_elempack = elempack;
        offset_elempack = elempack;
    }

    offset_elempack = std::min(offset_elempack, elempack);

    size_t out_elemsize = elemsize / elempack * out_elempack;

    // unpacking
    VkMat bottom_blob_unpacked = bottom_blob;
    if (elempack > offset_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
    }

    if (dims == 1)
    {
        top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 2)
    {
        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else if (dims == 3)
    {
        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else // if (dims == 4)
    {
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob_unpacked;
    bindings[1] = top_blob;
    bindings[2] = per_channel_pad_data_gpu;

    if (dims == 4)
    {
        std::vector<vk_constant_type> constants(15);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.d;
        constants[4].i = bottom_blob_unpacked.c;
        constants[5].i = bottom_blob_unpacked.cstep;
        constants[6].i = top_blob.dims;
        constants[7].i = top_blob.w;
        constants[8].i = top_blob.h;
        constants[9].i = top_blob.d;
        constants[10].i = top_blob.c;
        constants[11].i = top_blob.cstep;
        constants[12].i = _left;
        constants[13].i = _top;
        constants[14].i = _front;

        const Pipeline* pipeline = out_elempack == 4 ? pipeline_padding_3d_pack4 : pipeline_padding_3d;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    std::vector<vk_constant_type> constants(13);
    constants[0].i = bottom_blob_unpacked.dims;
    constants[1].i = bottom_blob_unpacked.w;
    constants[2].i = bottom_blob_unpacked.h;
    constants[3].i = bottom_blob_unpacked.c;
    constants[4].i = bottom_blob_unpacked.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].i = _left;
    constants[11].i = _top;
    constants[12].i = _front;

    const Pipeline* pipeline = 0;
    if (offset_elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_padding;
    }
    else if (offset_elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_padding_pack4;
    }
    else if (offset_elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_padding_pack1to4;
    }
    else if (offset_elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_padding_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
