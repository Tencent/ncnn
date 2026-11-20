// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Concat_vulkan::Concat_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_concat[0] = 0;
    pipeline_concat[1] = 0;
    pipeline_concat_pack4[0] = 0;
    pipeline_concat_pack4[1] = 0;
    pipeline_concat_pack4to1[0] = 0;
    pipeline_concat_pack4to1[1] = 0;
}

int Concat_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];
    int positive_axis = axis < 0 && shape.dims > 0 ? shape.dims + axis : axis;

    if (axis == -233 || axis < -4 || (shape.dims > 0 && positive_axis == -1))
        return 0;

    int elempack = 1;
    if (positive_axis == 0)
    {
        elempack = shape.elempack;

        for (size_t b = 1; b < bottom_shapes.size(); b++)
        {
            const Mat& shape1 = bottom_shapes[b];

            elempack = std::min(elempack, shape1.elempack);
        }
    }
    else
    {
        elempack = out_shape.elempack;
    }

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Mat out_shape_unpacked;
    if (out_shape.dims == 1) out_shape_unpacked = Mat(out_shape.w / elempack, (void*)0, elemsize, elempack);
    if (out_shape.dims == 2) out_shape_unpacked = Mat(out_shape.w, out_shape.h / elempack, (void*)0, elemsize, elempack);
    if (out_shape.dims == 3) out_shape_unpacked = Mat(out_shape.w, out_shape.h, out_shape.c / elempack, (void*)0, elemsize, elempack);
    if (out_shape.dims == 4) out_shape_unpacked = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(1 + 12);
    specializations[0].i = axis;
    specializations[1 + 0].i = 0; // TODO handle shape_packed for concat2
    specializations[1 + 1].i = 0;
    specializations[1 + 2].i = 0;
    specializations[1 + 3].i = 0;
    specializations[1 + 4].i = 0;
    specializations[1 + 5].i = 0;
    specializations[1 + 6].i = out_shape_unpacked.dims;
    specializations[1 + 7].i = out_shape_unpacked.w;
    specializations[1 + 8].i = out_shape_unpacked.h;
    specializations[1 + 9].i = out_shape_unpacked.d;
    specializations[1 + 10].i = out_shape_unpacked.c;
    specializations[1 + 11].i = out_shape_unpacked.cstep;

    Mat local_size_xyz; // TODO more precise group size guessed from out_shape_unpacked
    if (out_shape_unpacked.dims == 1)
    {
        local_size_xyz.w = 64;
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_unpacked.dims == 2)
    {
        local_size_xyz.w = 8;
        local_size_xyz.h = 8;
        local_size_xyz.c = 1;
    }
    if (out_shape_unpacked.dims == 3)
    {
        local_size_xyz.w = 4;
        local_size_xyz.h = 4;
        local_size_xyz.c = 4;
    }
    if (out_shape_unpacked.dims == 4)
    {
        local_size_xyz.w = 4;
        local_size_xyz.h = 4;
        local_size_xyz.c = 4;
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_concat[0] = new Pipeline(vkdev);
        pipeline_concat[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat[0]->create(LayerShaderType::concat, opt, specializations);
        pipeline_concat[1] = new Pipeline(vkdev);
        pipeline_concat[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat[1]->create(LayerShaderType::concat, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_concat_pack4[0] = new Pipeline(vkdev);
        pipeline_concat_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat_pack4[0]->create(LayerShaderType::concat_pack4, opt, specializations);
        pipeline_concat_pack4[1] = new Pipeline(vkdev);
        pipeline_concat_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat_pack4[1]->create(LayerShaderType::concat_pack4, opt, specializations);
    }

    // pack4to1
    if ((positive_axis <= 0 && shape.dims == 0) || elempack == 1)
    {
        pipeline_concat_pack4to1[0] = new Pipeline(vkdev);
        pipeline_concat_pack4to1[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat_pack4to1[0]->create(LayerShaderType::concat_pack4to1, opt, specializations);
        pipeline_concat_pack4to1[1] = new Pipeline(vkdev);
        pipeline_concat_pack4to1[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_concat_pack4to1[1]->create(LayerShaderType::concat_pack4to1, opt, specializations);
    }

    return 0;
}

int Concat_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_concat[0];
    delete pipeline_concat[1];
    pipeline_concat[0] = 0;
    pipeline_concat[1] = 0;

    delete pipeline_concat_pack4[0];
    delete pipeline_concat_pack4[1];
    pipeline_concat_pack4[0] = 0;
    pipeline_concat_pack4[1] = 0;

    delete pipeline_concat_pack4to1[0];
    delete pipeline_concat_pack4to1[1];
    pipeline_concat_pack4to1[0] = 0;
    pipeline_concat_pack4to1[1] = 0;

    return 0;
}

#if NCNN_BATCH
int Concat_vulkan::forward_batch(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    const bool axis_is_batch = axis == -233;
    int positive_axis = axis_is_batch ? -1 : (axis < 0 ? dims + axis : axis);

    if (axis_is_batch)
    {
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int c = bottom_blobs[0].c;

        int top_n = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            if (bottom_blob.dims != dims || bottom_blob.w != w || bottom_blob.h != h || bottom_blob.d != d || bottom_blob.c != c || bottom_blob.elemsize != elemsize || bottom_blob.elempack != elempack || bottom_blob.cstep != bottom_blobs[0].cstep)
            {
                NCNN_LOGE("concat along batch axis with mismatched shape or storage layout");
                return -100;
            }
            top_n += bottom_blob.n;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create_like(bottom_blobs[0], top_n, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int out_b = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            for (int j = 0; j < bottom_blob.n; j++)
            {
                const VkMat bottom_slice = bottom_blob.batch(j);
                VkMat top_slice = top_blob.batch(out_b + j);
                cmd.record_clone(bottom_slice, top_slice, opt);
            }
            out_b += bottom_blob.n;
        }

        return 0;
    }

    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    const Mat bottom_shape = bottom_blobs[0].shape();
    for (size_t b = 1; b < bottom_blobs.size(); b++)
    {
        if (!same_shape_except_axis(bottom_shape, bottom_blobs[b].shape(), positive_axis))
        {
            NCNN_LOGE("concat with mismatched shape on non-concat axis");
            return -100;
        }
    }

    int B = 1;
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        B = std::max(B, bottom_blobs[b].n);
    }
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        if (bottom_blobs[b].n != 1 && bottom_blobs[b].n != B)
        {
            NCNN_LOGE("concat batch size mismatch: expected batch 1 or %d but bottom %d has batch %d", B, (int)b, bottom_blobs[b].n);
            return -1;
        }
    }

    std::vector<VkMat> bottom_b(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        bottom_b[i] = bottom_blobs[i].n > 1 ? bottom_blobs[i].batch(0) : bottom_blobs[i];
    }
    std::vector<VkMat> top_b(1);
    int ret = forward(bottom_b, top_b, cmd, opt);
    if (ret != 0)
        return ret;

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(top_b[0], B, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    VkMat top_batch_slot = top_blob.batch(0);
    cmd.record_clone(top_b[0], top_batch_slot, opt);

    for (int b = 1; b < B; b++)
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_b[i] = bottom_blobs[i].n > 1 ? bottom_blobs[i].batch(b) : bottom_blobs[i];
        }
        top_b[0].release();
        ret = forward(bottom_b, top_b, cmd, opt);
        if (ret != 0)
            return ret;

        top_batch_slot = top_blob.batch(b);
        cmd.record_clone(top_b[0], top_batch_slot, opt);
    }

    return 0;
}
#endif // NCNN_BATCH

int Concat_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    int positive_axis = axis == -233 ? -1 : (axis < 0 ? dims + axis : axis);

#if NCNN_BATCH
    if (support_batch)
    {
        int B = 1;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            if (bottom_blobs[b].n > 1)
            {
                B = bottom_blobs[b].n;
                break;
            }
        }
        if (axis == -233 || B > 1)
            return forward_batch(bottom_blobs, top_blobs, cmd, opt);
    }
#else
    if (axis == -233)
    {
        NCNN_LOGE("please build ncnn with NCNN_BATCH enabled for batch concat");
        return -1;
    }
#endif

    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = top_w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        VkMat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkMat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(top_w / elempack, elemsize, elempack, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int woffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_unpacked;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob_unpacked.dims;
            constants[7].i = top_blob_unpacked.w;
            constants[8].i = top_blob_unpacked.h;
            constants[9].i = top_blob_unpacked.d;
            constants[10].i = top_blob_unpacked.c;
            constants[11].i = top_blob_unpacked.cstep;
            constants[12].i = woffset;

            const Pipeline* pipeline = 0;
            if (bottom_blob.elempack == 1 && elempack == 1)
            {
                pipeline = pipeline_concat[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 4)
            {
                pipeline = pipeline_concat_pack4[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 1)
            {
                pipeline = pipeline_concat_pack4to1[b % 2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            woffset += bottom_blob.w * bottom_blob.elempack / elempack;
        }

        // packing
        if (elempack < out_elempack)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = top_h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkMat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int hoffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_unpacked;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob_unpacked.dims;
            constants[7].i = top_blob_unpacked.w;
            constants[8].i = top_blob_unpacked.h;
            constants[9].i = top_blob_unpacked.d;
            constants[10].i = top_blob_unpacked.c;
            constants[11].i = top_blob_unpacked.cstep;
            constants[12].i = hoffset;

            const Pipeline* pipeline = 0;
            if (bottom_blob.elempack == 1 && elempack == 1)
            {
                pipeline = pipeline_concat[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 4)
            {
                pipeline = pipeline_concat_pack4[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 1)
            {
                pipeline = pipeline_concat_pack4to1[b % 2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            hoffset += bottom_blob.h * bottom_blob.elempack / elempack;
        }

        // packing
        if (elempack < out_elempack)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int woffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = woffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            woffset += bottom_blob.w;
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = top_channels % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkMat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, top_channels / elempack, elemsize, elempack, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int coffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_unpacked;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob_unpacked.dims;
            constants[7].i = top_blob_unpacked.w;
            constants[8].i = top_blob_unpacked.h;
            constants[9].i = top_blob_unpacked.d;
            constants[10].i = top_blob_unpacked.c;
            constants[11].i = top_blob_unpacked.cstep;
            constants[12].i = coffset;

            const Pipeline* pipeline = 0;
            if (bottom_blob.elempack == 1 && elempack == 1)
            {
                pipeline = pipeline_concat[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 4)
            {
                pipeline = pipeline_concat_pack4[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 1)
            {
                pipeline = pipeline_concat_pack4to1[b % 2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            coffset += bottom_blob.c * bottom_blob.elempack / elempack;
        }

        // packing
        if (elempack < out_elempack)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int hoffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = hoffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            hoffset += bottom_blob.h;
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int woffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = woffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            woffset += bottom_blob.w;
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 0)
    {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = top_channels % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        VkMat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_vkallocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int coffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob_unpacked;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob_unpacked.dims;
            constants[7].i = top_blob_unpacked.w;
            constants[8].i = top_blob_unpacked.h;
            constants[9].i = top_blob_unpacked.d;
            constants[10].i = top_blob_unpacked.c;
            constants[11].i = top_blob_unpacked.cstep;
            constants[12].i = coffset;

            const Pipeline* pipeline = 0;
            if (bottom_blob.elempack == 1 && elempack == 1)
            {
                pipeline = pipeline_concat[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 4)
            {
                pipeline = pipeline_concat_pack4[b % 2];
            }
            else if (bottom_blob.elempack == 4 && elempack == 1)
            {
                pipeline = pipeline_concat_pack4to1[b % 2];
            }

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            coffset += bottom_blob.c * bottom_blob.elempack / elempack;
        }

        // packing
        if (elempack < out_elempack)
        {
            vkdev->convert_packing(top_blob_unpacked, top_blob, out_elempack, cmd, opt);
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int doffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = doffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            doffset += bottom_blob.d;
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 2)
    {
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int hoffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = hoffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            hoffset += bottom_blob.h;
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 3)
    {
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        VkMat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        int woffset = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const VkMat& bottom_blob = bottom_blobs[b];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = top_blob.dims;
            constants[7].i = top_blob.w;
            constants[8].i = top_blob.h;
            constants[9].i = top_blob.d;
            constants[10].i = top_blob.c;
            constants[11].i = top_blob.cstep;
            constants[12].i = woffset;

            const Pipeline* pipeline = elempack == 4 ? pipeline_concat_pack4[b % 2] : pipeline_concat[b % 2];

            cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

            woffset += bottom_blob.w;
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
