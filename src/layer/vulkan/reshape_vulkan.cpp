// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_vulkan.h"

#include "layer_type.h"
#include "layer_shader_type.h"

namespace ncnn {

Reshape_vulkan::Reshape_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_reshape = 0;
    pipeline_reshape_pack4 = 0;
    pipeline_reshape_pack1to4 = 0;
    pipeline_reshape_pack4to1 = 0;
    pipeline_reshape_batch_reorder = 0;
}

int Reshape_vulkan::create_pipeline(const Option& opt)
{
    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    if (batch_mode != 0)
    {
        shape = Mat();
        out_shape = Mat();
    }

    std::vector<vk_specialization_type> specializations(1 + 12);
    specializations[0].i = ndim;
    specializations[1 + 0].i = shape.dims;
    specializations[1 + 1].i = shape.w;
    specializations[1 + 2].i = shape.h;
    specializations[1 + 3].i = shape.d;
    specializations[1 + 4].i = shape.c;
    specializations[1 + 5].i = shape.cstep;
    specializations[1 + 6].i = out_shape.dims;
    specializations[1 + 7].i = out_shape.w;
    specializations[1 + 8].i = out_shape.h;
    specializations[1 + 9].i = out_shape.d;
    specializations[1 + 10].i = out_shape.c;
    specializations[1 + 11].i = out_shape.cstep;

    Mat local_size_xyz_bottom; // pack4to1
    if (shape.dims == 1)
    {
        local_size_xyz_bottom.w = std::min(64, shape.w);
        local_size_xyz_bottom.h = 1;
        local_size_xyz_bottom.c = 1;
    }
    if (shape.dims == 2)
    {
        local_size_xyz_bottom.w = std::min(8, shape.w);
        local_size_xyz_bottom.h = std::min(8, shape.h);
        local_size_xyz_bottom.c = 1;
    }
    if (shape.dims == 3)
    {
        local_size_xyz_bottom.w = std::min(4, shape.w);
        local_size_xyz_bottom.h = std::min(4, shape.h);
        local_size_xyz_bottom.c = std::min(4, shape.c);
    }
    if (shape.dims == 4)
    {
        local_size_xyz_bottom.w = std::min(4, shape.w);
        local_size_xyz_bottom.h = std::min(4, shape.h * shape.d);
        local_size_xyz_bottom.c = std::min(4, shape.c);
    }

    Mat local_size_xyz;
    if (out_shape.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape.w);
        local_size_xyz.h = std::min(8, out_shape.h);
        local_size_xyz.c = 1;
    }
    if (out_shape.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h);
        local_size_xyz.c = std::min(4, out_shape.c);
    }
    if (out_shape.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h * out_shape.d);
        local_size_xyz.c = std::min(4, out_shape.c);
    }

    // pack1
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 1))
    {
        pipeline_reshape = new Pipeline(vkdev);
        pipeline_reshape->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape->create(LayerShaderType::reshape, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (shape.elempack == 4 && out_shape.elempack == 4))
    {
        pipeline_reshape_pack4 = new Pipeline(vkdev);
        pipeline_reshape_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack4->create(LayerShaderType::reshape_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 4))
    {
        pipeline_reshape_pack1to4 = new Pipeline(vkdev);
        pipeline_reshape_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack1to4->create(LayerShaderType::reshape_pack1to4, opt, specializations);
    }

    // pack4to1
    if (shape.dims == 0 || (shape.elempack == 4 && out_shape.elempack == 1))
    {
        pipeline_reshape_pack4to1 = new Pipeline(vkdev);
        pipeline_reshape_pack4to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_reshape_pack4to1->create(LayerShaderType::reshape_pack4to1, opt, specializations);
    }

    if (batch_mode != 0)
    {
        pipeline_reshape_batch_reorder = new Pipeline(vkdev);
        pipeline_reshape_batch_reorder->set_optimal_local_size_xyz(Mat(4, 4, 4, (void*)0));
        pipeline_reshape_batch_reorder->create(LayerShaderType::reshape_batch_reorder, opt, std::vector<vk_specialization_type>());
    }

    return 0;
}

int Reshape_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_reshape;
    pipeline_reshape = 0;

    delete pipeline_reshape_pack4;
    pipeline_reshape_pack4 = 0;

    delete pipeline_reshape_pack1to4;
    pipeline_reshape_pack1to4 = 0;

    delete pipeline_reshape_pack4to1;
    pipeline_reshape_pack4to1 = 0;

    delete pipeline_reshape_batch_reorder;
    pipeline_reshape_batch_reorder = 0;

    return 0;
}

int Reshape_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_blobs(1);
    bottom_blobs[0] = bottom_blob;
    std::vector<VkMat> top_blobs(1);
    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Reshape_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    VkMat& top_blob = top_blobs[0];

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int out_elempack = 0;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
    if (batch_mode == 1)
        total *= bottom_blob.n;

    if (batch_mode != 0 && (ndim == 0 || elempack != 1))
        return -1;

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        int er = eval_shape_expr(bottom_blob_shapes, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;

        if (outw == -1)
            outw = total;

        out_elempack = outw % 4 == 0 ? 4 : 1;

        if (batch_mode == 0 && dims == 1 && bottom_blob.w * elempack == outw && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        out_elempack = outh % 4 == 0 ? 4 : 1;

        if (batch_mode == 0 && dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        out_elempack = outc % 4 == 0 ? 4 : 1;

        if (batch_mode == 0 && dims == 3 && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outd == 0)
            outd = bottom_blob.d;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;

        out_elempack = outc % 4 == 0 ? 4 : 1;

        if (batch_mode == 0 && dims == 4 && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }
    }

    if (batch_mode == 1)
    {
        if (batch_axis != 0)
        {
            if (ndim == 1)
                top_blob.create(outw, elemsize, 1, opt.blob_vkallocator);
            if (ndim == 2)
                top_blob.create(outw, outh, elemsize, 1, opt.blob_vkallocator);
            if (ndim == 3)
                top_blob.create(outw, outh, outc, elemsize, 1, opt.blob_vkallocator);
            if (ndim == 4)
                top_blob.create(outw, outh, outd, outc, elemsize, 1, opt.blob_vkallocator);

            if (top_blob.empty())
                return -100;

            int shape[4] = {0, 0, 0, 0};
            if (ndim == 1)
                shape[0] = outw;
            if (ndim == 2)
            {
                shape[0] = outh;
                shape[1] = outw;
            }
            if (ndim == 3)
            {
                shape[0] = outc;
                shape[1] = outh;
                shape[2] = outw;
            }
            if (ndim == 4)
            {
                shape[0] = outc;
                shape[1] = outd;
                shape[2] = outh;
                shape[3] = outw;
            }

            int prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            int suffix = 1;
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(20);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = bottom_blob.n;
            constants[7].i = bottom_blob.nstep;
            constants[8].i = top_blob.dims;
            constants[9].i = top_blob.w;
            constants[10].i = top_blob.h;
            constants[11].i = top_blob.d;
            constants[12].i = top_blob.c;
            constants[13].i = top_blob.cstep;
            constants[14].i = top_blob.n;
            constants[15].i = top_blob.nstep;
            constants[16].i = 1;
            constants[17].i = prefix;
            constants[18].i = suffix;
            constants[19].i = bottom_blob.n;

            cmd.record_pipeline(pipeline_reshape_batch_reorder, bindings, constants, top_blob);

            return 0;
        }

        VkMat bottom_blob_flattened;
        bottom_blob_flattened.create(total, elemsize, 1, opt.blob_vkallocator);
        if (bottom_blob_flattened.empty())
            return -100;

        const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;
        int offset = 0;
        for (int b = 0; b < bottom_blob.n; b++)
        {
            for (int q = 0; q < bottom_blob.c; q++)
            {
                VkMat src(size, bottom_blob.data, elemsize, 1, bottom_blob.allocator);
                src.cstep = size;
                src.nstep = size;
                src.offset = bottom_blob.offset + ((size_t)bottom_blob.nstep * b + bottom_blob.cstep * q) * elemsize;

                VkMat dst(size, bottom_blob_flattened.data, elemsize, 1, bottom_blob_flattened.allocator);
                dst.cstep = size;
                dst.nstep = size;
                dst.offset = bottom_blob_flattened.offset + (size_t)offset * elemsize;

                cmd.record_clone(src, dst, opt);
                offset += size;
            }
        }

        if (ndim == 1 && outw == total)
        {
            top_blob = bottom_blob_flattened;
            return 0;
        }

        if (ndim == 1)
            top_blob.create(outw, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 2)
            top_blob.create(outw, outh, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc, elemsize, 1, opt.blob_vkallocator);

        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob_flattened;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob_flattened.dims;
        constants[1].i = bottom_blob_flattened.w;
        constants[2].i = bottom_blob_flattened.h;
        constants[3].i = bottom_blob_flattened.d;
        constants[4].i = bottom_blob_flattened.c;
        constants[5].i = bottom_blob_flattened.cstep;
        constants[6].i = top_blob.dims;
        constants[7].i = top_blob.w;
        constants[8].i = top_blob.h;
        constants[9].i = top_blob.d;
        constants[10].i = top_blob.c;
        constants[11].i = top_blob.cstep;

        cmd.record_pipeline(pipeline_reshape, bindings, constants, top_blob);

        return 0;
    }

    if (batch_mode == 2)
    {
        if (bottom_blob.n != 1)
            return -1;

        size_t out_total = outw;
        if (ndim == 2)
            out_total *= outh;
        if (ndim == 3)
            out_total *= (size_t)outh * outc;
        if (ndim == 4)
            out_total *= (size_t)outh * outd * outc;

        if (out_total == 0)
            return -1;

        const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
        const int batch = bottom_total / out_total;
        if ((size_t)batch * out_total != bottom_total)
            return -1;

        if (ndim == 1)
            top_blob.create_batch(outw, batch, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 2)
            top_blob.create_batch(outw, outh, batch, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 3)
            top_blob.create_batch(outw, outh, outc, batch, elemsize, 1, opt.blob_vkallocator);
        if (ndim == 4)
            top_blob.create_batch(outw, outh, outd, outc, batch, elemsize, 1, opt.blob_vkallocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis != 0)
        {
            int shape[4] = {0, 0, 0, 0};
            if (ndim == 1)
                shape[0] = outw;
            if (ndim == 2)
            {
                shape[0] = outh;
                shape[1] = outw;
            }
            if (ndim == 3)
            {
                shape[0] = outc;
                shape[1] = outh;
                shape[2] = outw;
            }
            if (ndim == 4)
            {
                shape[0] = outc;
                shape[1] = outd;
                shape[2] = outh;
                shape[3] = outw;
            }

            int prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            int suffix = 1;
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(20);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.d;
            constants[4].i = bottom_blob.c;
            constants[5].i = bottom_blob.cstep;
            constants[6].i = bottom_blob.n;
            constants[7].i = bottom_blob.nstep;
            constants[8].i = top_blob.dims;
            constants[9].i = top_blob.w;
            constants[10].i = top_blob.h;
            constants[11].i = top_blob.d;
            constants[12].i = top_blob.c;
            constants[13].i = top_blob.cstep;
            constants[14].i = top_blob.n;
            constants[15].i = top_blob.nstep;
            constants[16].i = 2;
            constants[17].i = prefix;
            constants[18].i = suffix;
            constants[19].i = batch;

            Mat dispatcher(top_blob.w, top_blob.h, top_blob.d, top_blob.c * batch, (void*)0);
            cmd.record_pipeline(pipeline_reshape_batch_reorder, bindings, std::vector<VkImageMat>(), constants, dispatcher);

            return 0;
        }

        VkMat bottom_blob_flattened;
        bottom_blob_flattened.create(bottom_total, elemsize, 1, opt.blob_vkallocator);
        if (bottom_blob_flattened.empty())
            return -100;

        const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;
        int offset = 0;
        for (int q = 0; q < bottom_blob.c; q++)
        {
            VkMat src(size, bottom_blob.data, elemsize, 1, bottom_blob.allocator);
            src.cstep = size;
            src.nstep = size;
            src.offset = bottom_blob.offset + bottom_blob.cstep * q * elemsize;

            VkMat dst(size, bottom_blob_flattened.data, elemsize, 1, bottom_blob_flattened.allocator);
            dst.cstep = size;
            dst.nstep = size;
            dst.offset = bottom_blob_flattened.offset + (size_t)offset * elemsize;

            cmd.record_clone(src, dst, opt);
            offset += size;
        }

        const int out_size = top_blob.w * top_blob.h * top_blob.d;
        offset = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                VkMat src(out_size, bottom_blob_flattened.data, elemsize, 1, bottom_blob_flattened.allocator);
                src.cstep = out_size;
                src.nstep = out_size;
                src.offset = bottom_blob_flattened.offset + (size_t)offset * elemsize;

                VkMat dst(out_size, top_blob.data, elemsize, 1, top_blob.allocator);
                dst.cstep = out_size;
                dst.nstep = out_size;
                dst.offset = top_blob.offset + ((size_t)top_blob.nstep * b + top_blob.cstep * q) * elemsize;

                cmd.record_clone(src, dst, opt);
                offset += out_size;
            }
        }

        return 0;
    }

    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (ndim == 1)
    {
        top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (ndim == 2)
    {
        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (ndim == 3)
    {
        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (ndim == 4)
    {
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to1, bindings, constants, bottom_blob);
    }

    return 0;
}

} // namespace ncnn
