// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_vulkan.h"

#include "expression.h"
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
#if NCNN_BATCH
    pipeline_reshape_batch_reorder = 0;
    pipeline_reshape_batch_reorder_pack4 = 0;
    pipeline_reshape_batch_reorder_pack1to4 = 0;
    pipeline_reshape_batch_reorder_pack4to1 = 0;
#endif
}

int Reshape_vulkan::create_pipeline(const Option& opt)
{
    Mat shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    Mat out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = input_batch_axis;
        specializations[1].i = output_batch_axis;

        // batch reshape may choose output packing at runtime
        pipeline_reshape_batch_reorder = new Pipeline(vkdev);
        pipeline_reshape_batch_reorder->set_optimal_local_size_xyz(Mat(4, 4, 4, (void*)0));
        pipeline_reshape_batch_reorder->create(LayerShaderType::reshape_batch_reorder, opt, specializations);

        pipeline_reshape_batch_reorder_pack4 = new Pipeline(vkdev);
        pipeline_reshape_batch_reorder_pack4->set_optimal_local_size_xyz(Mat(4, 4, 4, (void*)0));
        pipeline_reshape_batch_reorder_pack4->create(LayerShaderType::reshape_batch_reorder_pack4, opt, specializations);

        pipeline_reshape_batch_reorder_pack1to4 = new Pipeline(vkdev);
        pipeline_reshape_batch_reorder_pack1to4->set_optimal_local_size_xyz(Mat(4, 4, 4, (void*)0));
        pipeline_reshape_batch_reorder_pack1to4->create(LayerShaderType::reshape_batch_reorder_pack1to4, opt, specializations);

        pipeline_reshape_batch_reorder_pack4to1 = new Pipeline(vkdev);
        pipeline_reshape_batch_reorder_pack4to1->set_optimal_local_size_xyz(Mat(4, 4, 4, (void*)0));
        pipeline_reshape_batch_reorder_pack4to1->create(LayerShaderType::reshape_batch_reorder_pack4to1, opt, specializations);

        return 0;
    }
#endif

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

#if NCNN_BATCH
    delete pipeline_reshape_batch_reorder;
    pipeline_reshape_batch_reorder = 0;

    delete pipeline_reshape_batch_reorder_pack4;
    pipeline_reshape_batch_reorder_pack4 = 0;

    delete pipeline_reshape_batch_reorder_pack1to4;
    pipeline_reshape_batch_reorder_pack1to4 = 0;

    delete pipeline_reshape_batch_reorder_pack4to1;
    pipeline_reshape_batch_reorder_pack4to1 = 0;
#endif

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

#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
    {
        std::vector<Mat> bottom_blob_mats(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
            bottom_blob_mats[i] = bottom_blobs[i].shape();
        Mat input_shape;
        Mat output_shape;
        int input_axis = 233;
        int output_axis = 233;
        size_t input_total = 0;
        if (resolve_batch_shape(bottom_blob_mats, input_shape, output_shape, input_axis, output_axis, input_total) != 0)
            return -1;

        out_elempack = 1;
        if (opt.use_packing_layout)
            out_elempack = (output_shape.dims == 1 ? output_shape.w : output_shape.dims == 2 ? output_shape.h : output_shape.c) % 4 == 0 ? 4 : 1;

        const size_t scalar_elemsize = elemsize / elempack;
        const size_t out_elemsize = scalar_elemsize * out_elempack;

        bool reshape_zero_copy = input_axis == output_axis && output_shape.n == bottom_blob.n && elempack == out_elempack;
        if (reshape_zero_copy && elempack != 1)
        {
            const int pack_axis_size = bottom_blob.dims == 1 ? bottom_blob.w * elempack : bottom_blob.dims == 2 ? bottom_blob.h * elempack : bottom_blob.c * elempack;
            const int out_pack_axis_size = output_shape.dims == 1 ? output_shape.w : output_shape.dims == 2 ? output_shape.h : output_shape.c;
            reshape_zero_copy = pack_axis_size == out_pack_axis_size;
        }

        if (reshape_zero_copy)
        {
            int outw2 = output_shape.w;
            int outh2 = output_shape.h;
            int outd2 = output_shape.d;
            int outc2 = output_shape.c;
            if (output_shape.dims == 1)
                outw2 = output_shape.w / out_elempack;
            if (output_shape.dims == 2)
                outh2 = output_shape.h / out_elempack;
            if (output_shape.dims == 3 || output_shape.dims == 4)
                outc2 = output_shape.c / out_elempack;

            size_t outcstep = 0;
            if (output_shape.dims == 1)
                outcstep = alignSize((size_t)outw2 * out_elemsize, 16) / out_elemsize;
            if (output_shape.dims == 2)
                outcstep = alignSize((size_t)outw2 * outh2 * out_elemsize, 16) / out_elemsize;
            if (output_shape.dims == 3)
                outcstep = alignSize((size_t)outw2 * outh2 * out_elemsize, 16) / out_elemsize;
            if (output_shape.dims == 4)
                outcstep = alignSize((size_t)outw2 * outh2 * outd2 * out_elemsize, 16) / out_elemsize;

            const size_t batch_total = input_total / bottom_blob.n / elempack;
            if (bottom_blob.total() == batch_total && outcstep * outc2 == batch_total)
            {
                top_blob = bottom_blob;
                top_blob.dims = output_shape.dims;
                top_blob.w = outw2;
                top_blob.h = outh2;
                top_blob.d = outd2;
                top_blob.c = outc2;
                top_blob.cstep = outcstep;

                return 0;
            }
        }

        if (output_shape.dims == 1)
            top_blob.create(output_shape.w / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_vkallocator);
        if (output_shape.dims == 2)
            top_blob.create(output_shape.w, output_shape.h / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_vkallocator);
        if (output_shape.dims == 3)
            top_blob.create(output_shape.w, output_shape.h, output_shape.c / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_vkallocator);
        if (output_shape.dims == 4)
            top_blob.create(output_shape.w, output_shape.h, output_shape.d, output_shape.c / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_vkallocator);

        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(16);
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

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
            pipeline = pipeline_reshape_batch_reorder;
        else if (elempack == 4 && out_elempack == 4)
            pipeline = pipeline_reshape_batch_reorder_pack4;
        else if (elempack == 1 && out_elempack == 4)
            pipeline = pipeline_reshape_batch_reorder_pack1to4;
        else if (elempack == 4 && out_elempack == 1)
            pipeline = pipeline_reshape_batch_reorder_pack4to1;
        else
            return -1;

        Mat dispatcher(top_blob.w, top_blob.h, top_blob.d, top_blob.c * top_blob.n, (void*)0);
        cmd.record_pipeline(pipeline, bindings, std::vector<VkImageMat>(), constants, dispatcher);

        return 0;
    }
#endif // NCNN_BATCH

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

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

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;

        if (outw == -1)
            outw = total;

        out_elempack = outw % 4 == 0 ? 4 : 1;

        if (dims == 1 && bottom_blob.w * elempack == outw && elempack == out_elempack)
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

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
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

        if (dims == 3 && bottom_blob.c * elempack == outc && elempack == out_elempack)
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

        if (dims == 4 && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }
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
