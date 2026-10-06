// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

GLU_vulkan::GLU_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_glu = 0;
    pipeline_glu_pack4 = 0;
}

int GLU_vulkan::create_pipeline(const Option& opt)
{
    {
        pipeline_glu = new Pipeline(vkdev);
        pipeline_glu->set_local_size_xyz(8, 8, 4);
        std::vector<vk_specialization_type> specializations;
        pipeline_glu->create(LayerShaderType::glu, opt, specializations);
    }

    {
        pipeline_glu_pack4 = new Pipeline(vkdev);
        pipeline_glu_pack4->set_local_size_xyz(8, 8, 4);
        std::vector<vk_specialization_type> specializations;
        pipeline_glu_pack4->create(LayerShaderType::glu_pack4, opt, specializations);
    }

    return 0;
}

int GLU_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_glu;
    pipeline_glu = 0;

    delete pipeline_glu_pack4;
    pipeline_glu_pack4 = 0;

    return 0;
}

int GLU_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;
    const int elempack = bottom_blob.elempack;

    // unpacked sizes; the packed axis is always axis 0 (w for 1d, h for 2d, c for 3d/4d)
    const int W = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
    const int H = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
    const int D = dims == 4 ? bottom_blob.d : 1;
    const int C = dims >= 3 ? bottom_blob.c * elempack : 1;

    int outw = W;
    int outh = H;
    int outd = D;
    int outc = C;
    if (dims == 1)
        outw = W / 2;
    else if (dims == 2)
    {
        if (positive_axis == 0)
            outh = H / 2;
        else
            outw = W / 2;
    }
    else if (dims == 3)
    {
        if (positive_axis == 0)
            outc = C / 2;
        else if (positive_axis == 1)
            outh = H / 2;
        else
            outw = W / 2;
    }
    else // if (dims == 4)
    {
        if (positive_axis == 0)
            outc = C / 2;
        else if (positive_axis == 1)
            outd = D / 2;
        else if (positive_axis == 2)
            outh = H / 2;
        else
            outw = W / 2;
    }

    const int out_packed_len = dims == 1 ? outw : dims == 2 ? outh : outc;

    int out_elempack = 1;
    if (opt.use_packing_layout)
        out_elempack = out_packed_len % 4 == 0 ? 4 : 1;

    // a split along the packed axis stays vec4-wise only when the halves
    // remain 4-aligned, otherwise gather on the unpacked blob
    const bool split_packed_axis = positive_axis == 0;
    const int gather_elempack = elempack == 4 && !(split_packed_axis && out_packed_len % 4 != 0) ? 4 : 1;

    VkMat bottom_blob_gathered = bottom_blob;
    if (elempack > gather_elempack)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        vkdev->convert_packing(bottom_blob, bottom_blob_gathered, gather_elempack, cmd, opt_pack1);
    }

    const size_t gather_elemsize = bottom_blob.elemsize / elempack * gather_elempack;
    const size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;

    // stored-unit in/out sizes
    const int gw = bottom_blob_gathered.w;
    const int gh = bottom_blob_gathered.h;
    const int gd = dims == 4 ? bottom_blob_gathered.d : 1;
    const int gc = dims >= 3 ? bottom_blob_gathered.c : 1;
    const int gcstep = (int)bottom_blob_gathered.cstep;

    const int ow = dims == 1 ? outw / gather_elempack : outw;
    const int oh = dims == 2 ? outh / gather_elempack : outh;
    const int od = outd;
    const int oc = dims >= 3 ? outc / gather_elempack : 1;

    // partner linear offset in stored units
    int offset;
    if (dims == 1)
        offset = ow;
    else if (dims == 2)
        offset = positive_axis == 0 ? oh * gw : ow;
    else if (dims == 3)
        offset = positive_axis == 0 ? oc * gcstep : positive_axis == 1 ? oh * gw
                 : ow;
    else // if (dims == 4)
        offset = positive_axis == 0 ? oc * gcstep : positive_axis == 1 ? od * gh * gw
                 : positive_axis == 2   ? oh * gw
                 : ow;

    VkMat top_blob_gathered;
    {
        VkMat& dst_blob = gather_elempack == out_elempack ? top_blob : top_blob_gathered;
        VkAllocator* dst_allocator = gather_elempack == out_elempack ? opt.blob_vkallocator : opt.workspace_vkallocator;

        if (dims == 1)
            dst_blob.create(ow, gather_elemsize, gather_elempack, dst_allocator);
        if (dims == 2)
            dst_blob.create(ow, oh, gather_elemsize, gather_elempack, dst_allocator);
        if (dims == 3)
            dst_blob.create(ow, oh, oc, gather_elemsize, gather_elempack, dst_allocator);
        if (dims == 4)
            dst_blob.create(ow, oh, od, oc, gather_elemsize, gather_elempack, dst_allocator);
        if (dst_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob_gathered;
        bindings[1] = dst_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = dims;
        constants[1].i = gw;
        constants[2].i = gh;
        constants[3].i = gd;
        constants[4].i = gc;
        constants[5].i = gcstep;
        constants[6].i = ow;
        constants[7].i = oh;
        constants[8].i = od;
        constants[9].i = oc;
        constants[10].i = (int)dst_blob.cstep;
        constants[11].i = offset;

        const Pipeline* pipeline = gather_elempack == 4 ? pipeline_glu_pack4 : pipeline_glu;

        cmd.record_pipeline(pipeline, bindings, constants, dst_blob);
    }

    if (gather_elempack != out_elempack)
    {
        if (dims == 1)
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (dims == 2)
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (dims == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (dims == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        vkdev->convert_packing(top_blob_gathered, top_blob, out_elempack, cmd, opt);
    }

    return 0;
}

} // namespace ncnn
