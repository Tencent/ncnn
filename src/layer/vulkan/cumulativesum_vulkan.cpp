// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

CumulativeSum_vulkan::CumulativeSum_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_cumulativesum_blockscan = 0;
    pipeline_cumulativesum_blocksums_scan = 0;
    pipeline_cumulativesum_addoffset = 0;
}

int CumulativeSum_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    std::vector<vk_specialization_type> specializations;

    Mat local_size_xyz;
    local_size_xyz.w = 256;
    local_size_xyz.h = 1;
    local_size_xyz.c = 1;

    pipeline_cumulativesum_blockscan = new Pipeline(vkdev);
    pipeline_cumulativesum_blockscan->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_cumulativesum_blockscan->create(LayerShaderType::cumulativesum_blockscan, opt, specializations);

    pipeline_cumulativesum_blocksums_scan = new Pipeline(vkdev);
    pipeline_cumulativesum_blocksums_scan->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_cumulativesum_blocksums_scan->create(LayerShaderType::cumulativesum_blocksums_scan, opt, specializations);

    pipeline_cumulativesum_addoffset = new Pipeline(vkdev);
    pipeline_cumulativesum_addoffset->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_cumulativesum_addoffset->create(LayerShaderType::cumulativesum_addoffset, opt, specializations);

    return 0;
}

int CumulativeSum_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_cumulativesum_blockscan;
    pipeline_cumulativesum_blockscan = 0;

    delete pipeline_cumulativesum_blocksums_scan;
    pipeline_cumulativesum_blocksums_scan = 0;

    delete pipeline_cumulativesum_addoffset;
    pipeline_cumulativesum_addoffset = 0;

    return 0;
}

static inline int cumsum_positive_axis(int axis, int dims)
{
    return axis < 0 ? dims + axis : axis;
}

static inline void get_line_shape(int dims, int axis, int w, int h, int c, int& linecount, int& linelen)
{
    if (dims == 1)
    {
        linecount = 1;
        linelen = w;
        return;
    }

    if (dims == 2)
    {
        if (axis == 0)
        {
            // sum along h, each x is a line
            linecount = w;
            linelen = h;
        }
        else
        {
            // sum along w, each y is a line
            linecount = h;
            linelen = w;
        }
        return;
    }

    // dims == 3
    if (axis == 0)
    {
        // sum along c, each (x,y) is a line
        linecount = w * h;
        linelen = c;
    }
    else if (axis == 1)
    {
        // sum along h, each (q,x) is a line
        linecount = c * w;
        linelen = h;
    }
    else
    {
        // sum along w, each (q,y) is a line
        linecount = c * h;
        linelen = w;
    }
}

int CumulativeSum_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    if (bottom_top_blob.empty())
        return 0;

    if (bottom_top_blob.elempack != 1)
        return -100;

    const int dims = bottom_top_blob.dims;
    if (dims != 1 && dims != 2 && dims != 3)
        return -100;

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int c = bottom_top_blob.c;
    const int cstep = bottom_top_blob.cstep;

    int positive_axis = cumsum_positive_axis(axis, dims);

    if (dims == 1)
    {
        positive_axis = 0;
    }
    else if (dims == 2)
    {
        if (positive_axis < 0 || positive_axis > 1)
            return -100;
    }
    else // dims == 3
    {
        if (positive_axis < 0 || positive_axis > 2)
            return -100;
    }

    int linecount = 0;
    int linelen = 0;
    get_line_shape(dims, positive_axis, w, h, c, linecount, linelen);

    const int WG = 256;
    const int blocks_per_line = (linelen + WG - 1) / WG;

    // pass1 only
    if (blocks_per_line <= 1)
    {
        VkMat dummy_blocksums;
        dummy_blocksums.create(1, 1, bottom_top_blob.elemsize, 1, opt.workspace_vkallocator);
        if (dummy_blocksums.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = dummy_blocksums;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = dims;
        constants[1].i = positive_axis;
        constants[2].i = w;
        constants[3].i = h;
        constants[4].i = c;
        constants[5].i = cstep;
        constants[6].i = linelen;
        constants[7].i = linecount;

        VkMat dispatcher;
        dispatcher.w = WG;
        dispatcher.h = linecount;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_cumulativesum_blockscan, bindings, constants, dispatcher);
        return 0;
    }

    VkMat blocksums;
    blocksums.create(blocks_per_line, linecount, bottom_top_blob.elemsize, 1, opt.workspace_vkallocator);
    if (blocksums.empty())
        return -100;

    VkMat blockoffsets;
    blockoffsets.create(blocks_per_line, linecount, bottom_top_blob.elemsize, 1, opt.workspace_vkallocator);
    if (blockoffsets.empty())
        return -100;

    // pass1: blockscan
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = blocksums;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = dims;
        constants[1].i = positive_axis;
        constants[2].i = w;
        constants[3].i = h;
        constants[4].i = c;
        constants[5].i = cstep;
        constants[6].i = linelen;
        constants[7].i = linecount;

        VkMat dispatcher;
        dispatcher.w = blocks_per_line * WG;
        dispatcher.h = linecount;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_cumulativesum_blockscan, bindings, constants, dispatcher);
    }

    // pass2: scan blocksums
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = blocksums;
        bindings[1] = blockoffsets;

        std::vector<vk_constant_type> constants(2);
        constants[0].i = blocks_per_line;
        constants[1].i = linecount;

        VkMat dispatcher;
        dispatcher.w = WG;
        dispatcher.h = linecount;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_cumulativesum_blocksums_scan, bindings, constants, dispatcher);
    }

    // pass3: add offsets
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = blockoffsets;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = dims;
        constants[1].i = positive_axis;
        constants[2].i = w;
        constants[3].i = h;
        constants[4].i = c;
        constants[5].i = cstep;
        constants[6].i = linelen;
        constants[7].i = linecount;

        VkMat dispatcher;
        dispatcher.w = blocks_per_line * WG;
        dispatcher.h = linecount;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_cumulativesum_addoffset, bindings, constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
