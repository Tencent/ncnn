// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

CumulativeSum_vulkan::CumulativeSum_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_cumulativesum_blockscan = 0;
    pipeline_cumulativesum_blocksums_scan = 0;
    pipeline_cumulativesum_addoffset = 0;

    pipeline_cumulativesum_blockscan_pack4 = 0;
    pipeline_cumulativesum_blocksums_scan_pack4 = 0;
    pipeline_cumulativesum_addoffset_pack4 = 0;
}

int CumulativeSum_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(1 + 6);
    specializations[0].i = axis;
    specializations[1 + 0].i = shape.dims;
    specializations[1 + 1].i = shape.w;
    specializations[1 + 2].i = shape.h;
    specializations[1 + 3].i = shape.d;
    specializations[1 + 4].i = shape.c;
    specializations[1 + 5].i = shape.cstep;

    {
        pipeline_cumulativesum_blockscan = new Pipeline(vkdev);
        pipeline_cumulativesum_blockscan->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_blockscan->create(LayerShaderType::cumulativesum_blockscan, opt, specializations);
    }

    {
        pipeline_cumulativesum_blocksums_scan = new Pipeline(vkdev);
        pipeline_cumulativesum_blocksums_scan->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_blocksums_scan->create(LayerShaderType::cumulativesum_blocksums_scan, opt, specializations);
    }

    {
        pipeline_cumulativesum_addoffset = new Pipeline(vkdev);
        pipeline_cumulativesum_addoffset->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_addoffset->create(LayerShaderType::cumulativesum_addoffset, opt, specializations);
    }

    {
        pipeline_cumulativesum_blockscan_pack4 = new Pipeline(vkdev);
        pipeline_cumulativesum_blockscan_pack4->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_blockscan_pack4->create(LayerShaderType::cumulativesum_blockscan_pack4, opt, specializations);
    }

    {
        pipeline_cumulativesum_blocksums_scan_pack4 = new Pipeline(vkdev);
        pipeline_cumulativesum_blocksums_scan_pack4->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_blocksums_scan_pack4->create(LayerShaderType::cumulativesum_blocksums_scan_pack4, opt, specializations);
    }

    {
        pipeline_cumulativesum_addoffset_pack4 = new Pipeline(vkdev);
        pipeline_cumulativesum_addoffset_pack4->set_local_size_xyz(256, 1, 1);
        pipeline_cumulativesum_addoffset_pack4->create(LayerShaderType::cumulativesum_addoffset_pack4, opt, specializations);
    }

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

    delete pipeline_cumulativesum_blockscan_pack4;
    pipeline_cumulativesum_blockscan_pack4 = 0;

    delete pipeline_cumulativesum_blocksums_scan_pack4;
    pipeline_cumulativesum_blocksums_scan_pack4 = 0;

    delete pipeline_cumulativesum_addoffset_pack4;
    pipeline_cumulativesum_addoffset_pack4 = 0;

    return 0;
}

int CumulativeSum_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (elempack == 4)
    {
        // Cumulative sum over the packed axis requires intra-pack carry handling, so run scalar fallback.
        const bool packed_axis = (dims == 1) || (positive_axis == 0);
        if (packed_axis)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            VkMat bottom_top_blob_unpacked;
            vkdev->convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, cmd, opt_pack1);

            forward_inplace(bottom_top_blob_unpacked, cmd, opt_pack1);

            vkdev->convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, cmd, opt);
            return 0;
        }
    }

    int scan_size = 0;
    int scan_stride = 0;
    int num_bases = 0;

    if (dims == 1)
    {
        scan_size = w;
        scan_stride = 1;
        num_bases = 1;
    }
    else if (dims == 2)
    {
        if (positive_axis == 0)
        {
            scan_size = h;
            scan_stride = w;
            num_bases = w;
        }
        else
        {
            scan_size = w;
            scan_stride = 1;
            num_bases = h;
        }
    }
    else if (dims == 3)
    {
        if (positive_axis == 0)
        {
            scan_size = channels;
            scan_stride = bottom_top_blob.cstep;
            num_bases = w * h;
        }
        else if (positive_axis == 1)
        {
            scan_size = h;
            scan_stride = w;
            num_bases = w * channels;
        }
        else
        {
            scan_size = w;
            scan_stride = 1;
            num_bases = h * channels;
        }
    }
    else if (dims == 4)
    {
        if (positive_axis == 0)
        {
            scan_size = channels;
            scan_stride = bottom_top_blob.cstep;
            num_bases = w * h * d;
        }
        else if (positive_axis == 1)
        {
            scan_size = d;
            scan_stride = w * h;
            num_bases = w * h * channels;
        }
        else if (positive_axis == 2)
        {
            scan_size = h;
            scan_stride = w;
            num_bases = w * d * channels;
        }
        else
        {
            scan_size = w;
            scan_stride = 1;
            num_bases = h * d * channels;
        }
    }

    const int BLOCK_SIZE = 256;
    int num_blocks = (scan_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const Pipeline* pipeline_blockscan = elempack == 4 ? pipeline_cumulativesum_blockscan_pack4 : pipeline_cumulativesum_blockscan;
    const Pipeline* pipeline_blocksums_scan = elempack == 4 ? pipeline_cumulativesum_blocksums_scan_pack4 : pipeline_cumulativesum_blocksums_scan;
    const Pipeline* pipeline_addoffset = elempack == 4 ? pipeline_cumulativesum_addoffset_pack4 : pipeline_cumulativesum_addoffset;

    VkMat block_sums;
    block_sums.create(num_blocks * num_bases, elemsize, elempack, opt.workspace_vkallocator);

    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = block_sums;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = dims;
        constants[1].i = w;
        constants[2].i = h;
        constants[3].i = d;
        constants[4].i = channels;
        constants[5].i = bottom_top_blob.cstep;
        constants[6].i = scan_size;
        constants[7].i = num_blocks;
        constants[8].i = scan_stride;
        constants[9].i = 0;
        constants[10].i = 0;
        constants[11].i = positive_axis;

        VkMat dispatcher;
        dispatcher.w = num_blocks * BLOCK_SIZE;
        dispatcher.h = num_bases;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_blockscan, bindings, constants, dispatcher);
    }

    if (num_blocks > 1)
    {
        VkMat block_sums_scanned;
        block_sums_scanned.create(num_blocks * num_bases, elemsize, elempack, opt.workspace_vkallocator);

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = block_sums;
            bindings[1] = block_sums_scanned;

            std::vector<vk_constant_type> constants(12);
            constants[0].i = dims;
            constants[1].i = w;
            constants[2].i = h;
            constants[3].i = d;
            constants[4].i = channels;
            constants[5].i = bottom_top_blob.cstep;
            constants[6].i = num_blocks;
            constants[7].i = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
            constants[8].i = 1;
            constants[9].i = num_blocks;
            constants[10].i = 0;
            constants[11].i = 0;

            VkMat dispatcher;
            dispatcher.w = ((num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            dispatcher.h = num_bases;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_blocksums_scan, bindings, constants, dispatcher);
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_top_blob;
            bindings[1] = block_sums_scanned;

            std::vector<vk_constant_type> constants(12);
            constants[0].i = dims;
            constants[1].i = w;
            constants[2].i = h;
            constants[3].i = d;
            constants[4].i = channels;
            constants[5].i = bottom_top_blob.cstep;
            constants[6].i = scan_size;
            constants[7].i = num_blocks;
            constants[8].i = scan_stride;
            constants[9].i = 0;
            constants[10].i = 0;
            constants[11].i = positive_axis;

            VkMat dispatcher;
            dispatcher.w = num_blocks * BLOCK_SIZE;
            dispatcher.h = num_bases;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_addoffset, bindings, constants, dispatcher);
        }
    }

    return 0;
}

} // namespace ncnn
