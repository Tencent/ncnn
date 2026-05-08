// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "memorydata_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

MemoryData_vulkan::MemoryData_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
}

int MemoryData_vulkan::create_pipeline(const Option& /*opt*/)
{
    // const Mat& out_shape = top_shapes.empty() ? data.shape() : top_shapes[0];

    return 0;
}

int MemoryData_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    const Mat& shape = data.shape();

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

    Mat data_packed;
    convert_packing(data, data_packed, elempack, opt);

    cmd.record_upload(data_packed, data_gpu, opt, /*bool flatten*/ false);

    if (opt.lightmode)
    {
        data.release();
    }

    return 0;
}

int MemoryData_vulkan::forward(const std::vector<VkMat>& /*bottom_blobs*/, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    VkMat& top_blob = top_blobs[0];

    cmd.record_clone(data_gpu, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
