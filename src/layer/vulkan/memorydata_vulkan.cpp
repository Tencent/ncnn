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

int MemoryData_vulkan::create_pipeline(const Option& opt)
{
    const Mat& out_shape = top_shapes.empty() ? data.shape() : top_shapes[0];

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = out_shape.c % 4 == 0 ? 4 : 1;

    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        out_elemsize = out_elempack * 2u;
    }
    else
    {
        out_elemsize = out_elempack * 4u;
    }

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3 || out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

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
