// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "memorydata_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

MemoryData_vulkan::MemoryData_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;
}

int MemoryData_vulkan::create_pipeline(const Option& opt)
{
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        out_elemsize = out_elempack * 4u;
    }

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
    }

    return 0;
}

int MemoryData_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(data, data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(data, data_gpu, opt, /*bool flatten*/ false);
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

int MemoryData_vulkan::forward(const std::vector<VkImageMat>& /*bottom_blobs*/, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    VkImageMat& top_blob = top_blobs[0];

    cmd.record_clone(data_gpu_image, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
