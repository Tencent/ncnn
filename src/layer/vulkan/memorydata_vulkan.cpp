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
#include <algorithm>
#include "layer_shader_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(MemoryData_vulkan)

MemoryData_vulkan::MemoryData_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;
}

int MemoryData_vulkan::upload_model(VkTransfer& /*cmd*/, const Option& opt)
{
    // VkTransfer will flatten weight data
    // so we use VkCompute for uploading
    VkCompute cmd2(vkdev);

    if (opt.use_image_storage)
    {
        cmd2.record_upload(data, data_gpu_image, opt);
    }
    else
    {
        cmd2.record_upload(data, data_gpu, opt);
    }

    cmd2.submit_and_wait();

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
