// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_COMMAND_H
#define NCNN_COMMAND_H

#include "platform.h"

#if NCNN_VULKAN

#include <vulkan/vulkan.h>
#include "mat.h"

namespace ncnn {

class Layer;
class Command
{
public:
    Command(VulkanDevice* vkdev, VkAllocator* staging_allocator);
    ~Command();

    int begin();

    // 0 = undefined to transfer-dst-optimal
    // 1 = transfer-dst-optimal to general
    // 2 = undefined to general
    // 3 = general to transfer-src-optimal
    void record_imagelayout_barrier(VkMat& image, int type);

    void record_upload(const Mat& src, VkMat& dst);

    void record_download(VkMat& src, const Mat& dst);

    void record_layer(const Layer* layer, uint32_t group_count_xyz[3]);

    void record_compute_barrier();

    int end();

    int submit();

    int wait();

protected:
    int create_command_pool();
    int create_command_buffer();

public:
    VulkanDevice* vkdev;
    VkAllocator* staging_allocator;

    VkDevice device;
    VkQueue queue;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkFence fence;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
