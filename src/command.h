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
    Command(VulkanDevice* vkdev);
    ~Command();

    int begin();

    void record_upload(const VkMat& m);

    void record_upload_barrier(const VkMat& m);

    void record_download(const VkMat& m);

    void record_download_barrier(const VkMat& m);

    void record_clone(const VkMat& src, const VkMat& dst);

    void record_layer(const Layer* layer, const int* constants, int count);

    void record_dispatch(uint32_t* group_count_xyz);

    void record_compute_barrier(const VkMat& m);

    void record_compute_barrier();

    int end();

    int submit();

    int wait();

protected:
    int create_command_pool();
    int create_command_buffer();

public:
    VulkanDevice* vkdev;

    VkDevice device;
    VkQueue queue;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkFence fence;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
