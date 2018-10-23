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

class UploadEvent
{
public:
    VkEvent event;
    Mat src;
    VkMat dst;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
};

class DownloadEvent
{
public:
    VkEvent event;
    VkMat src;
    Mat dst;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
};

class Layer;
class LayerEvent
{
public:
    VkEvent event;
    const Layer* layer;
};

class Command
{
public:
    Command(VulkanDevice* vkdev, VkAllocator* staging_allocator);
    ~Command();

    int begin();

    UploadEvent record_upload(const Mat& src, const VkMat& dst);
    std::vector<UploadEvent> record_upload(const std::vector<Mat>& srcs, const std::vector<VkMat>& dsts);

    void wait_upload(UploadEvent& event);
    void wait_upload(std::vector<UploadEvent>& events);

    DownloadEvent record_download(const VkMat& src, const Mat& dst);
    std::vector<DownloadEvent> record_download(const std::vector<VkMat>& srcs, const std::vector<Mat>& dsts);

    void wait_download(DownloadEvent& event);
    void wait_download(std::vector<DownloadEvent>& events);

    LayerEvent record_layer(const Layer* layer, uint32_t group_count_xyz[3]);

    void wait_layer(LayerEvent& event);

    int end();

    int submit();

public:
    int copy_mat_to_staging(const Mat& src, VkDeviceMemory staging_memory);
    int copy_staging_to_mat(VkDeviceMemory staging_memory, Mat& dst);

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
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
