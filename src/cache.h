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

#ifndef NCNN_CACHE_H
#define NCNN_CACHE_H

#include "platform.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

namespace ncnn {

#if NCNN_VULKAN

class VulkanDevice;

class VkCache
{
public:
    VkCache(const VulkanDevice* _vkdev);

    ~VkCache();

    int init();

    int init_cache_data(const std::vector<unsigned char>& cache_data);

    std::vector<unsigned char> get_cache_data() const;

public:
    const VulkanDevice* vkdev;

    VkPipelineCache pipeline_cache;
};

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_CACHE_H
