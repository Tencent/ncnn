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

#include "cache.h"

#include "gpu.h"

namespace ncnn {

#if NCNN_VULKAN
VkCache::VkCache(const VulkanDevice* _vkdev)
{
    vkdev = _vkdev;

    pipeline_cache = 0;
}

VkCache::~VkCache()
{
    if (pipeline_cache)
    {
        vkDestroyPipelineCache(vkdev->vkdevice(), pipeline_cache, 0);
    }
}

int VkCache::init()
{
    std::vector<unsigned char> empty_cache_data;

    return init_cache_data(empty_cache_data);
}

int VkCache::init_cache_data(const std::vector<unsigned char>& cache_data)
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo;
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipelineCacheCreateInfo.pNext = 0;
    pipelineCacheCreateInfo.flags = 0;// TODO use VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT_EXT for unlocked version
    pipelineCacheCreateInfo.initialDataSize = cache_data.size();
    pipelineCacheCreateInfo.pInitialData = cache_data.data();

    VkResult ret = vkCreatePipelineCache(vkdev->vkdevice(), &pipelineCacheCreateInfo, 0, &pipeline_cache);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreatePipelineCache failed %d", ret);
        return -1;
    }

    return 0;
}

std::vector<unsigned char> VkCache::get_cache_data() const
{
    VkResult ret;

    size_t size = 0;
    std::vector<unsigned char> cache_data;
    ret = vkGetPipelineCacheData(vkdev->vkdevice(), pipeline_cache, &size, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetPipelineCacheData failed %d", ret);
        return std::vector<unsigned char>();
    }

    cache_data.resize(size);
    ret = vkGetPipelineCacheData(vkdev->vkdevice(), pipeline_cache, &size, cache_data.data());
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetPipelineCacheData failed %d", ret);
        return std::vector<unsigned char>();
    }

    return cache_data;
}
#endif // NCNN_VULKAN

} // namespace ncnn
