// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pipelinecache.h"

#if NCNN_WEBGPU

#include "option.h"
#include "pipeline.h"

namespace ncnn {

int create_webgpu_pipeline_bundle(const VulkanDevice* vkdev, int shader_type_index, const Option& opt,
                                  const std::vector<vk_specialization_type>& specializations,
                                  uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                  WebGpuPipelineBundle* bundle);

class PipelineCachePrivate
{
};

PipelineCache::PipelineCache(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new PipelineCachePrivate)
{
}

PipelineCache::~PipelineCache()
{
    delete d;
}

void PipelineCache::clear()
{
}

size_t PipelineCache::size() const
{
    return 0;
}

int PipelineCache::save_cache(std::vector<unsigned char>& data) const
{
    data.clear();
    return -1;
}

int PipelineCache::load_cache(const unsigned char*, size_t) const
{
    return -1;
}

int PipelineCache::load_cache(const std::vector<unsigned char>&) const
{
    return -1;
}

int PipelineCache::get_pipeline(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
                                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                WebGpuPipelineBundle* bundle) const
{
    return create_webgpu_pipeline_bundle(vkdev, shader_type_index, opt, specializations,
                                         local_size_x, local_size_y, local_size_z, bundle);
}

#if NCNN_STDIO
int PipelineCache::save_cache(FILE*) const
{
    return -1;
}

int PipelineCache::load_cache(FILE*) const
{
    return -1;
}

int PipelineCache::save_cache(const char*) const
{
    return -1;
}

int PipelineCache::load_cache(const char*) const
{
    return -1;
}

#if _WIN32
int PipelineCache::save_cache(const wchar_t*) const
{
    return -1;
}

int PipelineCache::load_cache(const wchar_t*) const
{
    return -1;
}
#endif // _WIN32
#endif // NCNN_STDIO

} // namespace ncnn

#endif // NCNN_WEBGPU
