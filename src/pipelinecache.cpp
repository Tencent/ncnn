// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pipelinecache.h"

#include "gpu.h"

namespace ncnn {

#if NCNN_VULKAN
// https://en.wikipedia.org/wiki/MurmurHash
static uint32_t murmur3_32(const uint32_t* data, int size)
{
    uint32_t h = 0;

    for (int i = 0; i < size; i++)
    {
        uint32_t k = *data++;

        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> (32 - 15));
        k *= 0x1b873593;

        h ^= k;
        h = (h << 13) | (h >> (32 - 13));
        h = (h * 5) + 0xe6546b64;
    }

    h ^= size * 4;

    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash
static uint32_t fnv1a_32(const uint8_t* data, int size)
{
    uint32_t h = 0x811c9dc5;

    for (int i = 0; i < size; i++)
    {
        h ^= (uint32_t)*data++;
        h *= 0x01000193;
    }

    return h;
}

class PipelineCachePrivate
{
public:
    // digest -> artifact
    struct pipeline_cache_digest
    {
        pipeline_cache_digest(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations,
                              uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size);
        pipeline_cache_digest(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
                              uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size);

        bool operator==(const pipeline_cache_digest& rhs) const
        {
            return d0 == rhs.d0 && d1 == rhs.d1 && d2 == rhs.d2 && d3 == rhs.d3;
        }

        bool operator!=(const pipeline_cache_digest& rhs) const
        {
            return d0 != rhs.d0 || d1 != rhs.d1 || d2 != rhs.d2 || d3 != rhs.d3;
        }

        union
        {
            struct
            {
                union
                {
                    uint32_t spv_data_murmur3;
                    int shader_type_index;
                };
                uint32_t opt_bits;
                uint32_t local_size_x;
                uint32_t local_size_y;
                uint32_t local_size_z;
                uint32_t subgroup_size;
                uint32_t specializations_murmur3;
                uint32_t specializations_fnv1a;
            };

            struct
            {
                uint64_t d0;
                uint64_t d1;
                uint64_t d2;
                uint64_t d3;
            };
        };
    };

    struct pipeline_cache_artifact
    {
        VkShaderModule shader_module;
        VkDescriptorSetLayout descriptorset_layout;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        VkDescriptorUpdateTemplateKHR descriptor_update_template;
        ShaderInfo shader_info; // TODO use pointer ?
    };

    struct spv_param
    {
        union
        {
            struct
            {
                int32_t shader_type_index;
                uint32_t opt_bits;
            };
            uint64_t d0;
        };
    };

    struct pipeline_cache_header
    {
        uint32_t magic = 0x5a545546;
        uint32_t vendorID;          // VkPhysicalDeviceProperties::vendorID
        uint32_t deviceID;          // VkPhysicalDeviceProperties::deviceID
        uint32_t driverVersion;     // VkPhysicalDeviceProperties::driverVersion
        uint8_t uuid[VK_UUID_SIZE]; // VkPhysicalDeviceProperties::pipelineCacheUUID

        uint32_t spv_size; // size of spirv data
        uint32_t pipeline_cache_size;
    };

    mutable std::vector<pipeline_cache_digest> cache_digests;
    mutable std::vector<pipeline_cache_artifact> cache_artifacts;

    VkPipelineCache vk_pipeline_cache;
    mutable std::vector<std::pair<spv_param, std::vector<uint32_t> > > cache_spirv_module; // digest(index,opt) -> spirv data

    mutable Mutex cache_lock;
};

PipelineCachePrivate::pipeline_cache_digest::pipeline_cache_digest(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations,
        uint32_t _local_size_x, uint32_t _local_size_y, uint32_t _local_size_z, uint32_t _subgroup_size)
{
    spv_data_murmur3 = murmur3_32(spv_data, spv_data_size / 4);

    opt_bits = 0;

    local_size_x = _local_size_x;
    local_size_y = _local_size_y;
    local_size_z = _local_size_z;
    subgroup_size = _subgroup_size;

    // encode specializations
    const int specialization_count = specializations.size();
    specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
    specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(vk_specialization_type));
}

PipelineCachePrivate::pipeline_cache_digest::pipeline_cache_digest(int _shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
        uint32_t _local_size_x, uint32_t _local_size_y, uint32_t _local_size_z, uint32_t _subgroup_size)
{
    shader_type_index = _shader_type_index;

    // encode opt
    opt_bits = 0 << 7
               | opt.use_fp16_packed << 6
               | opt.use_fp16_storage << 5
               | opt.use_fp16_arithmetic << 4
               | opt.use_int8_storage << 3
               | opt.use_int8_arithmetic << 2;

    local_size_x = _local_size_x;
    local_size_y = _local_size_y;
    local_size_z = _local_size_z;
    subgroup_size = _subgroup_size;

    // encode specializations
    const int specialization_count = specializations.size();
    specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
    specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(vk_specialization_type));
}

PipelineCache::PipelineCache(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new PipelineCachePrivate)
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo{};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipelineCacheCreateInfo.initialDataSize = 0; // zeros for empty cache
    pipelineCacheCreateInfo.pInitialData = nullptr;

    int ret = 0;
    ret = _vkdev->create_pipeline_cache(&pipelineCacheCreateInfo, 0, &d->vk_pipeline_cache);
    if (ret != 0)
    {
        NCNN_LOGE("create_pipeline_cache failed %d", ret);
        d->vk_pipeline_cache = 0;
    }
}

PipelineCache::~PipelineCache()
{
    clear();

    delete d;
}

PipelineCache::PipelineCache(const PipelineCache&)
    : d(0)
{
}

PipelineCache& PipelineCache::operator=(const PipelineCache&)
{
    return *this;
}

void PipelineCache::clear()
{
    MutexLockGuard lock(d->cache_lock);

    for (size_t i = 0; i < d->cache_artifacts.size(); i++)
    {
        const PipelineCachePrivate::pipeline_cache_artifact& cc = d->cache_artifacts[i];

        if (vkdev->info.support_VK_KHR_descriptor_update_template())
        {
            if (cc.descriptor_update_template)
            {
                vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->vkdevice(), cc.descriptor_update_template, 0);
            }
        }

        if (cc.pipeline)
        {
            vkDestroyPipeline(vkdev->vkdevice(), cc.pipeline, 0);
        }

        if (cc.pipeline_layout)
        {
            vkDestroyPipelineLayout(vkdev->vkdevice(), cc.pipeline_layout, 0);
        }

        if (cc.descriptorset_layout)
        {
            vkDestroyDescriptorSetLayout(vkdev->vkdevice(), cc.descriptorset_layout, 0);
        }

        if (cc.shader_module)
        {
            vkDestroyShaderModule(vkdev->vkdevice(), cc.shader_module, 0);
        }
    }

    d->cache_digests.clear();
    d->cache_artifacts.clear();
}

int PipelineCache::get_pipeline(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations,
                                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size,
                                VkShaderModule* _shader_module,
                                VkDescriptorSetLayout* descriptorset_layout,
                                VkPipelineLayout* pipeline_layout,
                                VkPipeline* pipeline,
                                VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                                ShaderInfo& shader_info) const
{
    MutexLockGuard lock(d->cache_lock);

    PipelineCachePrivate::pipeline_cache_digest key(spv_data, spv_data_size, specializations, local_size_x, local_size_y, local_size_z, subgroup_size);

    if (!vkdev->info.bug_corrupted_online_pipeline_cache())
    {
        // find cache
        for (size_t i = 0; i < d->cache_digests.size(); i++)
        {
            if (d->cache_digests[i] != key)
                continue;

            // hit cache
            const PipelineCachePrivate::pipeline_cache_artifact& cc = d->cache_artifacts[i];

            *_shader_module = cc.shader_module;
            *descriptorset_layout = cc.descriptorset_layout;
            *pipeline_layout = cc.pipeline_layout;
            *pipeline = cc.pipeline;
            *descriptor_update_template = cc.descriptor_update_template;
            shader_info = cc.shader_info;

            // NCNN_LOGE("get_pipeline hit %d", last_digest_index);

            return 0;
        }
    }

    int ret = 0;

    ret = resolve_shader_info(spv_data, spv_data_size, shader_info);
    if (ret != 0)
    {
        NCNN_LOGE("resolve_shader_info failed %d", ret);
        return -1;
    }

    VkShaderModule shader_module = vkdev->compile_shader_module(spv_data, spv_data_size, local_size_x, local_size_y, local_size_z);
    if (!shader_module)
    {
        NCNN_LOGE("create_shader_module failed");
        return -1;
    }

    ret = new_pipeline(shader_module, shader_info, specializations, subgroup_size, descriptorset_layout, pipeline_layout, pipeline, descriptor_update_template);
    if (ret != 0)
    {
        NCNN_LOGE("new_pipeline failed");
        vkDestroyShaderModule(vkdev->vkdevice(), shader_module, 0);
        return -1;
    }

    *_shader_module = shader_module;

    // save to cache
    {
        PipelineCachePrivate::pipeline_cache_artifact cc;

        cc.shader_module = *_shader_module;
        cc.descriptorset_layout = *descriptorset_layout;
        cc.pipeline_layout = *pipeline_layout;
        cc.pipeline = *pipeline;
        cc.descriptor_update_template = *descriptor_update_template;
        cc.shader_info = shader_info;

        d->cache_digests.push_back(key);
        d->cache_artifacts.push_back(cc);
    }

    // NCNN_LOGE("new_pipeline %d", last_digest_index);

    return 0;
}

int PipelineCache::get_pipeline(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
                                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size,
                                VkShaderModule* _shader_module,
                                VkDescriptorSetLayout* descriptorset_layout,
                                VkPipelineLayout* pipeline_layout,
                                VkPipeline* pipeline,
                                VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                                ShaderInfo& shader_info) const
{
    MutexLockGuard lock(d->cache_lock);

    PipelineCachePrivate::pipeline_cache_digest key(shader_type_index, opt, specializations, local_size_x, local_size_y, local_size_z, subgroup_size);

    if (!vkdev->info.bug_corrupted_online_pipeline_cache())
    {
        // find cache
        for (size_t i = 0; i < d->cache_digests.size(); i++)
        {
            if (d->cache_digests[i] != key)
                continue;

            // hit cache
            const PipelineCachePrivate::pipeline_cache_artifact& cc = d->cache_artifacts[i];

            *_shader_module = cc.shader_module;
            *descriptorset_layout = cc.descriptorset_layout;
            *pipeline_layout = cc.pipeline_layout;
            *pipeline = cc.pipeline;
            *descriptor_update_template = cc.descriptor_update_template;
            shader_info = cc.shader_info;

            // NCNN_LOGE("get_pipeline hit %d", last_digest_index);

            return 0;
        }
    }

    int ret = 0;

    // create new pipeline
    VkShaderModule shader_module = 0;
    ret = create_shader_module(shader_type_index, opt, local_size_x, local_size_y, local_size_z, &shader_module, shader_info);
    if (ret != 0)
    {
        NCNN_LOGE("create_shader_module failed");
        return -1;
    }

    ret = new_pipeline(shader_module, shader_info, specializations, subgroup_size, descriptorset_layout, pipeline_layout, pipeline, descriptor_update_template);
    if (ret != 0)
    {
        NCNN_LOGE("new_pipeline failed");
        vkDestroyShaderModule(vkdev->vkdevice(), shader_module, 0);
        return -1;
    }

    *_shader_module = shader_module;

    // save to cache
    {
        PipelineCachePrivate::pipeline_cache_artifact cc;

        cc.shader_module = *_shader_module;
        cc.descriptorset_layout = *descriptorset_layout;
        cc.pipeline_layout = *pipeline_layout;
        cc.pipeline = *pipeline;
        cc.descriptor_update_template = *descriptor_update_template;
        cc.shader_info = shader_info;

        d->cache_digests.push_back(key);
        d->cache_artifacts.push_back(cc);
    }

    // NCNN_LOGE("new_pipeline %d", last_digest_index);

    return 0;
}

int PipelineCache::save_cache(std::vector<unsigned char>& buf) const
{
    if (!vkdev)
    {
        NCNN_LOGE("vkdev is null");
        return -1;
    }
    MutexLockGuard lock(d->cache_lock);

    PipelineCachePrivate::pipeline_cache_header header;

    // Platform information
    header.vendorID = vkdev->info.vendor_id();
    header.deviceID = vkdev->info.device_id();
    header.driverVersion = vkdev->info.driver_version();
    memcpy(header.uuid, vkdev->info.pipeline_cache_uuid(), VK_UUID_SIZE);

    header.spv_size = d->cache_spirv_module.size();

    size_t buf_size = 0;
    if (vkGetPipelineCacheData(vkdev->vkdevice(), d->vk_pipeline_cache, &buf_size, nullptr) != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetPipelineCacheData failed");
        return -1;
    }
    header.pipeline_cache_size = (uint32_t)buf_size;

    std::vector<unsigned char> pipe_data(header.pipeline_cache_size);
    if (vkGetPipelineCacheData(vkdev->vkdevice(), d->vk_pipeline_cache, &buf_size, pipe_data.data()) != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetPipelineCacheData failed");
        return -1;
    }

    buf.resize(sizeof(header));
    memcpy(buf.data(), &header, sizeof(header));

    // spv_digest and spv_data
    for (size_t i = 0; i < d->cache_spirv_module.size(); i++)
    {
        const PipelineCachePrivate::spv_param& sd = d->cache_spirv_module[i].first;
        const std::vector<uint32_t>& spv_data = d->cache_spirv_module[i].second;
        uint32_t size = (uint32_t)spv_data.size();

        size_t current_buf_size = buf.size();
        buf.resize(current_buf_size + sizeof(sd) + sizeof(size) + spv_data.size() * sizeof(uint32_t));

        memcpy(buf.data() + current_buf_size, &sd, sizeof(sd));
        current_buf_size += sizeof(sd);
        memcpy(buf.data() + current_buf_size, &size, sizeof(size));
        current_buf_size += sizeof(size);

        memcpy(buf.data() + current_buf_size, spv_data.data(), spv_data.size() * sizeof(uint32_t));
    }

    buf.insert(buf.end(), pipe_data.begin(), pipe_data.end());
    return 0;
}

int PipelineCache::load_cache(const std::vector<unsigned char>& buf) const
{
    if (!vkdev)
    {
        NCNN_LOGE("vkdev is null");
        return -1;
    }
    MutexLockGuard lock(d->cache_lock);

    // Corrected struct name to pipeline_cache_header (lowercase h)
    if (buf.size() < sizeof(PipelineCachePrivate::pipeline_cache_header))
    {
        NCNN_LOGE("Invalid cache buffer size: too small for header");
        return -1;
    }

    PipelineCachePrivate::pipeline_cache_header header;
    memcpy(&header, buf.data(), sizeof(header));

    // Validate magic number
    if (header.magic != 0x5a545546)
    {
        NCNN_LOGE("Invalid cache magic number");
        return -1;
    }

    // Validate platform information for compatibility
    if (header.vendorID != vkdev->info.vendor_id() || header.deviceID != vkdev->info.device_id() || header.driverVersion != vkdev->info.driver_version() || memcmp(header.uuid, vkdev->info.pipeline_cache_uuid(), VK_UUID_SIZE) != 0)
    {
        NCNN_LOGE("Cache platform mismatch, might be incompatible.");
        return -1;
    }

    size_t current_offset = sizeof(header);

    // Load SPIR-V data and associated spv_param
    d->cache_spirv_module.reserve(header.spv_size);

    for (uint32_t i = 0; i < header.spv_size; ++i)
    {
        if (current_offset + sizeof(PipelineCachePrivate::spv_param) + sizeof(uint32_t) > buf.size())
        {
            NCNN_LOGE("Invalid cache buffer size: incomplete spv_param or size for entry %u", i);
            return -1;
        }

        PipelineCachePrivate::spv_param sd;
        memcpy(&sd, buf.data() + current_offset, sizeof(sd));
        current_offset += sizeof(sd);

        uint32_t spv_vec_size_uint32; // Size in uint32_t units
        memcpy(&spv_vec_size_uint32, buf.data() + current_offset, sizeof(spv_vec_size_uint32));
        current_offset += sizeof(spv_vec_size_uint32);

        size_t spv_data_byte_size = spv_vec_size_uint32 * sizeof(uint32_t);

        if (current_offset + spv_data_byte_size > buf.size())
        {
            NCNN_LOGE("Invalid cache buffer size: incomplete spv_data for entry %u", i);
            return -1;
        }

        std::vector<uint32_t> spirv_data(spv_vec_size_uint32);
        memcpy(spirv_data.data(), buf.data() + current_offset, spv_data_byte_size);
        current_offset += spv_data_byte_size;

        d->cache_spirv_module.push_back({sd, spirv_data});
    }

    // Load Vulkan Pipeline Cache Data
    if (current_offset + header.pipeline_cache_size > buf.size())
    {
        NCNN_LOGE("Invalid cache buffer size: incomplete pipeline cache data");
        return -1;
    }

    if (d->vk_pipeline_cache)
    {
        vkDestroyPipelineCache(vkdev->vkdevice(), d->vk_pipeline_cache, 0);
        d->vk_pipeline_cache = 0;
    }

    VkPipelineCacheCreateInfo pipelineCacheCreateInfo{};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipelineCacheCreateInfo.initialDataSize = header.pipeline_cache_size;
    pipelineCacheCreateInfo.pInitialData = buf.data() + current_offset;

    int ret = vkdev->create_pipeline_cache(&pipelineCacheCreateInfo, 0, &d->vk_pipeline_cache);
    if (ret != 0)
    {
        NCNN_LOGE("create_pipeline_cache with initial data failed %d", ret);
        d->vk_pipeline_cache = 0;
        return -1;
    }

    return 0;
}

int PipelineCache::save_cache(FILE* fp) const
{
    if (!fp)
    {
        NCNN_LOGE("Invalid FILE pointer for saving cache.");
        return -1;
    }

    std::vector<unsigned char> buf;
    int ret = save_cache(buf);
    if (ret != 0)
    {
        NCNN_LOGE("Failed to get cache data into buffer for saving to file.");
        return ret;
    }

    if (fwrite(buf.data(), 1, buf.size(), fp) != buf.size())
    {
        NCNN_LOGE("Failed to write cache data to file.");
        return -1;
    }

    return 0;
}

int PipelineCache::load_cache(FILE* fp) const
{
    if (!fp)
    {
        NCNN_LOGE("Invalid FILE pointer for loading cache.");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size < 0)
    {
        NCNN_LOGE("Failed to determine file size for loading cache.");
        return -1;
    }

    std::vector<unsigned char> buf(file_size);
    if (fread(buf.data(), 1, file_size, fp) != (size_t)file_size)
    {
        NCNN_LOGE("Failed to read cache data from file.");
        return -1;
    }

    return load_cache(buf);
}

int PipelineCache::save_cache(const char* filename) const
{
    if (!filename)
    {
        NCNN_LOGE("Invalid filename for saving cache.");
        return -1;
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp)
    {
        NCNN_LOGE("Failed to open file %s for writing cache.", filename);
        return -1;
    }

    int ret = save_cache(fp);
    fclose(fp);

    return ret;
}

int PipelineCache::load_cache(const char* filename) const
{
    if (!filename)
    {
        NCNN_LOGE("Invalid filename for loading cache.");
        return -1;
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        NCNN_LOGE("Failed to open file %s for reading cache.", filename);
        return -1;
    }

    int ret = load_cache(fp);
    fclose(fp);

    return ret;
}

int PipelineCache::create_shader_module(int shader_type_index, const Option& opt,
                                        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                        VkShaderModule* _shader_module, ShaderInfo& si) const
{
    uint32_t opt_bits = 0 << 7
                        | opt.use_fp16_packed << 6
                        | opt.use_fp16_storage << 5
                        | opt.use_fp16_arithmetic << 4
                        | opt.use_int8_storage << 3
                        | opt.use_int8_arithmetic << 2;

    std::vector<uint32_t> spirv;

    for (int i = 0; i < d->cache_spirv_module.size(); i++)
    {
        if (d->cache_spirv_module[i].first.d0 == PipelineCachePrivate::spv_param({shader_type_index, opt_bits}).d0) // hit cache
        {
            spirv = d->cache_spirv_module[i].second;
            goto hit_cache;
        }
    }

    int retc = compile_spirv_module(shader_type_index, opt, spirv);
    if (retc != 0)
    {
        NCNN_LOGE("compile_spirv_module failed %d", retc);
        return -1;
    }
    d->cache_spirv_module.push_back({{shader_type_index, opt_bits}, spirv});
hit_cache:
    const uint32_t* spv_data = spirv.data();
    size_t spv_data_size = spirv.size() * 4;

    int ret = resolve_shader_info(spv_data, spv_data_size, si);
    if (ret != 0)
    {
        NCNN_LOGE("resolve_shader_info failed %d", ret);
        return -1;
    }

    VkShaderModule shader_module = vkdev->compile_shader_module(spv_data, spv_data_size, local_size_x, local_size_y, local_size_z);

    if (!shader_module)
    {
        NCNN_LOGE("create_shader_module failed");
        return -1;
    }

    *_shader_module = shader_module;

    return 0;
}

int PipelineCache::new_pipeline(VkShaderModule shader_module, const ShaderInfo& shader_info,
                                const std::vector<vk_specialization_type>& specializations, uint32_t subgroup_size,
                                VkDescriptorSetLayout* _descriptorset_layout,
                                VkPipelineLayout* _pipeline_layout,
                                VkPipeline* _pipeline,
                                VkDescriptorUpdateTemplateKHR* _descriptor_update_template) const
{
    int ret = 0;

    VkDescriptorSetLayout descriptorset_layout = 0;
    VkPipelineLayout pipeline_layout = 0;
    VkPipeline pipeline = 0;
    VkDescriptorUpdateTemplateKHR descriptor_update_template = 0;

    // create new pipeline
    if ((int)specializations.size() != shader_info.specialization_count)
    {
        NCNN_LOGE("pipeline specialization count mismatch, expect %d but got %d", shader_info.specialization_count, (int)specializations.size());
        goto ERROR_PipelineCache;
    }

    ret = vkdev->create_descriptorset_layout(shader_info.binding_count, shader_info.binding_types, &descriptorset_layout);
    if (ret != 0)
        goto ERROR_PipelineCache;

    ret = vkdev->create_pipeline_layout(shader_info.push_constant_count, descriptorset_layout, &pipeline_layout);
    if (ret != 0)
        goto ERROR_PipelineCache;

    ret = vkdev->create_pipeline(shader_module, pipeline_layout, specializations, subgroup_size, &pipeline, d->vk_pipeline_cache);
    if (ret != 0)
        goto ERROR_PipelineCache;

    if (vkdev->info.support_VK_KHR_descriptor_update_template())
    {
        ret = vkdev->create_descriptor_update_template(shader_info.binding_count, shader_info.binding_types, descriptorset_layout, pipeline_layout, &descriptor_update_template);
        if (ret != 0)
            goto ERROR_PipelineCache;
    }

    *_descriptorset_layout = descriptorset_layout;
    *_pipeline_layout = pipeline_layout;
    *_pipeline = pipeline;
    *_descriptor_update_template = descriptor_update_template;

    return 0;

ERROR_PipelineCache:

    if (vkdev->info.support_VK_KHR_descriptor_update_template())
    {
        if (descriptor_update_template)
        {
            vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->vkdevice(), descriptor_update_template, 0);
        }
    }

    if (pipeline)
    {
        vkDestroyPipeline(vkdev->vkdevice(), pipeline, 0);
    }

    if (pipeline_layout)
    {
        vkDestroyPipelineLayout(vkdev->vkdevice(), pipeline_layout, 0);
    }

    if (descriptorset_layout)
    {
        vkDestroyDescriptorSetLayout(vkdev->vkdevice(), descriptorset_layout, 0);
    }

    return -1;
}

#endif // NCNN_VULKAN

} // namespace ncnn
