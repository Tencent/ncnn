// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pipelinecache.h"

#include "gpu.h"

#include <limits.h>
#include <string.h>

#if NCNN_STDIO
#include <stdio.h>
#if defined(_WIN32)
#include <process.h>
#include <wchar.h>
#else
#include <unistd.h>
#endif
#endif

namespace ncnn {

#if NCNN_VULKAN
#define NCNN_PIPELINE_CACHE_FILE_MAGIC   0x5a545546
#define NCNN_PIPELINE_CACHE_FILE_VERSION 1
#define NCNN_PIPELINE_CACHE_FILE_ENDIAN  0x12345678
#define NCNN_PIPELINE_CACHE_NCNN_VERSION NCNN_VERSION_NUMBER

uint64_t get_shader_source_hash(int shader_type_index);

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
    PipelineCachePrivate()
        : vk_pipeline_cache(0)
    {
    }

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

    struct cache_file_header
    {
        uint32_t magic;
        uint32_t version;
        uint32_t header_size;
        uint32_t ncnn_version;
        uint32_t endian;
        uint32_t pointer_size;

        uint32_t vendor_id;
        uint32_t device_id;
        uint32_t api_version;
        uint32_t driver_version;
        uint32_t driver_id;
        uint32_t device_name_hash;
        uint32_t driver_name_hash;
        uint8_t pipeline_cache_uuid[VK_UUID_SIZE];

        uint32_t reserved0;

        uint32_t spirv_entry_count;
        uint32_t spirv_cache_size;
        uint32_t spirv_cache_hash;

        uint32_t pipeline_cache_size;
        uint32_t pipeline_cache_hash;

        uint32_t reserved[8];
    };

    struct spirv_cache_entry_header
    {
        int32_t shader_type_index;
        uint32_t opt_bits;
        uint64_t shader_source_hash;
        uint32_t spv_size;
        uint32_t spv_hash_fnv1a;
        uint32_t spv_hash_murmur3;
        uint32_t reserved[1];
    };

    struct spirv_cache_entry
    {
        int shader_type_index;
        uint32_t opt_bits;
        uint64_t shader_source_hash;
        std::vector<uint32_t> spirv;
        ShaderInfo shader_info;
    };

    mutable std::vector<pipeline_cache_digest> cache_digests;
    mutable std::vector<pipeline_cache_artifact> cache_artifacts;
    mutable std::vector<spirv_cache_entry> cache_spirv_entries;
    mutable VkPipelineCache vk_pipeline_cache;
    mutable Mutex cache_lock;
};

static uint32_t hash_string(const char* s)
{
    if (!s)
        return 0;

    return fnv1a_32((const uint8_t*)s, (int)strlen(s));
}

static uint32_t encode_spirv_cache_opt_bits(const Option& opt)
{
    // Bump NCNN_PIPELINE_CACHE_FILE_VERSION if this bit layout changes.
    return 0
           | (uint32_t)opt.use_bf16_packed << 0
           | (uint32_t)opt.use_bf16_storage << 1
           | (uint32_t)opt.use_fp16_packed << 2
           | (uint32_t)opt.use_fp16_storage << 3
           | (uint32_t)opt.use_fp16_arithmetic << 4
           | (uint32_t)opt.use_fp16_uniform << 5
           | (uint32_t)opt.use_int8_packed << 6
           | (uint32_t)opt.use_int8_storage << 7
           | (uint32_t)opt.use_int8_arithmetic << 8
           | (uint32_t)opt.use_int8_uniform << 9
           | (uint32_t)opt.use_subgroup_ops << 10
           | (uint32_t)opt.use_shader_local_memory << 11
           | (uint32_t)opt.use_cooperative_matrix << 12;
}

static bool can_cache_spirv(const VulkanDevice* vkdev, const Option& opt)
{
    int device_index = opt.vulkan_device_index;
    if (device_index < 0 || device_index >= get_gpu_count())
        device_index = get_default_gpu_index();

    return device_index == vkdev->info.device_index();
}

static void append_data(std::vector<unsigned char>& data, const void* ptr, size_t size)
{
    if (size == 0)
        return;

    const unsigned char* p = (const unsigned char*)ptr;
    data.insert(data.end(), p, p + size);
}

static uint32_t read_u32(const unsigned char* data)
{
    uint32_t v;
    memcpy(&v, data, sizeof(v));
    return v;
}

static bool validate_vk_pipeline_cache_data(const unsigned char* data, size_t size, const VulkanDevice* vkdev)
{
    if (size == 0)
        return true;

    // VkPipelineCacheHeaderVersionOne is 32 bytes:
    // headerLength, headerVersion, vendorID, deviceID, pipelineCacheUUID
    if (size < 32)
        return false;

    const uint32_t header_length = read_u32(data);
    const uint32_t header_version = read_u32(data + 4);
    const uint32_t vendor_id = read_u32(data + 8);
    const uint32_t device_id = read_u32(data + 12);

    if (header_length < 32 || header_length > size)
        return false;
    if (header_version != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return false;
    if (vendor_id != vkdev->info.vendor_id())
        return false;
    if (device_id != vkdev->info.device_id())
        return false;
    if (memcmp(data + 16, vkdev->info.pipeline_cache_uuid(), VK_UUID_SIZE) != 0)
        return false;

    return true;
}

static void fill_cache_file_header(PipelineCachePrivate::cache_file_header& header, const VulkanDevice* vkdev)
{
    memset(&header, 0, sizeof(header));

    header.magic = NCNN_PIPELINE_CACHE_FILE_MAGIC;
    header.version = NCNN_PIPELINE_CACHE_FILE_VERSION;
    header.header_size = sizeof(PipelineCachePrivate::cache_file_header);
    header.ncnn_version = NCNN_PIPELINE_CACHE_NCNN_VERSION;
    header.endian = NCNN_PIPELINE_CACHE_FILE_ENDIAN;
    header.pointer_size = sizeof(void*);

    header.vendor_id = vkdev->info.vendor_id();
    header.device_id = vkdev->info.device_id();
    header.api_version = vkdev->info.api_version();
    header.driver_version = vkdev->info.driver_version();
    header.driver_id = vkdev->info.driver_id();
    header.device_name_hash = hash_string(vkdev->info.device_name());
    header.driver_name_hash = hash_string(vkdev->info.driver_name());
    memcpy(header.pipeline_cache_uuid, vkdev->info.pipeline_cache_uuid(), VK_UUID_SIZE);

    header.reserved0 = 0;
}

static bool validate_cache_file_header(const PipelineCachePrivate::cache_file_header& header, const VulkanDevice* vkdev)
{
    if (header.magic != NCNN_PIPELINE_CACHE_FILE_MAGIC)
        return false;
    if (header.version != NCNN_PIPELINE_CACHE_FILE_VERSION)
        return false;
    if (header.header_size != sizeof(PipelineCachePrivate::cache_file_header))
        return false;
    if (header.ncnn_version != NCNN_PIPELINE_CACHE_NCNN_VERSION)
        return false;
    if (header.endian != NCNN_PIPELINE_CACHE_FILE_ENDIAN)
        return false;
    if (header.pointer_size != sizeof(void*))
        return false;

    if (header.vendor_id != vkdev->info.vendor_id())
        return false;
    if (header.device_id != vkdev->info.device_id())
        return false;
    if (header.api_version != vkdev->info.api_version())
        return false;
    if (header.driver_version != vkdev->info.driver_version())
        return false;
    if (header.driver_id != vkdev->info.driver_id())
        return false;
    if (header.device_name_hash != hash_string(vkdev->info.device_name()))
        return false;
    if (header.driver_name_hash != hash_string(vkdev->info.driver_name()))
        return false;
    if (memcmp(header.pipeline_cache_uuid, vkdev->info.pipeline_cache_uuid(), VK_UUID_SIZE) != 0)
        return false;

    return true;
}

static int create_vk_pipeline_cache(const VulkanDevice* vkdev, const void* data, size_t size, VkPipelineCache* pipeline_cache)
{
    *pipeline_cache = 0;

    if (vkdev->info.bug_corrupted_online_pipeline_cache())
        return -1;

    VkPipelineCacheCreateInfo pipelineCacheCreateInfo;
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pipelineCacheCreateInfo.pNext = 0;
    pipelineCacheCreateInfo.flags = 0;
    pipelineCacheCreateInfo.initialDataSize = size;
    pipelineCacheCreateInfo.pInitialData = data;

    VkResult ret = vkCreatePipelineCache(vkdev->vkdevice(), &pipelineCacheCreateInfo, 0, pipeline_cache);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreatePipelineCache failed %d", ret);
        return -1;
    }

    return 0;
}

static int ensure_vk_pipeline_cache(const VulkanDevice* vkdev, PipelineCachePrivate* d)
{
    if (d->vk_pipeline_cache)
        return 0;

    return create_vk_pipeline_cache(vkdev, 0, 0, &d->vk_pipeline_cache);
}

#if NCNN_STDIO
static Mutex g_tmp_path_lock;
static unsigned int g_tmp_path_index = 0;

static int replace_file(const char* tmp_path, const char* path)
{
#if defined(_WIN32)
    return MoveFileExA(tmp_path, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(tmp_path, path);
#endif
}

#if defined(_WIN32)
static int replace_file(const wchar_t* tmp_path, const wchar_t* path)
{
    return MoveFileExW(tmp_path, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}
#endif
#endif // NCNN_STDIO

PipelineCachePrivate::pipeline_cache_digest::pipeline_cache_digest(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations,
        uint32_t _local_size_x, uint32_t _local_size_y, uint32_t _local_size_z, uint32_t _subgroup_size)
{
    spv_data_murmur3 = murmur3_32(spv_data, (int)(spv_data_size / 4));

    opt_bits = 0;

    local_size_x = _local_size_x;
    local_size_y = _local_size_y;
    local_size_z = _local_size_z;
    subgroup_size = _subgroup_size;

    // encode specializations
    const int specialization_count = (int)specializations.size();
    specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
    specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(vk_specialization_type));
}

PipelineCachePrivate::pipeline_cache_digest::pipeline_cache_digest(int _shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
        uint32_t _local_size_x, uint32_t _local_size_y, uint32_t _local_size_z, uint32_t _subgroup_size)
{
    shader_type_index = _shader_type_index;

    // encode opt
    opt_bits = encode_spirv_cache_opt_bits(opt);

    local_size_x = _local_size_x;
    local_size_y = _local_size_y;
    local_size_z = _local_size_z;
    subgroup_size = _subgroup_size;

    // encode specializations
    const int specialization_count = (int)specializations.size();
    specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
    specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(vk_specialization_type));
}

PipelineCache::PipelineCache(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new PipelineCachePrivate)
{
    ensure_vk_pipeline_cache(vkdev, d);
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
    d->cache_spirv_entries.clear();

    if (d->vk_pipeline_cache)
    {
        vkDestroyPipelineCache(vkdev->vkdevice(), d->vk_pipeline_cache, 0);
        d->vk_pipeline_cache = 0;
    }
}

size_t PipelineCache::size() const
{
    MutexLockGuard lock(d->cache_lock);

    return d->cache_artifacts.size();
}

int PipelineCache::save_cache(std::vector<unsigned char>& data) const
{
    MutexLockGuard lock(d->cache_lock);

    std::vector<unsigned char> spirv_cache_data;
    for (size_t i = 0; i < d->cache_spirv_entries.size(); i++)
    {
        const PipelineCachePrivate::spirv_cache_entry& entry = d->cache_spirv_entries[i];
        const size_t spv_size = entry.spirv.size() * sizeof(uint32_t);

        if (spv_size > UINT_MAX || spv_size > INT_MAX)
        {
            NCNN_LOGE("spirv cache data too large");
            return -1;
        }

        PipelineCachePrivate::spirv_cache_entry_header entry_header;
        memset(&entry_header, 0, sizeof(entry_header));
        entry_header.shader_type_index = entry.shader_type_index;
        entry_header.opt_bits = entry.opt_bits;
        entry_header.shader_source_hash = entry.shader_source_hash;
        entry_header.spv_size = (uint32_t)spv_size;
        entry_header.spv_hash_fnv1a = fnv1a_32((const uint8_t*)entry.spirv.data(), (int)spv_size);
        entry_header.spv_hash_murmur3 = murmur3_32(entry.spirv.data(), (int)entry.spirv.size());

        append_data(spirv_cache_data, &entry_header, sizeof(entry_header));
        append_data(spirv_cache_data, entry.spirv.data(), spv_size);
    }

    if (spirv_cache_data.size() > UINT_MAX || spirv_cache_data.size() > INT_MAX)
    {
        NCNN_LOGE("spirv cache chunk too large");
        return -1;
    }

    std::vector<unsigned char> pipeline_cache_data;
    if (d->vk_pipeline_cache)
    {
        size_t pipeline_cache_size = 0;
        VkResult ret = vkGetPipelineCacheData(vkdev->vkdevice(), d->vk_pipeline_cache, &pipeline_cache_size, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPipelineCacheData failed %d", ret);
        }
        else if (pipeline_cache_size > UINT_MAX || pipeline_cache_size > INT_MAX)
        {
            NCNN_LOGE("vulkan pipeline cache data too large");
        }
        else if (pipeline_cache_size > 0)
        {
            pipeline_cache_data.resize(pipeline_cache_size);
            ret = vkGetPipelineCacheData(vkdev->vkdevice(), d->vk_pipeline_cache, &pipeline_cache_size, pipeline_cache_data.data());
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkGetPipelineCacheData failed %d", ret);
                pipeline_cache_data.clear();
            }
            else
            {
                pipeline_cache_data.resize(pipeline_cache_size);

                if (!validate_vk_pipeline_cache_data(pipeline_cache_data.data(), pipeline_cache_data.size(), vkdev))
                {
                    NCNN_LOGE("vulkan pipeline cache data validation failed");
                    pipeline_cache_data.clear();
                }
            }
        }
    }

    PipelineCachePrivate::cache_file_header header;
    fill_cache_file_header(header, vkdev);
    header.spirv_entry_count = (uint32_t)d->cache_spirv_entries.size();
    header.spirv_cache_size = (uint32_t)spirv_cache_data.size();
    header.spirv_cache_hash = spirv_cache_data.empty() ? 0 : fnv1a_32(spirv_cache_data.data(), (int)spirv_cache_data.size());
    header.pipeline_cache_size = (uint32_t)pipeline_cache_data.size();
    header.pipeline_cache_hash = pipeline_cache_data.empty() ? 0 : fnv1a_32(pipeline_cache_data.data(), (int)pipeline_cache_data.size());

    data.clear();
    data.reserve(sizeof(header) + spirv_cache_data.size() + pipeline_cache_data.size());
    append_data(data, &header, sizeof(header));
    append_data(data, spirv_cache_data.data(), spirv_cache_data.size());
    append_data(data, pipeline_cache_data.data(), pipeline_cache_data.size());

    return 0;
}

int PipelineCache::load_cache(const unsigned char* data, size_t size) const
{
    if (!data || size < sizeof(PipelineCachePrivate::cache_file_header))
        return -1;

    MutexLockGuard lock(d->cache_lock);

    PipelineCachePrivate::cache_file_header header;
    memcpy(&header, data, sizeof(header));

    if (!validate_cache_file_header(header, vkdev))
    {
        NCNN_LOGE("pipeline cache header validation failed");
        return -1;
    }

    size_t offset = sizeof(header);
    if (header.spirv_cache_size > size - offset)
        return -1;

    const unsigned char* spirv_cache_data = data + offset;
    if (header.spirv_cache_size > 0)
    {
        if (header.spirv_cache_size > INT_MAX)
            return -1;

        uint32_t spirv_cache_hash = fnv1a_32(spirv_cache_data, (int)header.spirv_cache_size);
        if (spirv_cache_hash != header.spirv_cache_hash)
        {
            NCNN_LOGE("spirv cache hash mismatch");
            return -1;
        }
    }

    std::vector<PipelineCachePrivate::spirv_cache_entry> cache_spirv_entries;
    size_t spirv_offset = 0;
    for (uint32_t i = 0; i < header.spirv_entry_count; i++)
    {
        if (spirv_offset + sizeof(PipelineCachePrivate::spirv_cache_entry_header) > header.spirv_cache_size)
            return -1;

        PipelineCachePrivate::spirv_cache_entry_header entry_header;
        memcpy(&entry_header, spirv_cache_data + spirv_offset, sizeof(entry_header));
        spirv_offset += sizeof(entry_header);

        if (entry_header.shader_source_hash == 0 || entry_header.shader_source_hash != get_shader_source_hash(entry_header.shader_type_index))
        {
            NCNN_LOGE("spirv entry shader source hash mismatch");
            return -1;
        }
        if (entry_header.spv_size == 0 || entry_header.spv_size % 4 != 0)
            return -1;
        if (entry_header.spv_size > header.spirv_cache_size - spirv_offset)
            return -1;
        if (entry_header.spv_size > INT_MAX)
            return -1;

        const unsigned char* spv_data = spirv_cache_data + spirv_offset;
        uint32_t spv_hash_fnv1a = fnv1a_32(spv_data, (int)entry_header.spv_size);
        if (spv_hash_fnv1a != entry_header.spv_hash_fnv1a)
        {
            NCNN_LOGE("spirv entry hash mismatch");
            return -1;
        }

        std::vector<uint32_t> spirv(entry_header.spv_size / 4);
        memcpy(spirv.data(), spv_data, entry_header.spv_size);
        spirv_offset += entry_header.spv_size;

        uint32_t spv_hash_murmur3 = murmur3_32(spirv.data(), (int)spirv.size());
        if (spv_hash_murmur3 != entry_header.spv_hash_murmur3)
        {
            NCNN_LOGE("spirv entry hash mismatch");
            return -1;
        }

        for (size_t j = 0; j < cache_spirv_entries.size(); j++)
        {
            if (cache_spirv_entries[j].shader_type_index == entry_header.shader_type_index && cache_spirv_entries[j].opt_bits == entry_header.opt_bits)
            {
                NCNN_LOGE("duplicate spirv cache entry");
                return -1;
            }
        }

        ShaderInfo si;
        if (resolve_shader_info(spirv.data(), spirv.size() * sizeof(uint32_t), si) != 0)
        {
            NCNN_LOGE("resolve_shader_info failed for spirv cache entry");
            return -1;
        }

        PipelineCachePrivate::spirv_cache_entry entry;
        entry.shader_type_index = entry_header.shader_type_index;
        entry.opt_bits = entry_header.opt_bits;
        entry.shader_source_hash = entry_header.shader_source_hash;
        entry.spirv = spirv;
        entry.shader_info = si;
        cache_spirv_entries.push_back(entry);
    }

    if (spirv_offset != header.spirv_cache_size)
        return -1;

    offset += header.spirv_cache_size;
    if (header.pipeline_cache_size > size - offset)
        return -1;
    if (offset + header.pipeline_cache_size != size)
        return -1;

    const unsigned char* pipeline_cache_data = data + offset;
    if (header.pipeline_cache_size > 0)
    {
        if (header.pipeline_cache_size > INT_MAX)
            return -1;

        uint32_t pipeline_cache_hash = fnv1a_32(pipeline_cache_data, (int)header.pipeline_cache_size);
        if (pipeline_cache_hash != header.pipeline_cache_hash)
        {
            NCNN_LOGE("vulkan pipeline cache hash mismatch");
            return -1;
        }

        if (!validate_vk_pipeline_cache_data(pipeline_cache_data, header.pipeline_cache_size, vkdev))
        {
            NCNN_LOGE("vulkan pipeline cache data validation failed");
            return -1;
        }
    }

    VkPipelineCache pipeline_cache = 0;
    if (header.pipeline_cache_size > 0)
    {
        if (create_vk_pipeline_cache(vkdev, pipeline_cache_data, header.pipeline_cache_size, &pipeline_cache) != 0)
            return -1;
    }

    if (pipeline_cache)
    {
        if (d->vk_pipeline_cache)
        {
            VkResult ret = vkMergePipelineCaches(vkdev->vkdevice(), d->vk_pipeline_cache, 1, &pipeline_cache);
            vkDestroyPipelineCache(vkdev->vkdevice(), pipeline_cache, 0);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkMergePipelineCaches failed %d", ret);
                return -1;
            }
        }
        else
        {
            d->vk_pipeline_cache = pipeline_cache;
        }
    }

    for (size_t i = 0; i < cache_spirv_entries.size(); i++)
    {
        const PipelineCachePrivate::spirv_cache_entry& entry = cache_spirv_entries[i];

        bool updated = false;
        for (size_t j = 0; j < d->cache_spirv_entries.size(); j++)
        {
            PipelineCachePrivate::spirv_cache_entry& old_entry = d->cache_spirv_entries[j];
            if (old_entry.shader_type_index != entry.shader_type_index || old_entry.opt_bits != entry.opt_bits)
                continue;

            old_entry = entry;
            updated = true;
            break;
        }

        if (!updated)
            d->cache_spirv_entries.push_back(entry);
    }

    return 0;
}

int PipelineCache::load_cache(const std::vector<unsigned char>& data) const
{
    if (data.empty())
        return -1;

    return load_cache(data.data(), data.size());
}

#if NCNN_STDIO
int PipelineCache::save_cache(FILE* fp) const
{
    if (!fp)
        return -1;

    std::vector<unsigned char> data;
    int ret = save_cache(data);
    if (ret != 0)
        return ret;

    if (fwrite(data.data(), 1, data.size(), fp) != data.size())
        return -1;

    if (fflush(fp) != 0)
        return -1;

    return 0;
}

int PipelineCache::load_cache(FILE* fp) const
{
    if (!fp)
        return -1;

    if (fseek(fp, 0, SEEK_END) != 0)
        return -1;
    long file_size = ftell(fp);
    if (file_size <= 0)
        return -1;

    const size_t cache_file_size_limit = (size_t)256 * 1024 * 1024;
    if ((size_t)file_size > cache_file_size_limit)
        return -1;
    if (fseek(fp, 0, SEEK_SET) != 0)
        return -1;

    std::vector<unsigned char> data((size_t)file_size);
    if (fread(data.data(), 1, data.size(), fp) != data.size())
        return -1;

    return load_cache(data);
}

int PipelineCache::save_cache(const char* path) const
{
    if (!path)
        return -1;

    std::vector<unsigned char> data;
    int ret = save_cache(data);
    if (ret != 0)
        return ret;

    std::string tmp_path;
    {
        MutexLockGuard lock(g_tmp_path_lock);

        char tmp_path_suffix[64];
#if defined(_WIN32)
        snprintf(tmp_path_suffix, sizeof(tmp_path_suffix), ".tmp.%u.%u", (unsigned int)_getpid(), ++g_tmp_path_index);
#else
        snprintf(tmp_path_suffix, sizeof(tmp_path_suffix), ".tmp.%u.%u", (unsigned int)getpid(), ++g_tmp_path_index);
#endif
        tmp_path = std::string(path) + tmp_path_suffix;
    }

    FILE* fp = fopen(tmp_path.c_str(), "wb");
    if (!fp)
        return -1;

    if (fwrite(data.data(), 1, data.size(), fp) != data.size())
    {
        fclose(fp);
        remove(tmp_path.c_str());
        return -1;
    }

    if (fclose(fp) != 0)
    {
        remove(tmp_path.c_str());
        return -1;
    }

    if (replace_file(tmp_path.c_str(), path) != 0)
    {
        remove(tmp_path.c_str());
        return -1;
    }

    return 0;
}

int PipelineCache::load_cache(const char* path) const
{
    if (!path)
        return -1;

    FILE* fp = fopen(path, "rb");
    if (!fp)
        return -1;

    int ret = load_cache(fp);
    fclose(fp);

    return ret;
}

#if defined(_WIN32)
int PipelineCache::save_cache(const wchar_t* path) const
{
    if (!path)
        return -1;

    std::vector<unsigned char> data;
    int ret = save_cache(data);
    if (ret != 0)
        return ret;

    std::wstring tmp_path;
    {
        MutexLockGuard lock(g_tmp_path_lock);

        wchar_t tmp_path_suffix[64];
        swprintf(tmp_path_suffix, sizeof(tmp_path_suffix) / sizeof(wchar_t), L".tmp.%u.%u", (unsigned int)_getpid(), ++g_tmp_path_index);
        tmp_path = std::wstring(path) + tmp_path_suffix;
    }

    FILE* fp = _wfopen(tmp_path.c_str(), L"wb");
    if (!fp)
        return -1;

    if (fwrite(data.data(), 1, data.size(), fp) != data.size())
    {
        fclose(fp);
        _wremove(tmp_path.c_str());
        return -1;
    }

    if (fclose(fp) != 0)
    {
        _wremove(tmp_path.c_str());
        return -1;
    }

    if (replace_file(tmp_path.c_str(), path) != 0)
    {
        _wremove(tmp_path.c_str());
        return -1;
    }

    return 0;
}

int PipelineCache::load_cache(const wchar_t* path) const
{
    if (!path)
        return -1;

    FILE* fp = _wfopen(path, L"rb");
    if (!fp)
        return -1;

    int ret = load_cache(fp);
    fclose(fp);

    return ret;
}
#endif // defined(_WIN32)
#endif // NCNN_STDIO

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

int PipelineCache::create_shader_module(int shader_type_index, const Option& opt,
                                        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                        VkShaderModule* _shader_module, ShaderInfo& si) const
{
    std::vector<uint32_t> spirv;
    const std::vector<uint32_t>* spirv_ptr = 0;
    const uint32_t opt_bits = encode_spirv_cache_opt_bits(opt);
    const uint64_t shader_source_hash = get_shader_source_hash(shader_type_index);
    const bool use_spirv_cache = shader_source_hash != 0 && can_cache_spirv(vkdev, opt);

    if (use_spirv_cache)
    {
        for (size_t i = 0; i < d->cache_spirv_entries.size(); i++)
        {
            const PipelineCachePrivate::spirv_cache_entry& entry = d->cache_spirv_entries[i];
            if (entry.shader_type_index != shader_type_index || entry.opt_bits != opt_bits || entry.shader_source_hash != shader_source_hash)
                continue;

            spirv_ptr = &entry.spirv;
            si = entry.shader_info;
            break;
        }
    }

    if (!spirv_ptr)
    {
        int retc = compile_spirv_module(shader_type_index, opt, spirv);
        if (retc != 0)
        {
            NCNN_LOGE("compile_spirv_module failed %d", retc);
            return -1;
        }

        const uint32_t* spv_data = spirv.data();
        size_t spv_data_size = spirv.size() * 4;

        int ret = resolve_shader_info(spv_data, spv_data_size, si);
        if (ret != 0)
        {
            NCNN_LOGE("resolve_shader_info failed %d", ret);
            return -1;
        }

        if (use_spirv_cache)
        {
            PipelineCachePrivate::spirv_cache_entry entry;
            entry.shader_type_index = shader_type_index;
            entry.opt_bits = opt_bits;
            entry.shader_source_hash = shader_source_hash;
            entry.spirv = spirv;
            entry.shader_info = si;

            bool updated = false;
            for (size_t i = 0; i < d->cache_spirv_entries.size(); i++)
            {
                PipelineCachePrivate::spirv_cache_entry& old_entry = d->cache_spirv_entries[i];
                if (old_entry.shader_type_index != shader_type_index || old_entry.opt_bits != opt_bits)
                    continue;

                old_entry = entry;
                updated = true;
                break;
            }

            if (!updated)
                d->cache_spirv_entries.push_back(entry);
        }

        spirv_ptr = &spirv;
    }

    const uint32_t* spv_data = spirv_ptr->data();
    size_t spv_data_size = spirv_ptr->size() * 4;

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

    ensure_vk_pipeline_cache(vkdev, d);

    ret = vkdev->create_pipeline(shader_module, pipeline_layout, specializations, subgroup_size, d->vk_pipeline_cache, &pipeline);
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
