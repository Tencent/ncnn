// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pipelinecache.h"

#include "gpu.h"

#include <cstdio>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <direct.h>
#endif

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

static int atomic_rename(const char* old_path, const char* new_path)
{
#ifdef _WIN32
    if (MoveFileExA(old_path, new_path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return 0;
    return -1;
#else
    return std::rename(old_path, new_path);
#endif // _WIN32
}

static int make_dir(const std::string& dirpath)
{
    if (dirpath.empty())
        return -1;

    std::string dir = dirpath;

#ifdef _WIN32
    for (size_t i = 0; i < dir.size(); i++)
    {
        if (dir[i] == '/')
            dir[i] = '\\';
    }

    size_t start = (dir.size() > 2 && dir[1] == ':') ? 3 : 0;

    for (size_t i = start; i <= dir.size(); i++)
    {
        if (i == dir.size() || dir[i] == '\\')
        {
            char tmp = dir[i];
            dir[i] = '\0';
            if (_mkdir(dir.c_str()) != 0 && errno != EEXIST)
            {
                return -1;
            }
            dir[i] = tmp;
        }
    }
#else
    size_t start = dir[0] == '/' ? 1 : 0;

    for (size_t i = start; i <= dir.size(); i++)
    {
        if (i == dir.size() || dir[i] == '/')
        {
            char tmp = dir[i];
            dir[i] = '\0';
            if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
            {
                return -1;
            }
            dir[i] = tmp;
        }
    }
#endif

    return 0;
}

static constexpr uint32_t spv_cache_magic()
{
    return ('S' | 'P' << 8 | 'V' << 16 | 'C' << 24);
}

enum class PipelineCacheIOResult
{
    Success,
    FileFailure,
    InvalidFile,
    InvalidCache,
    DataCorruption,
    CreationFailure,
};

class PipelineCachePrivate
{
public:
    static constexpr uint32_t CURRENT_SPV_CACHE_HEADER_VERSION = 1;
    static constexpr uint32_t CURRENT_PIPELINE_CACHE_VERSION = 1;

    PipelineCachePrivate()
    {
#ifdef _WIN32
        shader_cache_dir = std::string(getenv("LOCALAPPDATA") ? getenv("LOCALAPPDATA") : ".") + "/ncnn/shadercache";
#else
        shader_cache_dir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.ncnn/shadercache";
#endif
    }

    struct pipeline_cache_prefix_header
    {
        uint32_t magic;
        uint32_t version;
        uint32_t data_size;
        uint32_t data_hash_fnv1a; // fnv1a hash

        uint32_t vendor_id;
        uint32_t device_id;
        uint32_t driver_version;
        uint32_t driver_abi;

        uint8_t uuid[VK_UUID_SIZE];

        uint32_t reserved[4];
    };

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

    struct spv_cache_header
    {
        uint32_t magic;          // magic number, 'SPVC' in host endian
        uint32_t header_version; // version of cache header format
        uint32_t ncnn_version;   // ncnn version when the cache is created
        // if ncnn upgrade and update glslang, shader code or preprocessing steps
        // we want the cache to be invalid

        uint32_t spv_size;          // size of spv binary data
        uint32_t data_hash_fnv1a;   // hash of spv binary data using fnv1a
        uint32_t data_hash_murmur3; // second hash of spv binary data using murmur3

        // since a driver update/device switch might lead changes to supported extensions
        // and change the defines added in code, we want to verify that the cache is valid for the current device
        uint32_t vendor_id;
        uint32_t device_id;
        uint32_t driver_version;
        uint8_t uuid[VK_UUID_SIZE];
        uint32_t reserved[4]; // reserved for future use, must be zero
    };

    mutable std::vector<pipeline_cache_digest> cache_digests;
    mutable std::vector<pipeline_cache_artifact> cache_artifacts;
    mutable std::map<uint64_t, std::vector<uint32_t> > spv_code_cache;
    mutable VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    mutable Mutex cache_lock;
    mutable std::string shader_cache_dir;

    int load_spv_code_cache_from_disk(const VulkanDevice& device, uint64_t shader_key) const;
    PipelineCacheIOResult try_load_pipeline_cache_from_disk(const VulkanDevice* vkdev, const char* path);
    int save_spv_code_cache_to_disk(uint64_t shader_key, const VulkanDevice& device, const std::vector<uint32_t>& spirv) const;

    static constexpr uint32_t vk_pipeline_cache_header_magic()
    {
        return ('V' | 'P' << 8 | 'C' << 16 | 'H' << 24); // Vulkan Pipeline Cache Header
    }

    static bool validate_pipeline_cache_header(const pipeline_cache_prefix_header& header, const VkPhysicalDeviceProperties& physical_device_properties)
    {
        if (header.magic != vk_pipeline_cache_header_magic())
            return false;
        if (header.vendor_id != physical_device_properties.vendorID)
            return false;
        if (header.device_id != physical_device_properties.deviceID)
            return false;
        if (header.driver_version != physical_device_properties.driverVersion)
            return false;
        if (header.driver_abi != sizeof(void*))
            return false;
        if (std::memcmp(header.uuid, physical_device_properties.pipelineCacheUUID, VK_UUID_SIZE) != 0)
            return false;
        return true;
    }

    static bool validate_spv_code_cache(const spv_cache_header& header, const VkPhysicalDeviceProperties& physical_device_properties)
    {
        if (header.magic != spv_cache_magic())
            return false;
        if (header.header_version != CURRENT_SPV_CACHE_HEADER_VERSION)
            return false;
        if (header.vendor_id != physical_device_properties.vendorID)
            return false;
        if (header.device_id != physical_device_properties.deviceID)
            return false;
        if (header.driver_version != physical_device_properties.driverVersion)
            return false;
        if (header.spv_size % 4 != 0)
            return false;
        if (std::memcmp(header.uuid, physical_device_properties.pipelineCacheUUID, VK_UUID_SIZE) != 0)
            return false;
        return true;
    }
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

static uint32_t encode_opt_bits(const Option& opt)
{
    return 0 << 7
           | opt.use_fp16_packed << 6
           | opt.use_fp16_storage << 5
           | opt.use_fp16_arithmetic << 4
           | opt.use_int8_storage << 3
           | opt.use_int8_arithmetic << 2;
}

static uint64_t shader_spv_key(int shader_type_index, const Option& opt)
{
    // TODO: if shader code is changed, using shader_type_index is not enough
    return static_cast<uint64_t>(shader_type_index) << 32
           | static_cast<uint64_t>(opt.use_fp16_uniform) << 31
           | static_cast<uint64_t>(opt.use_int8_uniform) << 30
           | static_cast<uint64_t>(opt.use_int8_packed) << 29
           | static_cast<uint64_t>(opt.use_subgroup_ops) << 28
           | static_cast<uint64_t>(opt.use_shader_pack8) << 27
           | static_cast<uint64_t>(opt.use_shader_local_memory) << 26
           | encode_opt_bits(opt);
}

PipelineCachePrivate::pipeline_cache_digest::pipeline_cache_digest(int _shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
        uint32_t _local_size_x, uint32_t _local_size_y, uint32_t _local_size_z, uint32_t _subgroup_size)
{
    shader_type_index = _shader_type_index;

    // encode opt
    opt_bits = encode_opt_bits(opt);

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

    if (d->pipeline_cache)
    {
        vkDestroyPipelineCache(vkdev->vkdevice(), d->pipeline_cache, 0);
        d->pipeline_cache = VK_NULL_HANDLE;
    }

    d->spv_code_cache.clear();

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

int PipelineCachePrivate::load_spv_code_cache_from_disk(const VulkanDevice& device, uint64_t shader_key) const
{
    std::string cachepath = shader_cache_dir + "/" + std::to_string(shader_key) + ".spvcache";

    FILE* fp = fopen(cachepath.c_str(), "rb");
    if (!fp)
    {
        return -1;
    }

    spv_cache_header header;
    if (fread(&header, sizeof(header), 1, fp) != 1)
    {
        NCNN_LOGE("load_spv_code_cache_from_disk fread header failed", errno);
        fclose(fp);
        return -1;
    }

    if (!validate_spv_code_cache(header, device.info.physicalDeviceProperties()))
    {
        NCNN_LOGE("load_spv_code_cache_from_disk validate_spv_code_cache failed");
        fclose(fp);
        return -1;
    }

    std::vector<uint32_t> spirv;
    spirv.resize(header.spv_size / 4);
    size_t nread = fread(spirv.data(), 1, header.spv_size, fp);
    fclose(fp);

    if (nread != header.spv_size)
    {
        NCNN_LOGE("load_spv_code_cache_from_disk fread spirv data failed %zu != %d", nread, header.spv_size);
        return -1;
    }

    uint32_t hash_fnv1a = fnv1a_32(reinterpret_cast<const uint8_t*>(spirv.data()), header.spv_size);
    if (hash_fnv1a != header.data_hash_fnv1a)
    {
        NCNN_LOGE("load_spv_code_cache_from_disk data hash1 mismatch %x != %x", hash_fnv1a, header.data_hash_fnv1a);
        return -1;
    }

    uint32_t hash_murmur3 = murmur3_32(spirv.data(), spirv.size());
    if (hash_murmur3 != header.data_hash_murmur3)
    {
        NCNN_LOGE("load_spv_code_cache_from_disk data hash2 mismatch %x != %x", hash_murmur3, header.data_hash_murmur3);
        return -1;
    }

    spv_code_cache[shader_key] = std::move(spirv);
    return 0;
}
PipelineCacheIOResult PipelineCachePrivate::try_load_pipeline_cache_from_disk(const VulkanDevice* vkdev, const char* path)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        return PipelineCacheIOResult::FileFailure;
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    if (file_size == -1)
    {
        fclose(file);
        return PipelineCacheIOResult::FileFailure;
    }
    rewind(file);

    if (file_size < sizeof(PipelineCachePrivate::pipeline_cache_prefix_header))
    {
        fclose(file);
        return PipelineCacheIOResult::InvalidFile;
    }

    std::vector<char> buffer(file_size - sizeof(PipelineCachePrivate::pipeline_cache_prefix_header));
    PipelineCachePrivate::pipeline_cache_prefix_header header;
    if (fread(&header, sizeof(PipelineCachePrivate::pipeline_cache_prefix_header), 1, file) != 1)
    {
        fclose(file);
        return PipelineCacheIOResult::InvalidFile;
    }
    if (fread(buffer.data(), 1, file_size - sizeof(PipelineCachePrivate::pipeline_cache_prefix_header), file) != file_size - sizeof(PipelineCachePrivate::pipeline_cache_prefix_header))
    {
        fclose(file);
        return PipelineCacheIOResult::DataCorruption;
    }
    fclose(file);

    if (header.magic != PipelineCachePrivate::vk_pipeline_cache_header_magic())
    {
        return PipelineCacheIOResult::InvalidCache;
    }

    if (header.version != PipelineCachePrivate::CURRENT_PIPELINE_CACHE_VERSION)
    {
        return PipelineCacheIOResult::InvalidCache;
    }

    void* cache_data_begin = buffer.data();
    const VkPhysicalDeviceProperties& device_properties = vkdev->info.physicalDeviceProperties();
    if (!PipelineCachePrivate::validate_pipeline_cache_header(header, device_properties))
    {
        return PipelineCacheIOResult::InvalidCache;
    }

    size_t cache_data_size = header.data_size;
    if (cache_data_size == 0 || cache_data_size > buffer.size())
    {
        return PipelineCacheIOResult::DataCorruption;
    }

    uint64_t hash = fnv1a_32(reinterpret_cast<const uint8_t*>(cache_data_begin), cache_data_size);
    if (hash != header.data_hash_fnv1a)
    {
        return PipelineCacheIOResult::DataCorruption;
    }

    if (vkdev->create_pipeline_cache_with_data(cache_data_begin, cache_data_size, &pipeline_cache) != VK_SUCCESS)
    {
        return PipelineCacheIOResult::CreationFailure;
    }

    return PipelineCacheIOResult::Success;
}
int PipelineCachePrivate::save_spv_code_cache_to_disk(uint64_t shader_key, const VulkanDevice& device, const std::vector<uint32_t>& spirv) const
{
    std::string cachepath = shader_cache_dir + "/" + std::to_string(shader_key) + ".spvcache";
    std::string tmp_cachepath = cachepath + ".tmp";

    make_dir(shader_cache_dir);

    FILE* fp = fopen(tmp_cachepath.c_str(), "wb");
    if (!fp)
    {
        NCNN_LOGE("save_spv_code_cache_to_disk fopen %s failed", tmp_cachepath.c_str());
        return -1;
    }

    spv_cache_header header;
    header.magic = spv_cache_magic();
    header.header_version = CURRENT_SPV_CACHE_HEADER_VERSION;
    header.spv_size = spirv.size() * sizeof(uint32_t);

    header.data_hash_fnv1a = fnv1a_32((const uint8_t*)spirv.data(), header.spv_size);   // fnv1a hash
    header.data_hash_murmur3 = murmur3_32((const uint32_t*)spirv.data(), spirv.size()); // murmur3 hash

    const VkPhysicalDeviceProperties& physical_device_properties = device.info.physicalDeviceProperties();
    header.vendor_id = physical_device_properties.vendorID;
    header.device_id = physical_device_properties.deviceID;
    header.driver_version = physical_device_properties.driverVersion;
    std::memcpy(header.uuid, physical_device_properties.pipelineCacheUUID, VK_UUID_SIZE);
    std::memset(header.reserved, 0, sizeof(header.reserved));
    if (fwrite(&header, sizeof(header), 1, fp) != 1)
    {
        NCNN_LOGE("save_spv_code_cache_to_disk fwrite header failed", errno);
        fclose(fp);
        return -1;
    }

    if (fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), fp) != spirv.size())
    {
        NCNN_LOGE("save_spv_code_cache_to_disk fwrite spirv data failed", errno);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    if (atomic_rename(tmp_cachepath.c_str(), cachepath.c_str()) != 0)
    {
        NCNN_LOGE("save_spv_code_cache_to_disk rename %s to %s failed", tmp_cachepath.c_str(), cachepath.c_str());
        return -1;
    }

    return 0;
}

int PipelineCache::load_pipeline_cache(const char* path) const
{
    MutexLockGuard lock(d->cache_lock);
    if (d->pipeline_cache != VK_NULL_HANDLE)
    {
        NCNN_LOGE("a valid pipeline cache already exists, stop loading");
        return 0;
    }
    PipelineCacheIOResult result = d->try_load_pipeline_cache_from_disk(vkdev, path);
    if (result == PipelineCacheIOResult::Success) return 0;
    switch (result)
    {
    case PipelineCacheIOResult::FileFailure:
        NCNN_LOGE("Failed to open pipeline cache file: %s", path);
        break;
    case PipelineCacheIOResult::InvalidFile:
        NCNN_LOGE("File %s is not a valid file for pipeline cache", path);
        break;
    case PipelineCacheIOResult::InvalidCache:
        NCNN_LOGE("The cache in file %s is not valid for current platform", path);
        break;
    case PipelineCacheIOResult::DataCorruption:
        NCNN_LOGE("Data in file %s is corrupted", path);
        break;
    case PipelineCacheIOResult::CreationFailure:
        NCNN_LOGE("Failed to create pipeline cache from data in file %s", path);
        break;
    default:
        ;
    }

    NCNN_LOGE("Failed to load pipeline cache from file %s, fall back to create empty pipeline cache", path);
    if (vkdev->create_empty_pipeline_cache(&d->pipeline_cache) != 0)
    {
        NCNN_LOGE("Failed to create pipeline cache");
        return -1;
    }

    return 0;
}

int PipelineCache::save_pipeline_cache(const char* path) const
{
    MutexLockGuard lock(d->cache_lock);
    if (d->pipeline_cache == VK_NULL_HANDLE) return 0;
    size_t cache_data_size;
    if (vkGetPipelineCacheData(vkdev->vkdevice(), d->pipeline_cache, &cache_data_size, nullptr) != VK_SUCCESS)
    {
        NCNN_LOGE("Failed to get pipeline cache data");
        return -1;
    }

    std::vector<char> buffer(cache_data_size);
    if (vkGetPipelineCacheData(vkdev->vkdevice(), d->pipeline_cache, &cache_data_size, buffer.data()) != VK_SUCCESS)
    {
        NCNN_LOGE("Failed to get pipeline cache data");
        return -1;
    }

    const VkPhysicalDeviceProperties& device_properties = vkdev->info.physicalDeviceProperties();

    PipelineCachePrivate::pipeline_cache_prefix_header header = {};
    header.vendor_id = device_properties.vendorID;
    header.device_id = device_properties.deviceID;
    header.driver_version = device_properties.driverVersion;
    header.driver_abi = sizeof(void*);
    header.version = PipelineCachePrivate::CURRENT_PIPELINE_CACHE_VERSION;
    std::copy_n(device_properties.pipelineCacheUUID, VK_UUID_SIZE, header.uuid);
    header.data_size = cache_data_size;
    header.magic = PipelineCachePrivate::vk_pipeline_cache_header_magic();

    header.data_hash_fnv1a = fnv1a_32(reinterpret_cast<const uint8_t*>(buffer.data()), cache_data_size); // fnv1a hash

    std::string expected_path = path;
    std::string temp_file_path = expected_path + ".tmp";
    FILE* file = fopen(temp_file_path.c_str(), "wb");
    if (!file)
    {
        NCNN_LOGE("Failed to open temporary file %s for writing pipeline cache", temp_file_path.c_str());
        return -1;
    }

    size_t header_bytes_written = fwrite(&header, 1, sizeof(PipelineCachePrivate::pipeline_cache_prefix_header), file);
    size_t data_bytes_written = fwrite(buffer.data(), 1, cache_data_size, file);
    if (header_bytes_written != sizeof(PipelineCachePrivate::pipeline_cache_prefix_header) || data_bytes_written != cache_data_size)
    {
        NCNN_LOGE("Failed to write pipeline cache data to file %s", temp_file_path.c_str());
        fclose(file);
        return -1;
    }

    fclose(file);

    if (atomic_rename(temp_file_path.c_str(), expected_path.c_str()) != 0)
    {
        NCNN_LOGE("Failed to rename file %s to %s", temp_file_path.c_str(), path);
        return -1;
    }

    return 0;
}

int PipelineCache::create_shader_module(int shader_type_index, const Option& opt,
                                        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                        VkShaderModule* _shader_module, ShaderInfo& si) const
{
    const uint32_t* spv_data = nullptr;
    size_t spv_data_size = 0;
    uint64_t key = shader_spv_key(shader_type_index, opt);

    std::vector<uint32_t> spirv;
    if (d->spv_code_cache.find(key) != d->spv_code_cache.end() || d->load_spv_code_cache_from_disk(*vkdev, key) == 0)
    {
        const std::vector<uint32_t>& spirv_cache = d->spv_code_cache[key];
        spv_data = spirv_cache.data();
        spv_data_size = spirv_cache.size() * 4;
    }
    else
    {
        int retc = compile_spirv_module(shader_type_index, opt, spirv);
        if (retc != 0)
        {
            NCNN_LOGE("compile_spirv_module failed");
            return -1;
        }

        d->spv_code_cache[key] = spirv;
        int ret = d->save_spv_code_cache_to_disk(key, *vkdev, spirv);
        if (ret != 0)
        {
            NCNN_LOGE("save_spv_code_cache_to_disk failed", ret);
        }

        spv_data = spirv.data();
        spv_data_size = spirv.size() * 4;
    }

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

    if (!d->pipeline_cache)
    {
        ret = vkdev->create_empty_pipeline_cache(&d->pipeline_cache);
        if (ret != 0)
            NCNN_LOGE("vkdev->create_empty_pipeline_cache failed, don't use cache");
    }
    ret = vkdev->create_pipeline(shader_module, pipeline_layout, specializations, subgroup_size, &d->pipeline_cache, &pipeline);
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

void PipelineCache::set_shader_cache_dir(const char* dir)
{
    MutexLockGuard lock(d->cache_lock);
    d->shader_cache_dir = dir;
}

static bool clear_directory(const std::string& path)
{
#ifdef _WIN32
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA((path + "\\*").c_str(), &findData);
    if (hFind == INVALID_HANDLE_VALUE) return false;

    do {
        std::string name = findData.cFileName;
        if (name == "." || name == "..") continue;

        std::string fullPath = path + "\\" + name;
        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            clear_directory(fullPath);
            RemoveDirectoryA(fullPath.c_str());
        }
        else
        {
            DeleteFileA(fullPath.c_str());
        }
    } while (FindNextFileA(hFind, &findData));

    FindClose(hFind);
    return true;
#else
    DIR* dir = opendir(path.c_str());
    if (!dir) return false;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string fullPath = path + "/" + name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0)
        {
            if (S_ISDIR(st.st_mode))
            {
                remove_all_in_dir(fullPath);
                rmdir(fullPath.c_str());
            }
            else
            {
                unlink(fullPath.c_str());
            }
        }
    }
    closedir(dir);
    return true;
#endif
}

int PipelineCache::clear_shader_cache() const
{
    MutexLockGuard lock(d->cache_lock);
    d->spv_code_cache.clear();

    if (clear_directory(d->shader_cache_dir)) return 0;
    return -1;
}

#endif // NCNN_VULKAN

} // namespace ncnn
