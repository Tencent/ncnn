// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"

#if NCNN_VULKAN

#include "option.h"
#include "pipeline.h"

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

namespace ncnn {

size_t weight_buffer_alignment(const VulkanDevice* vkdev);
int weight_buffer_memory_size(const VulkanDevice* vkdev, size_t size, size_t& allocation_size, bool concurrent = false);
int create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t budget, VkBufferMemory*& buffer, size_t& allocation_size, bool allocate = true);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

int get_weight_cache_limits(const VkWeightAllocator* allocator, size_t& device_budget, size_t& host_budget, size_t& device_size, size_t& host_size);
int get_weight_source_generation(const VkAllocator* allocator, uint64_t& generation);

struct VkBufferBinding;

#if NCNN_BENCHMARK
// Internal instrumentation for statically linked diagnostic tools.
thread_local void (*weight_benchmark_callback)(const char* name, uint64_t count, double milliseconds) = 0;
#endif

static const int weight_transfer_slots = 32;
static const int weight_compute_slots = 4;

// Exclusively owned by one command. Model-owned sources remain immutable.
struct VkWeightCache
{
    struct source_entry
    {
        const VkBufferMemory* source;
        const VkAllocator* allocator;
        uint64_t generation;
    };

    struct Entry
    {
        const VkBufferMemory* source;
        const VkAllocator* allocator;
        uint64_t generation;
        size_t offset;
        size_t size;
        uint64_t compute_serial;
        bool valid;
    };

    struct HostEntry
    {
        const VkBufferMemory* source;
        const VkAllocator* allocator;
        uint64_t generation;
        size_t offset;
        size_t size;
    };

    VkWeightCache(const VulkanDevice* _vkdev, const Option& _opt, bool _automatic_device_size)
        : vkdev(_vkdev), opt(_opt), automatic_device_size(_automatic_device_size), host_allocator(0), device_allocator(0), host_arena(0), device_arena(0), host_size(_opt.weight_host_cache_size), device_size(_opt.weight_device_cache_size), alignment(weight_buffer_alignment(_vkdev)), host_head(0), device_head(0), handoff(0), handoff_pending(false), unified(false), initialized(false), error(0)
    {
#if NCNN_BENCHMARK
        memset(&stats, 0, sizeof(stats));
#endif
    }

    ~VkWeightCache()
    {
        clear();
    }

    const source_entry* find_source(const VkBufferMemory* source) const
    {
        for (size_t i = 0; i < sources.size(); i++)
            if (sources[i].source == source)
                return &sources[i];
        return 0;
    }

    void clear()
    {
        vkDestroySemaphore(vkdev->vkdevice(), handoff, 0);
        if (host_arena)
        {
            if (host_allocator)
                host_allocator->fastFree(host_arena);
            else
                destroy_weight_buffer(vkdev, host_arena);
        }
        if (device_arena)
        {
            if (device_allocator)
                device_allocator->fastFree(device_arena);
            else
                destroy_weight_buffer(vkdev, device_arena);
        }
        host_arena = 0;
        device_arena = 0;
        host_head = device_head = 0;
        entries.clear();
        host_entries.clear();
        handoff = 0;
        handoff_pending = false;
        initialized = false;
    }

    const VulkanDevice* vkdev;
    Option opt;
    bool automatic_device_size;
    VkAllocator* host_allocator;
    VkAllocator* device_allocator;
    VkBufferMemory* host_arena;
    VkBufferMemory* device_arena;
    size_t host_size;
    size_t device_size;
    size_t alignment;
    size_t host_head;
    size_t device_head;
    // Source identity is only needed when registering and validating a recording.
    // Arena growth preserves this list along with the pending commands.
    std::vector<source_entry> sources;
    std::vector<Entry> entries;
    std::vector<HostEntry> host_entries;
    VkSemaphore handoff;
    bool handoff_pending;
    bool unified;
    bool initialized;
    int error;
#if NCNN_BENCHMARK
    struct
    {
        uint64_t file_reads;
        uint64_t bytes_read;
        uint64_t bytes_uploaded;
        uint64_t cache_hits;
        uint64_t cache_misses;
        uint64_t transfer_submissions;
        uint64_t compute_submissions;
        uint64_t ownership_transfers;
        uint64_t host_bytes_staged;
        uint64_t host_cache_hits;
        uint64_t host_resident_bytes_uploaded;
        uint64_t upload_chunks;
        uint64_t unified_bytes_written;
        uint32_t peak_pending_compute;
        uint32_t peak_pending_transfer;
        double disk_read_time;
        double upload_time;
        double compute_time;
        double transfer_wait_time;
        double compute_wait_time;
        bool upload_time_available;
        size_t host_capacity;
        size_t device_capacity;
        size_t host_allocation_size;
        size_t device_allocation_size;
        bool separate_transfer_queue;
        bool separate_transfer_family;
        bool unified_memory;
    } stats;
#endif
};

static int resolve_weight_cache_options(const VkWeightAllocator* weights, Option& opt)
{
    size_t device_budget, host_budget, device_size, host_size;
    const int ret = get_weight_cache_limits(weights, device_budget, host_budget, device_size, host_size);
    if (ret <= 0)
        return ret;
    if (device_budget)
        opt.weight_device_budget = opt.weight_device_budget ? std::min(opt.weight_device_budget, device_budget) : device_budget;
    if (host_budget)
        opt.weight_host_budget = opt.weight_host_budget ? std::min(opt.weight_host_budget, host_budget) : host_budget;
    if (!opt.weight_device_cache_size)
        opt.weight_device_cache_size = device_size;
    if (!opt.weight_host_cache_size)
        opt.weight_host_cache_size = host_size;
    return ret;
}

static bool is_deferred_buffer(const VkBufferMemory* memory)
{
    return memory && memory->storage_type != 0 && memory->external_size != 0;
}

static bool needs_external_copy(const VkBufferMemory* memory)
{
    return memory->storage_type == 2 || !memory->buffer;
}

struct VkBufferBinding
{
    VkBufferBinding()
        : source(0), view_offset(0)
    {
        descriptor.buffer = VK_NULL_HANDLE;
        descriptor.offset = 0;
        descriptor.range = 0;
    }

    explicit VkBufferBinding(const VkMat& mat)
        : source(is_deferred_buffer(mat.data) ? mat.data : 0), view_offset(mat.offset)
    {
        descriptor.buffer = mat.buffer();
        descriptor.offset = mat.buffer_offset();
        descriptor.range = mat.buffer_capacity();
    }

    const VkBufferMemory* source;
    size_t view_offset;
    VkDescriptorBufferInfo descriptor;
};

class VkComputePrivate
{
public:
    VkComputePrivate(const VulkanDevice* _vkdev, const Option* opt = 0, int queue_mode = 0, const VkWeightAllocator* weights = 0);
    ~VkComputePrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();

    bool delayed() const
    {
        return weight_recording || !vkdev->info.support_VK_KHR_push_descriptor();
    }
    int ensure_weight_cache(const VkMat& mat, const Option* opt = 0);
    int submit_managed();
    void clear_records();

    struct Range
    {
        size_t offset;
        size_t size;
    };

    struct Upload
    {
        Upload()
        {
            command = 0;
            release_command = 0;
            fence = 0;
            release_fence = 0;
            ready = 0;
            returned = 0;
            pending = false;
            release_pending = false;
            host_active = false;
            consumer_serial = 0;
            host_offset = 0;
            host_size = 0;
            offset = 0;
            size = 0;
        }
        VkCommandBuffer command;
        VkCommandBuffer release_command;
        VkFence fence;
        VkFence release_fence;
        VkSemaphore ready;
        VkSemaphore returned;
        bool pending;
        bool release_pending;
        bool host_active;
        uint64_t consumer_serial;
        size_t host_offset;
        size_t host_size;
        size_t offset;
        size_t size;
    };

    struct Compute
    {
        Compute()
            : command(0), fence(0), serial(0), pending(false)
        {
        }
        VkCommandBuffer command;
        VkFence fence;
        uint64_t serial;
        bool pending;
    };

    int initialize_weight_cache(size_t minimum_device_size, size_t max_source_size, size_t max_external_size);
    int weight_binding_source(const VkBufferBinding& binding, const VkBufferMemory*& source) const;
    size_t weight_source_size(const VkBufferMemory* source) const;
    int begin_weight_commands();
    int submit_weight_prefix(VkCommandBuffer command, VkFence fence);
    int prepare_weights(const std::vector<const VkBufferMemory*>& weights);
    int begin_weight_compute(VkCommandBuffer& command);
    int submit_weight_compute();
    int finish_weight_commands(int result = 0);
    int resolve_weight_binding(const VkBufferBinding& binding, VkDescriptorBufferInfo& descriptor) const;
    int weight_device_entry(const VkBufferMemory* source) const;
    int weight_host_entry(const VkBufferMemory* source) const;
    int allocate_weight_cache(size_t minimum_device_size, size_t max_source_size, size_t max_external_size);
    int create_weight_commands();
    void destroy_weight_commands();
    void release_weight_cache();
    int upload_weight(const VkBufferMemory* source, size_t offset);
    int upload_weight_chunk(Upload& u, const VkBufferMemory* source, size_t source_offset, size_t offset, size_t size, const Range& range, bool last);
    int place_weight(const VkBufferMemory* source, const std::vector<const VkBufferMemory*>& pinned);
    void evict_weight(size_t slot);
    int evict_weights();
    int wait_weight_fence(VkFence fence, bool transfer);
    int wait_weight_compute(uint64_t serial);
    int retire_weight_upload(Upload& u);
    int weight_host_range(size_t size, size_t& offset);
    int cache_host_weight(const VkBufferMemory* source, size_t& offset);
    int flush_weight_host(size_t offset, size_t size);
    void update_weight_peaks();
    void weight_ownership_barrier(VkCommandBuffer command, const Range& range, bool to_compute, bool release);
#if NCNN_BENCHMARK
    int collect_weight_time(uint32_t query, uint32_t bits, double& elapsed);
    VkQueryPool timing_pool;
    uint32_t transfer_timestamp_bits;
    uint32_t compute_timestamp_bits;
#endif

    VkWeightCache* weight_cache;
    bool weight_recording;
    Option weight_opt;
    bool weight_options_set;
    bool weight_automatic_device_size;
    bool weight_automatic_host_size;
    int weight_config_error;
    std::vector<Range> pending_releases;
    std::vector<int> pending_uploads;
    std::vector<const VkBufferMemory*> current_sources;
    VkCommandPool transfer_pool;
    VkQueue compute_queue;
    VkQueue transfer_queue;
    VkFence prefix_fence;
    VkFence handoff_fence;
    int weight_queue_mode;
    uint32_t weight_transfer_family;
    Upload uploads[weight_transfer_slots];
    Compute computes[weight_compute_slots];
    int next_upload;
    int next_compute;
    int current_compute;
    uint64_t compute_serial;
    uint64_t completed_compute_serial;
    bool weight_commands_initialized;
    bool executing;
    int record_error;

    struct PipelineRecord
    {
        VkPipeline pipeline;
        VkPipelineLayout pipeline_layout;
        VkDescriptorSetLayout descriptorset_layout;
        VkDescriptorUpdateTemplateKHR descriptor_update_template;
        ShaderInfo shader_info;
        std::vector<VkBufferBinding> buffers;
        std::vector<VkDescriptorImageInfo> images;
        std::vector<vk_constant_type> constants;
        uint32_t group_count_x;
        uint32_t group_count_y;
        uint32_t group_count_z;
    };
    std::vector<PipelineRecord> pipelines;
    int create_descriptorset(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const unsigned char* descriptor_infos, VkDescriptorSet& descriptorset);
    int emit_pipeline(const PipelineRecord& r, VkCommandBuffer command);

    const VulkanDevice* vkdev;

    VkCommandPool compute_command_pool;

    VkCommandBuffer compute_command_buffer;

    VkFence compute_command_fence;

    std::vector<VkMat> upload_staging_buffers;
    std::vector<VkMat> download_post_buffers;
    std::vector<Mat> download_post_mats_fp16;
    std::vector<Mat> download_post_mats;

    std::vector<VkImageMemory*> image_blocks_to_destroy;

    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;

    struct record
    {
        enum
        {
            TYPE_pipeline,
            TYPE_copy_buffer,
            TYPE_copy_image,
            TYPE_copy_buffer_to_image,
            TYPE_copy_image_to_buffer,
            TYPE_bind_pipeline,
            TYPE_bind_descriptorsets,
            TYPE_push_constants,
            TYPE_push_descriptors,
            TYPE_dispatch,
            TYPE_memory_barrers,
            TYPE_buffer_barrers,
            TYPE_image_barrers,

#if NCNN_BENCHMARK
            TYPE_write_timestamp,
#endif // NCNN_BENCHMARK

            TYPE_post_download,
            TYPE_post_cast_float16_to_float32,
            TYPE_post_cast_bfloat16_to_float32,
        };

        int type;
        VkBufferBinding buffer_binding;
        size_t pipeline_index;

        union
        {
            struct
            {
                VkBuffer src;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferCopy* regions;
            } copy_buffer;
            struct
            {
                VkImage src;
                VkImageLayout src_layout;
                VkImage dst;
                VkImageLayout dst_layout;
                uint32_t region_count;
                const VkImageCopy* regions;
            } copy_image;
            struct
            {
                VkBuffer src;
                VkImage dst;
                VkImageLayout layout;
                uint32_t region_count;
                const VkBufferImageCopy* regions;
            } copy_buffer_to_image;
            struct
            {
                VkImage src;
                VkImageLayout layout;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferImageCopy* regions;
            } copy_image_to_buffer;

            struct
            {
                VkPipelineBindPoint bind_point;
                VkPipeline pipeline;
#if NCNN_COVERAGE
                VkPipelineLayout pipeline_layout;
#endif
            } bind_pipeline;
            struct
            {
                VkPipelineBindPoint bind_point;
                VkPipelineLayout pipeline_layout;
                uint32_t descriptorset_count;
                uint32_t descriptorset_offset;
            } bind_descriptorsets;
            struct
            {
                VkDescriptorUpdateTemplateKHR update_template;
                VkPipelineLayout pipeline_layout;
                const unsigned char* data;
            } push_descriptors;

            struct
            {
                VkPipelineLayout pipeline_layout;
                VkShaderStageFlags stage_flags;
                uint32_t size;
                const void* values;
            } push_constants;

            struct
            {
                uint32_t group_count_x;
                uint32_t group_count_y;
                uint32_t group_count_z;
            } dispatch;

            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkMemoryBarrier* barriers;
            } memory_barrers;
            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkBufferMemoryBarrier* barriers;
            } buffer_barrers;
            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkImageMemoryBarrier* barriers;
            } image_barrers;

#if NCNN_BENCHMARK
            struct
            {
                uint32_t query;
            } write_timestamp;
#endif // NCNN_BENCHMARK

            struct
            {
                uint32_t download_post_buffer_mat_offset;
                uint32_t download_post_mat_fp16_offset;
            } post_download;
            struct
            {
                uint32_t download_post_mat_fp16_offset;
                uint32_t download_post_mat_offset;
                int num_threads;
            } post_cast_float16_to_float32;
            struct
            {
                uint32_t download_post_mat_bf16_offset;
                uint32_t download_post_mat_offset;
                int num_threads;
            } post_cast_bfloat16_to_float32;
        };
    };

    int replay_record(const record& r, VkCommandBuffer command);
    int record_sources(const record& r, std::vector<const VkBufferMemory*>& weights);

    std::vector<record> delayed_records;

    uint64_t pending_dispatch_total;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
    bool query_results_ready;
#endif // NCNN_BENCHMARK
};

VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev, const Option* opt, int queue_mode, const VkWeightAllocator* weights)
    : weight_opt(opt ? *opt : Option()), vkdev(_vkdev)
{
    weight_options_set = opt != 0;
    weight_automatic_device_size = !weight_opt.weight_device_cache_size;
    weight_automatic_host_size = !weight_opt.weight_host_cache_size;
    const int ret = weights ? resolve_weight_cache_options(weights, weight_opt) : 0;
    weight_config_error = ret < 0 ? ret : 0;
    weight_cache = 0;
    weight_recording = false;
    transfer_pool = 0;
    compute_queue = 0;
    transfer_queue = 0;
    prefix_fence = 0;
    handoff_fence = 0;
    weight_queue_mode = queue_mode;
    weight_transfer_family = queue_mode ? vkdev->info.compute_queue_family_index() : vkdev->info.transfer_queue_family_index();
    next_upload = 0;
    next_compute = 0;
    current_compute = -1;
    compute_serial = 0;
    completed_compute_serial = 0;
    weight_commands_initialized = false;
    executing = false;
#if NCNN_BENCHMARK
    timing_pool = 0;
    transfer_timestamp_bits = 0;
    compute_timestamp_bits = 0;
#endif
    record_error = weight_config_error;

    compute_command_pool = 0;
    compute_command_buffer = 0;
    compute_command_fence = 0;

    pending_dispatch_total = 0;

#if NCNN_BENCHMARK
    query_count = 0;
    query_pool = 0;
    query_results_ready = false;
#endif // NCNN_BENCHMARK

    init();
}

VkComputePrivate::~VkComputePrivate()
{
    destroy_weight_commands();
    release_weight_cache();
    clear_records();

    for (size_t i = 0; i < image_blocks_to_destroy.size(); i++)
    {
        VkImageMemory* ptr = image_blocks_to_destroy[i];

        int old_command_refcount = NCNN_XADD(&ptr->command_refcount, -1);
        if (ptr->refcount == 0 && old_command_refcount == 1)
        {
            // no userspace reference and we are the last command reference
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }
        else
        {
            // reference exists in user code or other command
        }
    }
    image_blocks_to_destroy.clear();

    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        for (size_t i = 0; i < descriptor_pools.size(); i++)
        {
            vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pools[i], 0);
        }
    }

#if NCNN_BENCHMARK
    if (query_pool)
    {
        // all submitted commands that refer to queryPool must have completed execution
        vkResetCommandBuffer(compute_command_buffer, 0);

        vkDestroyQueryPool(vkdev->vkdevice(), query_pool, 0);
    }
#endif // NCNN_BENCHMARK

    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
}

int VkComputePrivate::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }

    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        begin_command_buffer();

#if NCNN_BENCHMARK
        if (query_pool)
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkComputePrivate::begin_command_buffer()
{
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;

    VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
        return -1;
    }

    return 0;
}

int VkComputePrivate::end_command_buffer()
{
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(compute_command_buffer);
#endif
    VkResult ret = vkEndCommandBuffer(compute_command_buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }

    return 0;
}

int VkComputePrivate::ensure_weight_cache(const VkMat& mat, const Option* opt)
{
    if (record_error)
        return record_error;
    if (!is_deferred_buffer(mat.data))
        return 0;
    if (mat.allocator && mat.allocator->vkdev != vkdev)
    {
        NCNN_LOGE("weight source and command must use the same Vulkan device");
        record_error = -1;
        return record_error;
    }
    if (!weight_cache)
    {
        if (opt && !weight_options_set)
        {
            weight_opt = *opt;
            weight_options_set = true;
            weight_automatic_device_size = !weight_opt.weight_device_cache_size;
            weight_automatic_host_size = !weight_opt.weight_host_cache_size;
        }
        weight_cache = new VkWeightCache(vkdev, weight_opt, weight_automatic_device_size);
    }
    weight_recording = true;
    const VkWeightCache::source_entry* recorded = weight_cache->find_source(mat.data);
    if (recorded)
    {
        if (recorded->allocator != mat.allocator)
            record_error = -1;
        return record_error;
    }

    VkWeightCache::source_entry source;
    source.source = mat.data;
    source.allocator = mat.allocator;
    source.generation = 0;
    bool found_allocator = false;
    for (size_t i = 0; i < weight_cache->sources.size(); i++)
    {
        if (weight_cache->sources[i].allocator != mat.allocator)
            continue;
        source.generation = weight_cache->sources[i].generation;
        found_allocator = true;
        break;
    }
    if (!found_allocator && get_weight_source_generation(mat.allocator, source.generation) < 0)
    {
        NCNN_LOGE("weight source must be finalized before recording");
        record_error = -1;
        return record_error;
    }

    // Reject stale identities before any pointer-only cache lookup. Eviction
    // retains the queue ownership handoff needed before reusing a DEVICE range.
    for (size_t i = 0; i < weight_cache->entries.size(); i++)
    {
        const VkWeightCache::Entry& entry = weight_cache->entries[i];
        if (entry.source == mat.data && (entry.allocator != source.allocator || entry.generation != source.generation))
            evict_weight(i);
    }
    for (size_t i = 0; i < weight_cache->host_entries.size(); i++)
    {
        VkWeightCache::HostEntry& entry = weight_cache->host_entries[i];
        if (entry.source == mat.data && (entry.allocator != source.allocator || entry.generation != source.generation))
            entry.source = 0;
    }
    weight_cache->sources.push_back(source);
    return 0;
}

void VkComputePrivate::release_weight_cache()
{
    delete weight_cache;
    weight_cache = 0;
}

void VkComputePrivate::clear_records()
{
    for (size_t i = 0; i < delayed_records.size(); i++)
    {
        const record& r = delayed_records[i];
        switch (r.type)
        {
        case record::TYPE_copy_buffer:
            delete[] r.copy_buffer.regions;
            break;
        case record::TYPE_copy_image:
            delete[] r.copy_image.regions;
            break;
        case record::TYPE_copy_buffer_to_image:
            delete[] r.copy_buffer_to_image.regions;
            break;
        case record::TYPE_copy_image_to_buffer:
            delete[] r.copy_image_to_buffer.regions;
            break;
        case record::TYPE_push_descriptors:
            delete[] r.push_descriptors.data;
            break;
        case record::TYPE_push_constants:
            delete[](const unsigned char*) r.push_constants.values;
            break;
        case record::TYPE_memory_barrers:
            delete[] r.memory_barrers.barriers;
            break;
        case record::TYPE_buffer_barrers:
            delete[] r.buffer_barrers.barriers;
            break;
        case record::TYPE_image_barrers:
            delete[] r.image_barrers.barriers;
            break;
        default:
            break;
        }
    }
    delayed_records.clear();
    pipelines.clear();
    if (weight_cache)
        weight_cache->sources.clear();
}

int VkComputePrivate::create_descriptorset(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const unsigned char* descriptor_infos, VkDescriptorSet& descriptorset)
{
    int buffer_binding_count = 0;
    int image_binding_count = 0;
    int sampler_binding_count = 0;
    for (int i = 0; i < binding_count; i++)
    {
        if (binding_types[i] == 1)
            buffer_binding_count++;
        else if (binding_types[i] == 2)
            image_binding_count++;
        else if (binding_types[i] == 3)
            sampler_binding_count++;
    }

    // skip unused descriptor types because zero counts are invalid
    // see https://github.com/Tencent/ncnn/issues/6951
    VkDescriptorPoolSize poolSizes[3];
    uint32_t pool_size_count = 0;
    if (buffer_binding_count > 0)
    {
        poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[pool_size_count].descriptorCount = buffer_binding_count;
        pool_size_count++;
    }
    if (image_binding_count > 0)
    {
        poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[pool_size_count].descriptorCount = image_binding_count;
        pool_size_count++;
    }
    if (sampler_binding_count > 0)
    {
        poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[pool_size_count].descriptorCount = sampler_binding_count;
        pool_size_count++;
    }

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = 0;
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = pool_size_count;
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descriptor_pool;
    VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
        return -1;
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = 0;
    descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

    ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
        vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pool, 0);
        return -1;
    }
    descriptor_pools.push_back(descriptor_pool);
    descriptorsets.push_back(descriptorset);

    if (vkdev->info.support_VK_KHR_descriptor_update_template())
    {
        vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, descriptor_update_template, descriptor_infos);
    }
    else
    {
        std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
        const unsigned char* p_descriptorInfos = descriptor_infos;
        for (int i = 0; i < binding_count; i++)
        {
            int binding_type = binding_types[i];

            writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[i].pNext = 0;
            writeDescriptorSets[i].dstSet = descriptorset;
            writeDescriptorSets[i].dstBinding = i;
            writeDescriptorSets[i].dstArrayElement = 0;
            writeDescriptorSets[i].descriptorCount = 1;
            writeDescriptorSets[i].pTexelBufferView = 0;

            if (binding_type == 1)
            {
                writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeDescriptorSets[i].pImageInfo = 0;
                writeDescriptorSets[i].pBufferInfo = (const VkDescriptorBufferInfo*)p_descriptorInfos;

                p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
            }
            else // if (binding_type == 2 || binding_type == 3)
            {
                writeDescriptorSets[i].descriptorType = binding_type == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                writeDescriptorSets[i].pBufferInfo = 0;

                p_descriptorInfos += sizeof(VkDescriptorImageInfo);
            }
        }

        vkUpdateDescriptorSets(vkdev->vkdevice(), binding_count, writeDescriptorSets.data(), 0, 0);
    }

    return 0;
}

int VkComputePrivate::emit_pipeline(const PipelineRecord& r, VkCommandBuffer command)
{
    const ShaderInfo& shader_info = r.shader_info;
    const int buffer_binding_count = (int)r.buffers.size();
    const int image_binding_count = (int)r.images.size();
    const int binding_count = buffer_binding_count + image_binding_count;
    std::vector<unsigned char> descriptorInfos(sizeof(VkDescriptorBufferInfo) * buffer_binding_count + sizeof(VkDescriptorImageInfo) * image_binding_count);
    unsigned char* p = descriptorInfos.data();
    int bi = 0;
    int ii = 0;
    for (int i = 0; i < binding_count; i++)
    {
        if (shader_info.binding_types[i] == 1)
        {
            VkDescriptorBufferInfo descriptor;
            const VkBufferBinding& binding = r.buffers[bi++];
            descriptor = binding.descriptor;
            if (weight_cache && resolve_weight_binding(binding, descriptor))
                return -1;
            memcpy(p, &descriptor, sizeof(descriptor));
            p += sizeof(descriptor);
        }
        else
        {
            memcpy(p, &r.images[ii++], sizeof(VkDescriptorImageInfo));
            p += sizeof(VkDescriptorImageInfo);
        }
    }
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline);
#if NCNN_COVERAGE
    vkdev->bind_shader_coverage(command, r.pipeline_layout);
#endif
    if (binding_count > 0)
    {
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command, r.descriptor_update_template, r.pipeline_layout, 0, descriptorInfos.data());
        }
        else
        {
            VkDescriptorSet descriptorset;
            if (create_descriptorset(binding_count, shader_info.binding_types, r.descriptorset_layout, r.descriptor_update_template, descriptorInfos.data(), descriptorset))
                return -1;

            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline_layout, 0, 1, &descriptorset, 0, 0);
        }
    }
    if (!r.constants.empty())
        vkCmdPushConstants(command, r.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, r.constants.size() * sizeof(vk_constant_type), r.constants.data());
    vkCmdDispatch(command, r.group_count_x, r.group_count_y, r.group_count_z);
    return 0;
}

int VkComputePrivate::record_sources(const record& r, std::vector<const VkBufferMemory*>& weights)
{
    const size_t count = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers.size() : 1;
    for (size_t i = 0; i < count; i++)
    {
        const VkBufferBinding& binding = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers[i] : r.buffer_binding;
        const VkBufferMemory* source;
        if (weight_binding_source(binding, source))
            return -1;
        if (!source)
            continue;
        bool found = false;
        for (size_t j = 0; j < weights.size(); j++)
            if (weights[j] == source)
                found = true;
        if (!found)
            weights.push_back(source);
    }
    return 0;
}

int VkComputePrivate::submit_managed()
{
    // Validate each allocator once, before dereferencing any recorded source.
    for (size_t i = 0; i < weight_cache->sources.size(); i++)
    {
        const VkWeightCache::source_entry& source = weight_cache->sources[i];
        if (!source.generation)
            continue;
        bool checked = false;
        for (size_t j = 0; j < i; j++)
        {
            if (weight_cache->sources[j].allocator == source.allocator)
            {
                checked = true;
                break;
            }
        }
        if (checked)
            continue;
        uint64_t generation;
        if (get_weight_source_generation(source.allocator, generation) < 0 || generation != source.generation)
        {
            NCNN_LOGE("weight source was cleared or changed while recording");
            return -1;
        }
    }
    size_t minimum_device_size = 0;
    size_t max_source_size = 0;
    size_t max_external_size = 0;
    std::vector<const VkBufferMemory*> dependencies;
    std::vector<size_t> dependency_offsets;
    std::vector<const VkBufferMemory*> sources;
    dependency_offsets.push_back(0);
    for (size_t i = 0; i < delayed_records.size(); i++)
    {
        sources.clear();
        if (record_sources(delayed_records[i], sources) || sources.size() > (size_t)weight_transfer_slots)
            return -1;
        size_t total = 0;
        for (size_t j = 0; j < sources.size(); j++)
        {
            const size_t size = weight_source_size(sources[j]);
            if (size > (size_t)-1 - total)
                return -1;
            total += size;
            max_source_size = std::max(max_source_size, size);
            if (needs_external_copy(sources[j]))
                max_external_size = std::max(max_external_size, size);
            dependencies.push_back(sources[j]);
        }
        dependency_offsets.push_back(dependencies.size());
        minimum_device_size = std::max(minimum_device_size, total);
    }
    if (initialize_weight_cache(minimum_device_size, max_source_size, max_external_size))
        return -1;
    struct Batch
    {
        size_t begin;
        size_t end;
        std::vector<const VkBufferMemory*> weights;
    };
    std::vector<Batch> batches;
    Batch batch;
    batch.begin = 0;
    size_t batch_size = 0;
    const size_t capacity = weight_cache->device_size;
    if (minimum_device_size > capacity)
    {
        NCNN_LOGE("simultaneous weight bindings exceed device cache budget");
        return -1;
    }
    for (size_t i = 0; i < delayed_records.size(); i++)
    {
        sources.clear();
        size_t added_size = 0;
        for (size_t j = dependency_offsets[i]; j < dependency_offsets[i + 1]; j++)
        {
            const VkBufferMemory* source = dependencies[j];
            bool found = false;
            for (size_t k = 0; k < batch.weights.size(); k++)
                found |= batch.weights[k] == source;
            if (found)
                continue;
            const size_t size = weight_source_size(source);
            if (size > (size_t)-1 - added_size)
                return -1;
            added_size += size;
            sources.push_back(source);
        }
        if (!batch.weights.empty() && !sources.empty()
                && (batch_size > capacity / 2 || added_size > capacity / 2 - batch_size
                    || batch.weights.size() + sources.size() > (size_t)weight_transfer_slots / 2))
        {
            batch.end = i;
            batches.push_back(batch);
            batch.begin = i;
            batch.weights.clear();
            batch_size = 0;
            for (size_t j = dependency_offsets[i]; j < dependency_offsets[i + 1]; j++)
            {
                batch.weights.push_back(dependencies[j]);
                batch_size += weight_source_size(dependencies[j]);
            }
        }
        else
        {
            for (size_t j = 0; j < sources.size(); j++)
                batch.weights.push_back(sources[j]);
            batch_size += added_size;
        }
    }
    batch.end = delayed_records.size();
    batches.push_back(batch);

    if (begin_weight_commands())
        return -1;
    int ret = 0;
    // an external command may have recorded an ordinary prefix before attaching
    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        ret = end_command_buffer();
        if (!ret)
            ret = submit_weight_prefix(compute_command_buffer, compute_command_fence);
    }
    for (size_t i = 0; i < batches.size() && !ret; i++)
    {
        ret = prepare_weights(batches[i].weights);
        if (ret)
            break;
        VkCommandBuffer command;
        ret = begin_weight_compute(command);
        if (ret)
            break;
#if NCNN_BENCHMARK
        if (i == 0 && query_pool && !vkdev->info.support_VK_KHR_push_descriptor())
            vkCmdResetQueryPool(command, query_pool, 0, query_count);
#endif
        for (size_t j = batches[i].begin; j < batches[i].end && !ret; j++)
            ret = replay_record(delayed_records[j], command);
        if (!ret)
            ret = submit_weight_compute();
    }
    ret = finish_weight_commands(ret);
    return ret;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev))
{
}

VkCompute::VkCompute(const VulkanDevice* _vkdev, const Option& opt, int queue_mode)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev, &opt, queue_mode))
{
}

VkCompute::VkCompute(const VkWeightAllocator& weights, const Option& opt, int queue_mode)
    : vkdev(weights.vkdev), d(new VkComputePrivate(weights.vkdev, &opt, queue_mode, &weights))
{
}

VkCompute::~VkCompute()
{
    delete d;
}

void VkCompute::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
    // NCNN_LOGE("record_upload buffer");

    Mat src_fp16;
    if (src.elemsize == src.elempack * 4u)
    {
        // cpu cast to fp16 (discrete gpu)
        if (vkdev->info.type() == 0 && (opt.use_bf16_storage || opt.use_bf16_packed))
        {
            ncnn::cast_float32_to_bfloat16(src, src_fp16, opt);
        }
        else if (vkdev->info.type() == 0 && (opt.use_fp16_storage || opt.use_fp16_packed))
        {
            ncnn::cast_float32_to_float16(src, src_fp16, opt);
        }
        else
        {
            src_fp16 = src;
        }
    }
    else
    {
        src_fp16 = src;
    }

    // vkdev->convert_packing only handles elempack=1/4
    if (src_fp16.elempack > 4)
    {
        Mat src_fp16_pack4;
        ncnn::convert_packing(src_fp16, src_fp16_pack4, 4, opt);
        src_fp16 = src_fp16_pack4;
    }

    // upload
    VkMat dst_staging;
    dst_staging.create_like(src_fp16, opt.staging_vkallocator);
    if (dst_staging.empty())
        return;

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);

    //     NCNN_LOGE("upload_staging_buffer %p  ->   %p +%d ~%d", src_fp16.data, dst_staging.buffer(), dst_staging.buffer_offset(), dst_staging.buffer_capacity());

    // memcpy src to device
    for (int b = 0; b < src_fp16.n; b++)
    {
        const Mat src_b = src_fp16.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }
    dst_staging.allocator->flush(dst_staging.data);

    // mark device host-write @ null
    dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    // resolve dst_elempack
    int dims = src_fp16.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src_fp16.elempack * src_fp16.w;
    if (dims == 2) elemcount = src_fp16.elempack * src_fp16.h;
    if (dims == 3 || dims == 4) elemcount = src_fp16.elempack * src_fp16.c;

    int dst_elempack = elemcount % 4 == 0 ? 4 : 1;

    // gpu cast to fp16 on the fly (integrated gpu)
    int cast_type_to = 0;
    if (vkdev->info.type() != 0)
    {
        if (opt.use_bf16_storage || opt.use_bf16_packed)
            cast_type_to = 5;
        else if (opt.use_fp16_storage || opt.use_fp16_packed)
            cast_type_to = 2;
        else
            cast_type_to = 1;
    }
    vkdev->convert_packing(dst_staging, dst, dst_elempack, cast_type_to, *this, opt);
}

void VkCompute::record_download(const VkMat& src, Mat& dst, const Option& opt)
{
    // NCNN_LOGE("record_download buffer");

    // resolve dst_elempack
    int dims = src.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src.elempack * src.w;
    if (dims == 2) elemcount = src.elempack * src.h;
    if (dims == 3 || dims == 4) elemcount = src.elempack * src.c;

    int dst_elempack = 1;
    if (opt.use_packing_layout)
        dst_elempack = elemcount % 4 == 0 ? 4 : 1;
    else
        dst_elempack = 1;

    // gpu cast to fp32 on the fly (integrated gpu)
    Option opt_staging = opt;
    if (!opt_staging.blob_vkallocator->mappable || !opt_staging.blob_vkallocator->cached)
    {
        // On an integrated gpu the blob allocator prefers DEVICE_LOCAL|HOST_VISIBLE
        // "unified" memory, which is mappable but WRITE-COMBINED, not HOST_CACHED.
        // Uploading to it is fast, but memcpy'ing OUT of it is uncached-read slow --
        // measured 0.33 GB/s against 10.3 GB/s for upload on Radeon 8060S. Route the
        // download through the staging allocator, which requests HOST_CACHED.
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
    }
    int cast_type_to = 0;
    if (vkdev->info.type() != 0)
    {
        cast_type_to = 1;
    }

    if (src.elemsize == src.elempack * 1u)
    {
        cast_type_to = 4;
    }

    VkMat dst_staging;
    vkdev->convert_packing(src, dst_staging, dst_elempack, cast_type_to, *this, opt_staging);

    // barrier device any @ compute to host-read @ compute
    if (dst_staging.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || dst_staging.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = dst_staging.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = dst_staging.buffer();
        barriers[0].offset = dst_staging.data->offset;
        barriers[0].size = dst_staging.data->capacity;

        VkPipelineStageFlags src_stage = dst_staging.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device host-read @ any
        dst_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    // create dst
    Mat dst_fp16;
    dst_fp16.create_like(dst_staging, opt.blob_allocator);
    if (dst_fp16.empty())
        return;

    // download
    d->download_post_buffers.push_back(dst_staging);
    d->download_post_mats_fp16.push_back(dst_fp16);

    // post memcpy device to dst
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_post_download;
        r.post_download.download_post_buffer_mat_offset = d->download_post_buffers.size() - 1;
        r.post_download.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
        d->delayed_records.push_back(r);
    }

    // cast to fp32 (discrete gpu)
    if (dst_fp16.elemsize == dst_fp16.elempack * 2u)
    {
        if (vkdev->info.type() == 0 && (opt.use_bf16_storage || opt.use_bf16_packed))
        {
            int dims = dst_fp16.dims;
            if (dims == 1)
                dst.create(dst_fp16.w, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 2)
                dst.create(dst_fp16.w, dst_fp16.h, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 3)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 4)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.d, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32;
            r.post_cast_bfloat16_to_float32.download_post_mat_bf16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_bfloat16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_bfloat16_to_float32.num_threads = opt.num_threads;
            d->delayed_records.push_back(r);
        }
        else if (vkdev->info.type() == 0 && (opt.use_fp16_storage || opt.use_fp16_packed))
        {
            int dims = dst_fp16.dims;
            if (dims == 1)
                dst.create(dst_fp16.w, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 2)
                dst.create(dst_fp16.w, dst_fp16.h, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 3)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 4)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.d, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_float16_to_float32;
            r.post_cast_float16_to_float32.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_float16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_float16_to_float32.num_threads = opt.num_threads;
            d->delayed_records.push_back(r);
        }
        else
        {
            dst = dst_fp16;
        }
    }
    else
    {
        dst = dst_fp16;
    }
}

void VkCompute::record_clone(const Mat& src, VkMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone host to buffer");

    // host to staging
    VkMat dst_staging;
    dst_staging.create_like(src, opt.staging_vkallocator);
    if (dst_staging.empty())
        return;

    // memcpy src to device
    for (int b = 0; b < src.n; b++)
    {
        const Mat src_b = src.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }
    dst_staging.allocator->flush(dst_staging.data);

    // mark device host-write @ null
    dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    // staging to device
    record_clone(dst_staging, dst, opt);

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_clone(const Mat& src, VkImageMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone host to image");

    // host to staging
    VkMat dst_staging;
    Option opt_staging = opt;
    opt_staging.blob_vkallocator = opt.staging_vkallocator;
    record_clone(src, dst_staging, opt_staging);

    // staging to image
    record_clone(dst_staging, dst, opt);

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_clone(const VkMat& src, Mat& dst, const Option& opt)
{
    if (d->ensure_weight_cache(src, &opt))
        return;

    //     NCNN_LOGE("record_clone buffer to host");

    if (!src.mapped_ptr() || is_deferred_buffer(src.data))
    {
        // device to staging
        VkMat src_staging;
        Option opt_staging = opt;
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
        record_clone(src, src_staging, opt_staging);

        // staging to host
        record_clone(src_staging, dst, opt);

        return;
    }

    // create dst
    dst.create_like(src, opt.blob_allocator);
    if (dst.empty())
        return;

    // barrier device any @ compute to host-read @ compute
    if (src.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.data->offset;
        barriers[0].size = src.data->capacity;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device host-read @ any
        src.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    // stash download post buffer and mat
    d->download_post_buffers.push_back(src);
    d->download_post_mats_fp16.push_back(dst);

    // post memcpy device to dst
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_post_download;
        r.post_download.download_post_buffer_mat_offset = d->download_post_buffers.size() - 1;
        r.post_download.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
        d->delayed_records.push_back(r);
    }
}

void VkCompute::record_clone(const VkImageMat& src, Mat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to host");

    // image to staging
    VkMat src_staging;
    Option opt_staging = opt;
    opt_staging.blob_vkallocator = opt.staging_vkallocator;
    record_clone(src, src_staging, opt_staging);

    // staging to host
    record_clone(src_staging, dst, opt);
}

void VkCompute::record_clone(const VkMat& src, VkMat& dst, const Option& opt)
{
    if (d->ensure_weight_cache(src, &opt))
        return;

    //     NCNN_LOGE("record_clone buffer to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    if (!is_deferred_buffer(src.data) && (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT))
    {
        // barrier device any @ compute to transfer-read @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.data->offset;
        barriers[0].size = src.data->capacity;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device transfer-read @ transfer
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    {
        // barrier device any @ null to transfer-write @ compute

        // mark device transfer-write @ transfer
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record device to staging
    {
        VkBufferCopy* regions = new VkBufferCopy[1];
        regions[0].srcOffset = src.buffer_offset();
        regions[0].dstOffset = dst.buffer_offset();
        regions[0].size = std::min(src.buffer_capacity(), dst.buffer_capacity());

        if (!d->delayed())
        {
            vkCmdCopyBuffer(d->compute_command_buffer, src.buffer(), dst.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer;
            r.copy_buffer.src = src.buffer();
            r.buffer_binding = VkBufferBinding(src);
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            d->delayed_records.push_back(r);
        }
    }
}

void VkCompute::record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // image layout transform any @ any to transfer-src-optimal @ compute
    if (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->image_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].oldLayout = src.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-src-optimal @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // image layout transform undefined @ null to transfer-dst-optimal @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = dst.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-dst-optimal @ compute
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record device to staging
    {
        VkImageCopy* regions = new VkImageCopy[1];
        regions[0].srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[0].srcSubresource.mipLevel = 0;
        regions[0].srcSubresource.baseArrayLayer = 0;
        regions[0].srcSubresource.layerCount = 1;
        regions[0].srcOffset.x = 0;
        regions[0].srcOffset.y = 0;
        regions[0].srcOffset.z = 0;
        regions[0].dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[0].dstSubresource.mipLevel = 0;
        regions[0].dstSubresource.baseArrayLayer = 0;
        regions[0].dstSubresource.layerCount = 1;
        regions[0].dstOffset.x = 0;
        regions[0].dstOffset.y = 0;
        regions[0].dstOffset.z = 0;
        regions[0].extent.width = src.data->width;
        regions[0].extent.height = src.data->height;
        regions[0].extent.depth = src.data->depth;

        if (!d->delayed())
        {
            vkCmdCopyImage(d->compute_command_buffer, src.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_image;
            r.copy_image.src = src.image();
            r.copy_image.src_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            r.copy_image.dst = dst.image();
            r.copy_image.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            r.copy_image.region_count = 1;
            r.copy_image.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&src.data->command_refcount, 1);
    NCNN_XADD(&dst.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(src.data);
    d->image_blocks_to_destroy.push_back(dst.data);
}

void VkCompute::record_clone(const VkMat& src, VkImageMat& dst, const Option& opt)
{
    if (d->ensure_weight_cache(src, &opt))
        return;

    //     NCNN_LOGE("record_clone buffer to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // barrier device any @ any to transfer-read @ compute
    if (!is_deferred_buffer(src.data) && (src.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT))
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.data->offset;
        barriers[0].size = src.data->capacity;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device transfer-read @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // image layout transform undefined @ null to transfer-dst-optimal @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = dst.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-dst-optimal @ compute
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record device to image
    {
        int region_count;
        VkBufferImageCopy* regions;
        if (dst.elemsize * dst.w * dst.h % 16 == 0)
        {
            region_count = 1;
            regions = new VkBufferImageCopy[1];
            regions[0].bufferOffset = src.buffer_offset();
            regions[0].bufferRowLength = 0;
            regions[0].bufferImageHeight = 0;
            regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[0].imageSubresource.mipLevel = 0;
            regions[0].imageSubresource.baseArrayLayer = 0;
            regions[0].imageSubresource.layerCount = 1;
            regions[0].imageOffset.x = 0;
            regions[0].imageOffset.y = 0;
            regions[0].imageOffset.z = 0;
            regions[0].imageExtent.width = dst.data->width;
            regions[0].imageExtent.height = dst.data->height;
            regions[0].imageExtent.depth = dst.data->depth;
        }
        else
        {
            region_count = dst.c;
            regions = new VkBufferImageCopy[region_count];
            for (int i = 0; i < region_count; i++)
            {
                regions[i].bufferOffset = src.buffer_offset() + src.cstep * src.elemsize * i;
                regions[i].bufferRowLength = 0;
                regions[i].bufferImageHeight = 0;
                regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                regions[i].imageSubresource.mipLevel = 0;
                regions[i].imageSubresource.baseArrayLayer = 0;
                regions[i].imageSubresource.layerCount = 1;
                regions[i].imageOffset.x = 0;
                regions[i].imageOffset.y = 0;
                regions[i].imageOffset.z = i;
                regions[i].imageExtent.width = dst.data->width;
                regions[i].imageExtent.height = dst.data->height;
                regions[i].imageExtent.depth = 1;
            }
        }

        if (!d->delayed())
        {
            vkCmdCopyBufferToImage(d->compute_command_buffer, src.buffer(), dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer_to_image;
            r.copy_buffer_to_image.src = src.buffer();
            r.buffer_binding = VkBufferBinding(src);
            r.copy_buffer_to_image.dst = dst.image();
            r.copy_buffer_to_image.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            r.copy_buffer_to_image.region_count = region_count;
            r.copy_buffer_to_image.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&dst.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(dst.data);
}

void VkCompute::record_clone(const VkImageMat& src, VkMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // image layout transform any @ any to transfer-src-optimal @ compute
    if (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->image_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].oldLayout = src.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-src-optimal @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    {
        // barrier device any @ null to transfer-write @ compute

        // mark device transfer-write @ transfer
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record image to device
    {
        int region_count;
        VkBufferImageCopy* regions;
        if (src.elemsize * src.w * src.h % 16 == 0)
        {
            region_count = 1;
            regions = new VkBufferImageCopy[1];
            regions[0].bufferOffset = dst.buffer_offset();
            regions[0].bufferRowLength = 0;
            regions[0].bufferImageHeight = 0;
            regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[0].imageSubresource.mipLevel = 0;
            regions[0].imageSubresource.baseArrayLayer = 0;
            regions[0].imageSubresource.layerCount = 1;
            regions[0].imageOffset.x = 0;
            regions[0].imageOffset.y = 0;
            regions[0].imageOffset.z = 0;
            regions[0].imageExtent.width = src.data->width;
            regions[0].imageExtent.height = src.data->height;
            regions[0].imageExtent.depth = src.data->depth;
        }
        else
        {
            region_count = src.c;
            regions = new VkBufferImageCopy[region_count];
            for (int i = 0; i < region_count; i++)
            {
                regions[i].bufferOffset = dst.buffer_offset() + dst.cstep * dst.elemsize * i;
                regions[i].bufferRowLength = 0;
                regions[i].bufferImageHeight = 0;
                regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                regions[i].imageSubresource.mipLevel = 0;
                regions[i].imageSubresource.baseArrayLayer = 0;
                regions[i].imageSubresource.layerCount = 1;
                regions[i].imageOffset.x = 0;
                regions[i].imageOffset.y = 0;
                regions[i].imageOffset.z = i;
                regions[i].imageExtent.width = src.data->width;
                regions[i].imageExtent.height = src.data->height;
                regions[i].imageExtent.depth = 1;
            }
        }

        if (!d->delayed())
        {
            vkCmdCopyImageToBuffer(d->compute_command_buffer, src.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer(), region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_image_to_buffer;
            r.copy_image_to_buffer.src = src.image();
            r.copy_image_to_buffer.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            r.copy_image_to_buffer.dst = dst.buffer();
            r.copy_image_to_buffer.region_count = region_count;
            r.copy_image_to_buffer.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&src.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(src.data);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    record_pipeline(pipeline, bindings, std::vector<VkImageMat>(), constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkImageMat>& bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher)
{
    record_pipeline(pipeline, std::vector<VkMat>(), bindings, constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher)
{
    //     NCNN_LOGE("record_pipeline %p", pipeline);

    for (size_t i = 0; i < buffer_bindings.size(); i++)
    {
        if (!buffer_bindings[i].empty() && d->ensure_weight_cache(buffer_bindings[i]))
            return;
    }

    const int buffer_binding_count = (int)buffer_bindings.size();
    const int image_binding_count = (int)image_bindings.size();
    const int constant_count = (int)constants.size();

    const int binding_count = buffer_binding_count + image_binding_count;
    const ShaderInfo& shader_info = pipeline->shader_info();

    if (binding_count != shader_info.binding_count)
    {
        NCNN_LOGE("binding_count not match, expect %d but got %d + %d", shader_info.binding_count, buffer_binding_count, image_binding_count);
    }

    if (constant_count != shader_info.push_constant_count)
    {
        NCNN_LOGE("push_constant_count not match, expect %d but got %d", shader_info.push_constant_count, constant_count);
    }

    int buffer_index = 0;
    int image_index = 0;
    for (int i = 0; i < binding_count; i++)
    {
        int binding_type = shader_info.binding_types[i];

        if (binding_type == 1)
        {
            const VkMat& binding = buffer_bindings[buffer_index++];
            // Empty descriptors use device-owned placeholders below. They are
            // not accessed by this dispatch and have no mutable command state.
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d buffer = %d %d %d %d @ %lu %d = %p +%ld ~%ld", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.buffer(), binding.buffer_offset(), binding.buffer_capacity());

            barrier_readwrite(binding);
        }
        else if (binding_type == 2)
        {
            const VkImageMat& binding = image_bindings[image_index++];
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d image = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            barrier_readwrite(binding);

            // image and imageview can not be destroyed until command execution ends
            NCNN_XADD(&binding.data->command_refcount, 1);
            d->image_blocks_to_destroy.push_back(binding.data);
        }
        else // if (binding_type == 3)
        {
            const VkImageMat& binding = image_bindings[image_index++];
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d sampler = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            // if the same image used for both storage image and combined image sampler
            // only apply image layout transition to general
            bool image_read_write = false;
            for (int j = 0; j < image_binding_count; j++)
            {
                if (shader_info.binding_types[j] == 2 && binding.data == image_bindings[j].data)
                {
                    // the same image is used as storage image, skip it
                    image_read_write = true;
                    break;
                }
            }
            if (image_read_write)
                continue;

            barrier_readonly(binding);

            // image and imageview can not be destroyed until command execution ends
            NCNN_XADD(&binding.data->command_refcount, 1);
            d->image_blocks_to_destroy.push_back(binding.data);
        }
    }

    if (d->delayed())
    {
        VkComputePrivate::PipelineRecord p;
        p.pipeline = pipeline->pipeline();
        p.pipeline_layout = pipeline->pipeline_layout();
        p.descriptorset_layout = pipeline->descriptorset_layout();
        p.descriptor_update_template = pipeline->descriptor_update_template();
        p.shader_info = shader_info;
        for (int i = 0; i < buffer_binding_count; i++)
        {
            const VkMat& binding = buffer_bindings[i].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[i];
            p.buffers.push_back(VkBufferBinding(binding));
        }
        for (int i = 0; i < image_binding_count; i++)
        {
            const VkImageMat& binding = image_bindings[i].empty() ? vkdev->get_dummy_image() : image_bindings[i];
            VkDescriptorImageInfo info;
            info.sampler = 0;
            info.imageView = binding.imageview();
            info.imageLayout = binding.data->image_layout;
            p.images.push_back(info);
        }
        p.constants = constants;
        p.group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        p.group_count_y = (dispatcher.h * (dispatcher.d ? dispatcher.d : 1) + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        p.group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();
        d->pending_dispatch_total += (uint64_t)p.group_count_x * p.group_count_y * p.group_count_z;
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_pipeline;
        r.pipeline_index = d->pipelines.size();
        d->pipelines.push_back(p);
        d->delayed_records.push_back(r);
        return;
    }

    // record bind pipeline
    {
        vkCmdBindPipeline(d->compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
        vkdev->bind_shader_coverage(d->compute_command_buffer, pipeline->pipeline_layout());
#endif
    }

    // record update bindings
    if (binding_count > 0)
    {
        std::vector<unsigned char> descriptorInfos;
        {
            descriptorInfos.resize(sizeof(VkDescriptorBufferInfo) * buffer_binding_count + sizeof(VkDescriptorImageInfo) * image_binding_count);

            unsigned char* p_descriptorInfos = descriptorInfos.data();
            int descriptorBufferInfo_index = 0;
            int descriptorImageInfo_index = 0;
            for (int i = 0; i < binding_count; i++)
            {
                int binding_type = shader_info.binding_types[i];

                if (binding_type == 1)
                {
                    const VkMat& binding = buffer_bindings[descriptorBufferInfo_index].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[descriptorBufferInfo_index];
                    descriptorBufferInfo_index++;

                    VkDescriptorBufferInfo descriptorBufferInfo;
                    descriptorBufferInfo.buffer = binding.buffer();
                    descriptorBufferInfo.offset = binding.buffer_offset();
                    descriptorBufferInfo.range = binding.buffer_capacity();

                    memcpy(p_descriptorInfos, &descriptorBufferInfo, sizeof(VkDescriptorBufferInfo));
                    p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                }
                else //if (binding_type == 2 || binding_type == 3)
                {
                    const VkImageMat& binding = image_bindings[descriptorImageInfo_index].empty() ? vkdev->get_dummy_image() : image_bindings[descriptorImageInfo_index];
                    descriptorImageInfo_index++;

                    // we always use immutable nearest sampler set in descroptor layout during pipeline creation
                    VkDescriptorImageInfo descriptorImageInfo;
                    descriptorImageInfo.sampler = 0;
                    descriptorImageInfo.imageView = binding.imageview();
                    descriptorImageInfo.imageLayout = binding.data->image_layout;

                    memcpy(p_descriptorInfos, &descriptorImageInfo, sizeof(VkDescriptorImageInfo));
                    p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                }
            }
        }

        vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffer, pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, descriptorInfos.data());
    }

    // record push constants
    if (constant_count > 0)
    {
        vkCmdPushConstants(d->compute_command_buffer, pipeline->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dispatcher.h * (dispatcher.d ? dispatcher.d : 1) + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);

        d->pending_dispatch_total += group_count_x * group_count_y * group_count_z;
    }
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (!d->delayed())
    {
        if (d->query_pool)
            vkCmdWriteTimestamp(d->compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->query_pool, query);
    }
    else
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_write_timestamp;
        r.write_timestamp.query = query;
        d->delayed_records.push_back(r);
    }
}
#endif // NCNN_BENCHMARK

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
void VkCompute::record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst)
{
    // image layout transform undefined @ null to general @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }
    }

    // record bind pipeline
    {
        if (!d->delayed())
        {
            vkCmdBindPipeline(d->compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
            vkdev->bind_shader_coverage(d->compute_command_buffer, pipeline->pipeline_layout());
#endif
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_bind_pipeline;
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline();
#if NCNN_COVERAGE
            r.bind_pipeline.pipeline_layout = pipeline->pipeline_layout();
#endif
            d->delayed_records.push_back(r);
        }
    }

    // record update bindings
    {
        VkDescriptorImageInfo descriptorImageInfo;
        descriptorImageInfo.sampler = pipeline->sampler;
        descriptorImageInfo.imageView = src.imageview();
        descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = dst.buffer();
        descriptorBufferInfo.offset = dst.buffer_offset();
        descriptorBufferInfo.range = dst.total() * dst.elemsize;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            struct ImportAndroidHardwareBufferDescriptorInfo
            {
                VkDescriptorImageInfo imageInfo;
                VkDescriptorBufferInfo bufferInfo;
                VkDescriptorBufferInfo buffer4Info;
            };

            ImportAndroidHardwareBufferDescriptorInfo info;
            info.imageInfo = descriptorImageInfo;
            info.bufferInfo = descriptorBufferInfo;
            info.buffer4Info = descriptorBufferInfo;

            if (d->weight_recording)
            {
                VkComputePrivate::record r;
                r.type = VkComputePrivate::record::TYPE_push_descriptors;
                r.push_descriptors.update_template = pipeline->descriptor_update_template();
                r.push_descriptors.pipeline_layout = pipeline->pipeline_layout();
                unsigned char* data = new unsigned char[sizeof(info)];
                memcpy(data, &info, sizeof(info));
                r.push_descriptors.data = data;
                d->delayed_records.push_back(r);
            }
            else
            {
                vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffer, pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, &info);
            }
        }
        else
        {
            // create new descriptor_pool and descriptorset
            VkDescriptorPool descriptor_pool;
            {
                VkDescriptorPoolSize poolSizes[2];
                poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                poolSizes[0].descriptorCount = 1;
                poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSizes[1].descriptorCount = 2;

                VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
                descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCreateInfo.pNext = 0;
                descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                descriptorPoolCreateInfo.maxSets = 1;
                descriptorPoolCreateInfo.poolSizeCount = 2;
                descriptorPoolCreateInfo.pPoolSizes = poolSizes;

                VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
                    return;
                }
            }
            d->descriptor_pools.push_back(descriptor_pool);

            VkDescriptorSet descriptorset;
            {
                VkDescriptorSetLayout descriptorset_layout = pipeline->descriptorset_layout();

                VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
                descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocateInfo.pNext = 0;
                descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
                descriptorSetAllocateInfo.descriptorSetCount = 1;
                descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

                VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
                    return;
                }
            }
            d->descriptorsets.push_back(descriptorset);

            if (vkdev->info.support_VK_KHR_descriptor_update_template())
            {
                struct ImportAndroidHardwareBufferDescriptorInfo
                {
                    VkDescriptorImageInfo imageInfo;
                    VkDescriptorBufferInfo bufferInfo;
                    VkDescriptorBufferInfo buffer4Info;
                };

                ImportAndroidHardwareBufferDescriptorInfo info;
                info.imageInfo = descriptorImageInfo;
                info.bufferInfo = descriptorBufferInfo;
                info.buffer4Info = descriptorBufferInfo;

                vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template(), &info);
            }
            else
            {
                VkWriteDescriptorSet writeDescriptorSets[3];
                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].pNext = 0;
                writeDescriptorSets[0].dstSet = descriptorset;
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].dstArrayElement = 0;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[0].pImageInfo = &descriptorImageInfo;
                writeDescriptorSets[0].pBufferInfo = 0;
                writeDescriptorSets[0].pTexelBufferView = 0;
                writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[1].pNext = 0;
                writeDescriptorSets[1].dstSet = descriptorset;
                writeDescriptorSets[1].dstBinding = 1;
                writeDescriptorSets[1].dstArrayElement = 0;
                writeDescriptorSets[1].descriptorCount = 1;
                writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeDescriptorSets[1].pImageInfo = 0;
                writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfo;
                writeDescriptorSets[1].pTexelBufferView = 0;
                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].pNext = 0;
                writeDescriptorSets[2].dstSet = descriptorset;
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].dstArrayElement = 0;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeDescriptorSets[2].pImageInfo = 0;
                writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfo;
                writeDescriptorSets[2].pTexelBufferView = 0;

                vkUpdateDescriptorSets(vkdev->vkdevice(), 3, writeDescriptorSets, 0, 0);
            }

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_bind_descriptorsets;
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout();
            r.bind_descriptorsets.descriptorset_count = 1;
            r.bind_descriptorsets.descriptorset_offset = d->descriptorsets.size() - 1;
            d->delayed_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dst.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dst.h + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dst.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        if (!d->delayed())
        {
            vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_dispatch;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            d->delayed_records.push_back(r);
        }
    }
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

int VkComputePrivate::replay_record(const record& r, VkCommandBuffer command)
{
    switch (r.type)
    {
    case VkComputePrivate::record::TYPE_pipeline:
        return emit_pipeline(pipelines[r.pipeline_index], command);
    case VkComputePrivate::record::TYPE_copy_buffer:
    {
        if (!weight_cache || !r.buffer_binding.source)
        {
            vkCmdCopyBuffer(command, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
            break;
        }
        VkBuffer src = r.copy_buffer.src;
        std::vector<VkBufferCopy> regions(r.copy_buffer.region_count);
        for (uint32_t i = 0; i < r.copy_buffer.region_count; i++)
            regions[i] = r.copy_buffer.regions[i];
        if (weight_cache && r.buffer_binding.source)
        {
            VkDescriptorBufferInfo descriptor;
            if (resolve_weight_binding(r.buffer_binding, descriptor))
                return -1;
            src = descriptor.buffer;
            for (uint32_t i = 0; i < r.copy_buffer.region_count; i++)
                regions[i].srcOffset = descriptor.offset + regions[i].srcOffset - r.buffer_binding.descriptor.offset;
        }
        vkCmdCopyBuffer(command, src, r.copy_buffer.dst, r.copy_buffer.region_count, regions.data());
        break;
    }
    case VkComputePrivate::record::TYPE_copy_image:
    {
        vkCmdCopyImage(command, r.copy_image.src, r.copy_image.src_layout, r.copy_image.dst, r.copy_image.dst_layout, r.copy_image.region_count, r.copy_image.regions);
        break;
    }
    case VkComputePrivate::record::TYPE_copy_buffer_to_image:
    {
        if (!weight_cache || !r.buffer_binding.source)
        {
            vkCmdCopyBufferToImage(command, r.copy_buffer_to_image.src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, r.copy_buffer_to_image.regions);
            break;
        }
        VkBuffer src = r.copy_buffer_to_image.src;
        std::vector<VkBufferImageCopy> regions(r.copy_buffer_to_image.region_count);
        for (uint32_t i = 0; i < r.copy_buffer_to_image.region_count; i++)
            regions[i] = r.copy_buffer_to_image.regions[i];
        if (weight_cache && r.buffer_binding.source)
        {
            VkDescriptorBufferInfo descriptor;
            if (resolve_weight_binding(r.buffer_binding, descriptor))
                return -1;
            src = descriptor.buffer;
            for (uint32_t i = 0; i < r.copy_buffer_to_image.region_count; i++)
                regions[i].bufferOffset = descriptor.offset + regions[i].bufferOffset - r.buffer_binding.descriptor.offset;
        }
        vkCmdCopyBufferToImage(command, src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, regions.data());
        break;
    }
    case VkComputePrivate::record::TYPE_copy_image_to_buffer:
    {
        vkCmdCopyImageToBuffer(command, r.copy_image_to_buffer.src, r.copy_image_to_buffer.layout, r.copy_image_to_buffer.dst, r.copy_image_to_buffer.region_count, r.copy_image_to_buffer.regions);
        break;
    }
    case VkComputePrivate::record::TYPE_bind_pipeline:
    {
        vkCmdBindPipeline(command, r.bind_pipeline.bind_point, r.bind_pipeline.pipeline);
#if NCNN_COVERAGE
        vkdev->bind_shader_coverage(command, r.bind_pipeline.pipeline_layout);
#endif
        break;
    }
    case VkComputePrivate::record::TYPE_bind_descriptorsets:
    {
        vkCmdBindDescriptorSets(command, r.bind_descriptorsets.bind_point, r.bind_descriptorsets.pipeline_layout, 0, r.bind_descriptorsets.descriptorset_count, &descriptorsets[r.bind_descriptorsets.descriptorset_offset], 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_push_descriptors:
    {
        vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command, r.push_descriptors.update_template, r.push_descriptors.pipeline_layout, 0, r.push_descriptors.data);
        break;
    }
    case VkComputePrivate::record::TYPE_push_constants:
    {
        vkCmdPushConstants(command, r.push_constants.pipeline_layout, r.push_constants.stage_flags, 0, r.push_constants.size, r.push_constants.values);
        break;
    }
    case VkComputePrivate::record::TYPE_dispatch:
    {
        vkCmdDispatch(command, r.dispatch.group_count_x, r.dispatch.group_count_y, r.dispatch.group_count_z);
        break;
    }
    case VkComputePrivate::record::TYPE_memory_barrers:
    {
        vkCmdPipelineBarrier(command, r.memory_barrers.src_stage, r.memory_barrers.dst_stage, 0, r.memory_barrers.barrier_count, r.memory_barrers.barriers, 0, 0, 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_buffer_barrers:
    {
        vkCmdPipelineBarrier(command, r.buffer_barrers.src_stage, r.buffer_barrers.dst_stage, 0, 0, 0, r.buffer_barrers.barrier_count, r.buffer_barrers.barriers, 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_image_barrers:
    {
        vkCmdPipelineBarrier(command, r.image_barrers.src_stage, r.image_barrers.dst_stage, 0, 0, 0, 0, 0, r.image_barrers.barrier_count, r.image_barrers.barriers);
        break;
    }
#if NCNN_BENCHMARK
    case VkComputePrivate::record::TYPE_write_timestamp:
    {
        if (query_pool)
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, r.write_timestamp.query);
        break;
    }
#endif // NCNN_BENCHMARK
    case VkComputePrivate::record::TYPE_post_download:
    case VkComputePrivate::record::TYPE_post_cast_float16_to_float32:
    case VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32:
    default:
        break;
    }
    return 0;
}

int VkCompute::submit_and_wait()
{
    //     NCNN_LOGE("submit_and_wait");

    if (d->record_error)
        return d->record_error;
    bool managed = false;
    if (d->weight_recording)
    {
        for (size_t i = 0; i < d->delayed_records.size() && !managed; i++)
        {
            const VkComputePrivate::record& r = d->delayed_records[i];
            if (r.type == VkComputePrivate::record::TYPE_pipeline)
            {
                const std::vector<VkBufferBinding>& buffers = d->pipelines[r.pipeline_index].buffers;
                for (size_t j = 0; j < buffers.size(); j++)
                    managed |= buffers[j].source != 0;
            }
            else
                managed = r.buffer_binding.source != 0;
        }
    }
    if (managed)
    {
        int ret = d->submit_managed();
        if (ret != 0)
            return ret;
    }
    else
    {
        if (!vkdev->info.support_VK_KHR_push_descriptor() || d->weight_recording)
        {
            if (!vkdev->info.support_VK_KHR_push_descriptor())
            {
                d->begin_command_buffer();

#if NCNN_BENCHMARK
                if (d->query_pool)
                    vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
#endif // NCNN_BENCHMARK
            }

            const size_t record_count = d->delayed_records.size();

            // handle delayed records
            for (size_t i = 0; i < record_count; i++)
            {
                const VkComputePrivate::record& r = d->delayed_records[i];

                int ret = d->replay_record(r, d->compute_command_buffer);
                if (ret != 0)
                    return ret;
            }
        }

        // end command buffer
        {
            d->end_command_buffer();
        }

        // acquire queue and reclaim on return
        VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
        if (compute_queue == 0)
        {
            NCNN_LOGE("out of compute queue");
            return -1;
        }

        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &d->compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, d->compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return -1;
            }
        }

        vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);

        // wait
        {
            VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &d->compute_command_fence, VK_TRUE, (uint64_t)-1);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkWaitForFences failed %d", ret);
                return -1;
            }
        }
    }

    // handle delayed post records
    for (size_t i = 0; i < d->delayed_records.size(); i++)
    {
        const VkComputePrivate::record& r = d->delayed_records[i];

        switch (r.type)
        {
        case VkComputePrivate::record::TYPE_post_download:
        {
            const VkMat& src = d->download_post_buffers[r.post_download.download_post_buffer_mat_offset];
            Mat& dst = d->download_post_mats_fp16[r.post_download.download_post_mat_fp16_offset];

            // NCNN_LOGE("post_download  %p +%d ~%d  -> %p", src.buffer(), src.buffer_offset(), src.buffer_capacity(), dst.data);

            src.allocator->invalidate(src.data);
            for (int b = 0; b < dst.n; b++)
            {
                const VkMat src_b = src.batch(b);
                Mat dst_b = dst.batch(b);
                memcpy(dst_b.data, src_b.mapped_ptr(), dst_b.total() * dst_b.elemsize);
            }
            break;
        }
        case VkComputePrivate::record::TYPE_post_cast_float16_to_float32:
        {
            // NCNN_LOGE("post_cast_float16_to_float32");

            const Mat& src = d->download_post_mats_fp16[r.post_cast_float16_to_float32.download_post_mat_fp16_offset];
            Mat& dst = d->download_post_mats[r.post_cast_float16_to_float32.download_post_mat_offset];

            Option opt;
            opt.num_threads = r.post_cast_float16_to_float32.num_threads;
            opt.blob_allocator = dst.allocator;
            ncnn::cast_float16_to_float32(src, dst, opt);
            break;
        }
        case VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32:
        {
            // NCNN_LOGE("post_cast_bfloat16_to_float32");

            const Mat& src = d->download_post_mats_fp16[r.post_cast_bfloat16_to_float32.download_post_mat_bf16_offset];
            Mat& dst = d->download_post_mats[r.post_cast_bfloat16_to_float32.download_post_mat_offset];

            Option opt;
            opt.num_threads = r.post_cast_bfloat16_to_float32.num_threads;
            opt.blob_allocator = dst.allocator;
            ncnn::cast_bfloat16_to_float32(src, dst, opt);
            break;
        }
        default:
            break;
        }
    }

#if NCNN_BENCHMARK
    d->query_results_ready = true;
#endif
    d->clear_records();

    d->pending_dispatch_total = 0;

    return 0;
}

int VkCompute::reset()
{
    if (d->executing)
        return -1;
#if NCNN_BENCHMARK
    d->query_results_ready = false;
#endif
    d->record_error = d->weight_config_error;

    d->upload_staging_buffers.clear();
    d->download_post_buffers.clear();
    d->download_post_mats_fp16.clear();
    d->download_post_mats.clear();

    for (size_t i = 0; i < d->image_blocks_to_destroy.size(); i++)
    {
        VkImageMemory* ptr = d->image_blocks_to_destroy[i];

        int old_command_refcount = NCNN_XADD(&ptr->command_refcount, -1);
        if (ptr->refcount == 0 && old_command_refcount == 1)
        {
            // no userspace reference and we are the last command reference
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }
        else
        {
            // reference exists in user code or other command
        }
    }
    d->image_blocks_to_destroy.clear();

    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        for (size_t i = 0; i < d->descriptor_pools.size(); i++)
        {
            vkDestroyDescriptorPool(vkdev->vkdevice(), d->descriptor_pools[i], 0);
        }
        d->descriptor_pools.clear();
        d->descriptorsets.clear();
    }

    d->clear_records();
    // Failed uploads may leave synchronization primitives or cached regions
    // unusable. Healthy working sets remain owned by this command.
    if (d->weight_cache && d->weight_cache->error)
    {
        d->destroy_weight_commands();
        d->release_weight_cache();
    }
    d->weight_recording = false;

    d->pending_dispatch_total = 0;

    // reset command buffer and fence
    {
        VkResult ret = vkResetCommandBuffer(d->compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            return -1;
        }
    }
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &d->compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return -1;
        }
    }

    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        const int ret = d->begin_command_buffer();
        if (ret)
            return ret;

#if NCNN_BENCHMARK
        if (d->query_pool)
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkCompute::clear_weight_cache()
{
    if (d->executing || !d->delayed_records.empty() || d->pending_dispatch_total)
    {
        NCNN_LOGE("cannot clear a weight working set while recording or executing");
        return -1;
    }
    d->destroy_weight_commands();
    d->release_weight_cache();
    d->weight_recording = false;
    return 0;
}

uint64_t VkCompute::pending_dispatch_total() const
{
    return d->pending_dispatch_total;
}

#if NCNN_BENCHMARK
int VkCompute::create_query_pool(uint32_t _query_count)
{
    d->query_count = _query_count;

    VkQueryPoolCreateInfo queryPoolCreateInfo;
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.pNext = 0;
    queryPoolCreateInfo.flags = 0;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = d->query_count;
    queryPoolCreateInfo.pipelineStatistics = 0;

    VkResult ret = vkCreateQueryPool(vkdev->vkdevice(), &queryPoolCreateInfo, 0, &d->query_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateQueryPool failed %d", ret);
        return -1;
    }

    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        if (d->query_pool)
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
    }

    return 0;
}

int VkCompute::get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results)
{
    if (!d->query_results_ready || !d->query_pool || first_query > d->query_count || query_count > d->query_count - first_query)
        return -1;
    if (results.size() < first_query + query_count)
    {
        NCNN_LOGE("results not large enough");
        return -1;
    }

    VkResult ret = vkGetQueryPoolResults(vkdev->vkdevice(), d->query_pool, first_query, query_count,
                                         query_count * sizeof(uint64_t), results.data() + first_query, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (ret != VK_SUCCESS && ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetQueryPoolResults failed %d", ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_BENCHMARK

void VkCompute::barrier_readwrite(const VkMat& binding)
{
    if (is_deferred_buffer(binding.data))
        return;

    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // barrier device any @ compute/null to shader-readwrite @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = binding.buffer();
        barriers[0].offset = binding.data->offset;
        barriers[0].size = binding.data->capacity;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device shader-readwrite @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

void VkCompute::barrier_readwrite(const VkImageMat& binding)
{
    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_GENERAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // image layout transform any @ any to shader-write @ compute
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].oldLayout = binding.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = binding.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image shader-write @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        binding.data->image_layout = VK_IMAGE_LAYOUT_GENERAL;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

void VkCompute::barrier_readonly(const VkImageMat& binding)
{
    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // image layout transform any @ any to shader-readonly-optimal @ compute
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout = binding.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = binding.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (!d->delayed())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image shader-readonly-optimal @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        binding.data->image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

void VkComputePrivate::destroy_weight_commands()
{
    if (!weight_commands_initialized)
        return;
    if (executing)
        finish_weight_commands(weight_cache ? weight_cache->error : -1);
#if NCNN_BENCHMARK
    if (timing_pool)
        vkDestroyQueryPool(vkdev->vkdevice(), timing_pool, 0);
    timing_pool = 0;
#endif
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        Upload& u = uploads[i];
        vkDestroyFence(vkdev->vkdevice(), u.fence, 0);
        vkDestroyFence(vkdev->vkdevice(), u.release_fence, 0);
        vkDestroySemaphore(vkdev->vkdevice(), u.ready, 0);
        vkDestroySemaphore(vkdev->vkdevice(), u.returned, 0);
        if (u.release_command)
            vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &u.release_command);
        u = Upload();
    }
    for (int i = 0; i < weight_compute_slots; i++)
    {
        Compute& c = computes[i];
        vkDestroyFence(vkdev->vkdevice(), c.fence, 0);
        if (c.command)
            vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &c.command);
        c = Compute();
    }
    vkDestroyFence(vkdev->vkdevice(), handoff_fence, 0);
    handoff_fence = 0;
    if (transfer_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_pool, 0);
    transfer_pool = 0;
    prefix_fence = 0;
    next_upload = next_compute = 0;
    current_compute = -1;
    compute_serial = completed_compute_serial = 0;
    weight_commands_initialized = false;
    pending_releases.clear();
    pending_uploads.clear();
    current_sources.clear();
}

static int weight_command(VkDevice device, VkCommandPool pool, VkCommandBuffer& command)
{
    VkResult ret;
    VkCommandBufferAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    ret = vkAllocateCommandBuffers(device, &info, &command);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
        return -1;
    }
    return 0;
}

int VkComputePrivate::create_weight_commands()
{
    VkResult ret;
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = weight_transfer_family;
    ret = vkCreateCommandPool(vkdev->vkdevice(), &pool_info, 0, &transfer_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateCommandPool failed %d", ret);
        return -1;
    }
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (!weight_cache->handoff)
    {
        ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &weight_cache->handoff);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return -1;
        }
    }
    ret = vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &handoff_fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateFence failed %d", ret);
        return -1;
    }
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        Upload& u = uploads[i];
        if (weight_command(vkdev->vkdevice(), transfer_pool, u.command))
            return -1;
        if (weight_command(vkdev->vkdevice(), compute_command_pool, u.release_command))
            return -1;
        ret = vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &u.fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
        ret = vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &u.release_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
        ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &u.ready);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return -1;
        }
        ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &u.returned);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return -1;
        }
    }
    for (int i = 0; i < weight_compute_slots; i++)
    {
        if (weight_command(vkdev->vkdevice(), compute_command_pool, computes[i].command))
            return -1;
        ret = vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &computes[i].fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }
#if NCNN_BENCHMARK
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkdev->info.physicalDevice(), &count, 0);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(vkdev->info.physicalDevice(), &count, families.data());
    transfer_timestamp_bits = families[weight_transfer_family].timestampValidBits;
    // enable timing only for queues that can reset queries in vulkan 1.0
    // avoid extra submissions or dependencies just for measurements
    if (!(families[weight_transfer_family].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)))
        transfer_timestamp_bits = 0;
    weight_cache->stats.upload_time_available = transfer_timestamp_bits != 0;
    compute_timestamp_bits = families[vkdev->info.compute_queue_family_index()].timestampValidBits;
    if (transfer_timestamp_bits || compute_timestamp_bits)
    {
        VkQueryPoolCreateInfo query_info = {};
        query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2 * (weight_transfer_slots + weight_compute_slots);
        ret = vkCreateQueryPool(vkdev->vkdevice(), &query_info, 0, &timing_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateQueryPool failed %d", ret);
            return -1;
        }
    }
#endif
    return 0;
}

int VkComputePrivate::allocate_weight_cache(size_t minimum_device_size, size_t max_source_size, size_t max_external_size)
{
    weight_cache->unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    const size_t max_size = max_source_size;
    const size_t max_file_size = max_external_size;
    if (weight_cache->unified)
    {
        // unified weights are filled by the host and consumed directly by compute
        weight_transfer_family = vkdev->info.compute_queue_family_index();
        weight_queue_mode = 2;
    }
    if (max_size > (size_t)-1 / 2)
        return -1;
    const size_t device_budget = weight_cache->opt.weight_device_budget;
    const size_t host_budget = weight_cache->opt.weight_host_budget;
    if (!weight_cache->device_size)
        weight_cache->device_size = std::max(max_size * 2, minimum_device_size);
    else if (weight_cache->automatic_device_size)
        weight_cache->device_size = std::max(weight_cache->device_size, minimum_device_size);
    if (max_file_size && !weight_cache->unified)
    {
        if (!weight_cache->host_size)
            weight_cache->host_size = max_file_size * 2;
    }
    else
    {
        weight_cache->host_size = 0;
    }
    if (weight_cache->device_size < minimum_device_size || (max_file_size && !weight_cache->unified && weight_cache->host_size < weight_cache->alignment))
    {
        NCNN_LOGE("weight exchange too small: host %llu, device %llu needs %llu", (unsigned long long)weight_cache->host_size, (unsigned long long)weight_cache->device_size, (unsigned long long)minimum_device_size);
        return -1;
    }
    if (weight_cache->host_size > (size_t)-1 - weight_cache->alignment || weight_cache->device_size > (size_t)-1 - weight_cache->alignment)
        return -1;
    weight_cache->host_size = alignSize(weight_cache->host_size, weight_cache->alignment);
    weight_cache->device_size = alignSize(weight_cache->device_size, weight_cache->alignment);
    if (!max_size)
        return 0;

    size_t device_allocation;
    size_t host_allocation;
    if (weight_buffer_memory_size(vkdev, weight_cache->device_size, device_allocation)
            || weight_buffer_memory_size(vkdev, weight_cache->host_size, host_allocation, true)
            || (device_budget && device_allocation > device_budget)
            || (host_budget && host_allocation > host_budget))
    {
        NCNN_LOGE("weight exchange allocation exceeds the remaining residency budget");
        return -1;
    }
    weight_cache->device_allocator = weight_cache->opt.weight_device_vkallocator;
    if (weight_cache->device_allocator)
        weight_cache->device_arena = weight_cache->device_allocator->fastMalloc(weight_cache->device_size);
    else if (create_weight_buffer(vkdev, weight_cache->device_size, 0, weight_cache->unified, device_budget, weight_cache->device_arena, device_allocation))
        return -100;
    const size_t allocation_alignment = std::max(weight_cache->alignment, vkdev->info.memory_map_alignment());
    const VkBufferUsageFlags device_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!weight_cache->device_arena || !weight_cache->device_arena->buffer || !weight_cache->device_arena->memory || weight_cache->device_arena->capacity < weight_cache->device_size
            || weight_cache->device_arena->capacity - weight_cache->device_size >= allocation_alignment || weight_cache->device_arena->offset % weight_cache->alignment != 0
            || weight_cache->device_arena->storage_type != 0
            || weight_cache->device_arena->memory_type_index >= vkdev->info.physicalDeviceMemoryProperties().memoryTypeCount
            || !vkdev->is_device_local(weight_cache->device_arena->memory_type_index) || (weight_cache->device_arena->buffer_usage & device_usage) != device_usage
            || (weight_cache->unified && (!weight_cache->device_arena->mapped_ptr || !vkdev->is_mappable(weight_cache->device_arena->memory_type_index))))
    {
        NCNN_LOGE("weight device allocator must provide a compatible device-local buffer");
        return -100;
    }
    if (weight_cache->host_size)
    {
        weight_cache->host_allocator = weight_cache->opt.weight_host_vkallocator;
        if (weight_cache->host_allocator)
            weight_cache->host_arena = weight_cache->host_allocator->fastMalloc(weight_cache->host_size);
        else if (create_weight_buffer(vkdev, weight_cache->host_size, 1, false, host_budget, weight_cache->host_arena, host_allocation))
            return -100;
        if (!weight_cache->host_arena || !weight_cache->host_arena->buffer || !weight_cache->host_arena->memory || !weight_cache->host_arena->mapped_ptr
                || weight_cache->host_arena->memory_type_index >= vkdev->info.physicalDeviceMemoryProperties().memoryTypeCount
                || !vkdev->is_mappable(weight_cache->host_arena->memory_type_index) || weight_cache->host_arena->capacity < weight_cache->host_size
                || weight_cache->host_arena->capacity - weight_cache->host_size >= allocation_alignment || weight_cache->host_arena->offset % weight_cache->alignment != 0
                || !(weight_cache->host_arena->buffer_usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        {
            NCNN_LOGE("weight host allocator must provide a mappable transfer buffer");
            return -100;
        }
        const VkPhysicalDeviceMemoryProperties& properties = vkdev->info.physicalDeviceMemoryProperties();
        if (host_budget && (properties.memoryHeaps[properties.memoryTypes[weight_cache->host_arena->memory_type_index].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
        {
            NCNN_LOGE("weight host arena must use host memory within the explicit host budget");
            return -100;
        }
    }
#if NCNN_BENCHMARK
    weight_cache->stats.host_capacity = weight_cache->host_size;
    weight_cache->stats.device_capacity = weight_cache->device_size;
    weight_cache->stats.unified_memory = weight_cache->unified;
#endif
    // custom arenas can refer to a larger backing allocation
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), weight_cache->device_arena->buffer, &requirements);
    if (weight_cache->device_arena->offset > requirements.size || weight_cache->device_size > requirements.size - weight_cache->device_arena->offset
            || !(requirements.memoryTypeBits & ((uint32_t)1 << weight_cache->device_arena->memory_type_index))
            || (device_budget && requirements.size > device_budget))
    {
        NCNN_LOGE("weight device backing does not satisfy the arena range or budget");
        return -100;
    }
#if NCNN_BENCHMARK
    weight_cache->stats.device_allocation_size = (size_t)requirements.size;
#endif
    if (weight_cache->host_arena)
    {
        vkGetBufferMemoryRequirements(vkdev->vkdevice(), weight_cache->host_arena->buffer, &requirements);
#if NCNN_BENCHMARK
        weight_cache->stats.host_allocation_size = (size_t)requirements.size;
#endif
        if (weight_cache->host_arena->offset > requirements.size || weight_cache->host_size > requirements.size - weight_cache->host_arena->offset
                || !(requirements.memoryTypeBits & ((uint32_t)1 << weight_cache->host_arena->memory_type_index))
                || (host_budget && requirements.size > host_budget))
        {
            NCNN_LOGE("weight host backing does not satisfy the arena range or budget");
            return -100;
        }
    }
    return 0;
}

static int fit_weight_capacity(const VulkanDevice* vkdev, size_t budget, bool automatic, bool concurrent, size_t& capacity)
{
    if (!budget || !capacity)
        return 0;
    const size_t alignment = weight_buffer_alignment(vkdev);
    for (;;)
    {
        size_t allocation;
        if (weight_buffer_memory_size(vkdev, capacity, allocation, concurrent))
            return -1;
        if (allocation <= budget)
            return 0;
        if (!automatic)
            return -1;
        const size_t excess = allocation - budget;
        if (excess > (size_t)-1 - alignment)
            return -1;
        const size_t shrink = alignSize(excess, alignment);
        if (shrink >= capacity)
            return -1;
        capacity -= shrink;
    }
}

int VkComputePrivate::initialize_weight_cache(size_t minimum_device_size, size_t max_source_size, size_t max_external_size)
{
    if (weight_cache->error || executing || max_source_size > (size_t)-1 / 2 || max_external_size > (size_t)-1 / 2)
        return -1;

    // One command owns one exchange budget, independently of the number of
    // models recorded into it. The allocator-based constructor can explicitly
    // derive that budget from one model's finalized residency headroom.
    size_t device_budget = weight_opt.weight_device_budget;
    size_t host_budget = weight_opt.weight_host_budget;
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    if (unified)
    {
        device_budget = device_budget && host_budget ? std::min(device_budget, host_budget) : std::max(device_budget, host_budget);
        host_budget = 0;
    }
    size_t device_size = weight_cache->device_size;
    if (!device_size)
        device_size = std::max(max_source_size * 2, minimum_device_size);
    else if (weight_cache->automatic_device_size)
        device_size = std::max(device_size, minimum_device_size);
    size_t host_size = weight_cache->host_size;
    if (unified)
        host_size = 0;
    else if (max_external_size && !host_size)
        host_size = weight_opt.weight_host_cache_size ? weight_opt.weight_host_cache_size : max_external_size * 2;
    if (device_size > (size_t)-1 - weight_cache->alignment || host_size > (size_t)-1 - weight_cache->alignment)
        return -1;
    device_size = alignSize(device_size, weight_cache->alignment);
    host_size = alignSize(host_size, weight_cache->alignment);
    if (fit_weight_capacity(vkdev, device_budget, weight_cache->automatic_device_size, false, device_size)
        || fit_weight_capacity(vkdev, host_budget, weight_automatic_host_size, true, host_size)
        || device_size < minimum_device_size
        || (max_external_size && !unified && host_size < weight_cache->alignment))
    {
        NCNN_LOGE("weight working set exceeds command capacity or budget");
        return -1;
    }
    device_size = alignSize(device_size, weight_cache->alignment);
    host_size = alignSize(host_size, weight_cache->alignment);
    const bool resize = device_size != weight_cache->device_size || host_size != weight_cache->host_size;
    if (weight_cache->initialized && resize)
    {
        // Do not hold two arenas at once: both would count against the budget.
        // If rebuilding fails, reset() discards the failed state for a retry.
        destroy_weight_commands();
        weight_cache->clear();
    }
    weight_cache->opt.weight_device_budget = device_budget;
    weight_cache->opt.weight_host_budget = host_budget;
    weight_cache->device_size = device_size;
    weight_cache->host_size = host_size;
    if (!weight_cache->initialized)
    {
        weight_cache->error = allocate_weight_cache(minimum_device_size, max_source_size, max_external_size);
        weight_cache->initialized = true;
    }
    if (weight_cache->error)
        return weight_cache->error;
    if (weight_cache->unified)
        weight_queue_mode = 2;
    weight_transfer_family = weight_queue_mode ? vkdev->info.compute_queue_family_index() : vkdev->info.transfer_queue_family_index();
    if (max_external_size && !weight_cache->unified && !weight_cache->host_size)
        return -1;
    if (!weight_commands_initialized)
    {
        weight_cache->error = create_weight_commands();
        weight_commands_initialized = true;
    }
    return weight_cache->error;
}

int VkComputePrivate::weight_binding_source(const VkBufferBinding& binding, const VkBufferMemory*& source) const
{
    source = binding.source;
    if (!source)
        return 0;
    if (!is_deferred_buffer(source) || binding.view_offset > source->external_size
            || binding.descriptor.range > source->external_size - binding.view_offset
            || source->external_size > (size_t)-1 - weight_cache->alignment
            || (source->storage_type != 2 && source->storage_type != 1)
            || (needs_external_copy(source) && !source->external_ptr)
            || (!needs_external_copy(source) && alignSize(source->external_size, 4) > source->capacity))
    {
        NCNN_LOGE("invalid deferred buffer binding");
        return -1;
    }
    return 0;
}

size_t VkComputePrivate::weight_source_size(const VkBufferMemory* source) const
{
    return source && source->external_size <= (size_t)-1 - weight_cache->alignment ? alignSize(source->external_size, weight_cache->alignment) : (size_t)-1;
}

int VkComputePrivate::weight_device_entry(const VkBufferMemory* source) const
{
    for (size_t i = 0; i < weight_cache->entries.size(); i++)
        if (weight_cache->entries[i].source == source)
            return (int)i;
    return -1;
}

int VkComputePrivate::weight_host_entry(const VkBufferMemory* source) const
{
    for (size_t i = 0; i < weight_cache->host_entries.size(); i++)
        if (weight_cache->host_entries[i].source == source)
            return (int)i;
    return -1;
}

static bool weight_is_pinned(const VkBufferMemory* source, const std::vector<const VkBufferMemory*>& pinned)
{
    for (size_t i = 0; i < pinned.size(); i++)
        if (pinned[i] == source)
            return true;
    return false;
}

int VkComputePrivate::place_weight(const VkBufferMemory* source, const std::vector<const VkBufferMemory*>& pinned)
{
    const VkWeightCache::source_entry* recorded = weight_cache->find_source(source);
    if (!recorded)
        return -1;
    const size_t size = alignSize(source->external_size, weight_cache->alignment);
    size_t offset = weight_cache->device_head;
    bool found = false;
    for (int pass = 0; pass < 2 && !found; pass++)
    {
        for (size_t attempt = 0; attempt <= weight_cache->entries.size(); attempt++)
        {
            if (offset > weight_cache->device_size || size > weight_cache->device_size - offset)
                break;
            size_t next_offset = offset;
            for (size_t i = 0; i < weight_cache->entries.size(); i++)
            {
                const VkWeightCache::Entry& e = weight_cache->entries[i];
                if (!e.source || !weight_is_pinned(e.source, pinned))
                    continue;
                if (offset < e.offset + e.size && e.offset < offset + size)
                    next_offset = std::max(next_offset, e.offset + e.size);
            }
            if (next_offset == offset)
            {
                found = true;
                break;
            }
            offset = next_offset;
        }
        if (!found)
            offset = 0;
    }
    if (!found)
        return -1;

    int slot = -1;
    for (size_t i = 0; i < weight_cache->entries.size(); i++)
    {
        VkWeightCache::Entry& e = weight_cache->entries[i];
        if (e.source && offset < e.offset + e.size && e.offset < offset + size)
        {
            if (weight_cache->unified && wait_weight_compute(e.compute_serial))
                return -1;
            evict_weight(i);
        }
        if (!e.source)
            slot = (int)i;
    }
    VkWeightCache::Entry entry;
    entry.source = source;
    entry.allocator = recorded->allocator;
    entry.generation = recorded->generation;
    entry.offset = offset;
    entry.size = size;
    entry.compute_serial = 0;
    entry.valid = false;
    if (slot == -1)
    {
        slot = (int)weight_cache->entries.size();
        weight_cache->entries.push_back(entry);
    }
    else
    {
        weight_cache->entries[slot] = entry;
    }
    weight_cache->device_head = offset + size;
    return 0;
}

void VkComputePrivate::evict_weight(size_t slot)
{
    VkWeightCache::Entry& entry = weight_cache->entries[slot];
    if (!entry.source)
        return;
    if (entry.valid && !weight_cache->unified)
    {
        Range range;
        range.offset = entry.offset;
        range.size = entry.size;
        pending_releases.push_back(range);
    }
    entry.source = 0;
}

int VkComputePrivate::evict_weights()
{
    for (size_t i = 0; i < weight_cache->entries.size(); i++)
    {
        if (weight_cache->unified && wait_weight_compute(weight_cache->entries[i].compute_serial))
            return -1;
        evict_weight(i);
    }
    weight_cache->entries.clear();
    weight_cache->device_head = 0;
    return 0;
}

static int weight_begin(VkCommandBuffer command)
{
    VkResult ret;
    ret = vkResetCommandBuffer(command, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
        return -1;
    }
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ret = vkBeginCommandBuffer(command, &info);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
        return -1;
    }
    return 0;
}

int VkComputePrivate::wait_weight_fence(VkFence fence, bool transfer)
{
    VkResult ret = vkGetFenceStatus(vkdev->vkdevice(), fence);
    if (ret == VK_SUCCESS)
        return 0;
    if (ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetFenceStatus failed %d", ret);
        return -1;
    }
#if NCNN_BENCHMARK
    const double start = get_current_time();
#endif
    ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
#if NCNN_BENCHMARK
    if (transfer)
        weight_cache->stats.transfer_wait_time += get_current_time() - start;
    else
        weight_cache->stats.compute_wait_time += get_current_time() - start;
#else
    (void)transfer;
#endif
    if (ret != VK_SUCCESS)
        NCNN_LOGE("vkWaitForFences failed %d", ret);
    return ret == VK_SUCCESS ? 0 : -1;
}

int VkComputePrivate::wait_weight_compute(uint64_t serial)
{
    if (serial <= completed_compute_serial)
        return 0;
    for (int i = 0; i < weight_compute_slots; i++)
    {
        if (computes[i].pending && computes[i].serial == serial)
        {
            if (wait_weight_fence(computes[i].fence, false))
                return -1;
            completed_compute_serial = serial;
            return 0;
        }
    }
    return -1;
}

int VkComputePrivate::retire_weight_upload(Upload& u)
{
    if (u.pending)
    {
        if (!u.consumer_serial || wait_weight_compute(u.consumer_serial) || wait_weight_fence(u.fence, true))
            return -1;
    }
    if (u.release_pending && wait_weight_fence(u.release_fence, false))
        return -1;
#if NCNN_BENCHMARK
    if (u.pending && transfer_timestamp_bits && collect_weight_time((uint32_t)(&u - uploads) * 2, transfer_timestamp_bits, weight_cache->stats.upload_time))
        return -1;
#endif
    u.pending = false;
    u.release_pending = false;
    u.host_active = false;
    u.consumer_serial = 0;
    return 0;
}

int VkComputePrivate::weight_host_range(size_t size, size_t& offset)
{
    if (size > weight_cache->host_size)
        return -1;
    if (size > weight_cache->host_size - weight_cache->host_head)
        weight_cache->host_head = 0;
    offset = weight_cache->host_head;
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        Upload& u = uploads[i];
        if (u.host_active && offset < u.host_offset + u.host_size && u.host_offset < offset + size)
        {
            if (wait_weight_fence(u.fence, true))
                return -1;
            u.host_active = false;
        }
    }
    for (size_t i = 0; i < weight_cache->host_entries.size(); i++)
    {
        VkWeightCache::HostEntry& entry = weight_cache->host_entries[i];
        if (entry.source && offset < entry.offset + entry.size && entry.offset < offset + size)
        {
            entry.source = 0;
        }
    }
    weight_cache->host_head += size;
    return 0;
}

int VkComputePrivate::flush_weight_host(size_t offset, size_t size)
{
    if (!weight_cache->host_allocator)
        return flush_weight_buffer(vkdev, weight_cache->host_arena, offset, size);
    VkBufferMemory range = *weight_cache->host_arena;
    range.offset += offset;
    range.capacity = size;
    return weight_cache->host_allocator->flush(&range);
}

int VkComputePrivate::cache_host_weight(const VkBufferMemory* source, size_t& offset)
{
    const int cached = weight_host_entry(source);
    if (cached >= 0)
    {
        offset = weight_cache->host_entries[cached].offset;
#if NCNN_BENCHMARK
        weight_cache->stats.host_cache_hits++;
#endif
        return 0;
    }
    const VkWeightCache::source_entry* recorded = weight_cache->find_source(source);
    if (!recorded)
        return -1;
    const size_t size = alignSize(source->external_size, weight_cache->alignment);
    if (weight_host_range(size, offset))
        return -1;
    unsigned char* destination = (unsigned char*)weight_cache->host_arena->mapped_ptr + weight_cache->host_arena->offset + offset;
#if NCNN_BENCHMARK
    const double start = get_current_time();
#endif
    memcpy(destination, source->external_ptr, source->external_size);
#if NCNN_BENCHMARK
    if (source->storage_type == 2)
        weight_cache->stats.disk_read_time += get_current_time() - start;
#endif
    memset(destination + source->external_size, 0, size - source->external_size);
    if (flush_weight_host(offset, size))
        return -1;
    int slot = -1;
    for (size_t i = 0; i < weight_cache->host_entries.size(); i++)
        if (!weight_cache->host_entries[i].source)
            slot = (int)i;
    VkWeightCache::HostEntry entry;
    entry.source = source;
    entry.allocator = recorded->allocator;
    entry.generation = recorded->generation;
    entry.offset = offset;
    entry.size = size;
    if (slot < 0)
    {
        slot = (int)weight_cache->host_entries.size();
        weight_cache->host_entries.push_back(entry);
    }
    else
        weight_cache->host_entries[slot] = entry;
#if NCNN_BENCHMARK
    if (source->storage_type == 2)
    {
        weight_cache->stats.file_reads++;
        weight_cache->stats.bytes_read += source->external_size;
    }
    weight_cache->stats.host_bytes_staged += source->external_size;
#endif
    return 0;
}

void VkComputePrivate::update_weight_peaks()
{
#if NCNN_BENCHMARK
    uint32_t compute_count = 0;
    uint32_t transfer_count = 0;
    for (int i = 0; i < weight_compute_slots; i++)
        if (computes[i].pending && vkGetFenceStatus(vkdev->vkdevice(), computes[i].fence) == VK_NOT_READY)
            compute_count++;
    for (int i = 0; i < weight_transfer_slots; i++)
        if (uploads[i].pending && vkGetFenceStatus(vkdev->vkdevice(), uploads[i].fence) == VK_NOT_READY)
            transfer_count++;
    weight_cache->stats.peak_pending_compute = std::max(weight_cache->stats.peak_pending_compute, compute_count);
    weight_cache->stats.peak_pending_transfer = std::max(weight_cache->stats.peak_pending_transfer, transfer_count);
#endif
}

void VkComputePrivate::weight_ownership_barrier(VkCommandBuffer command, const Range& range, bool to_compute, bool release)
{
    const VkAccessFlags compute_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    const VkPipelineStageFlags compute_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = release ? (to_compute ? (VkAccessFlags)VK_ACCESS_TRANSFER_WRITE_BIT : compute_access) : 0;
    barrier.dstAccessMask = release ? 0 : (to_compute ? compute_access : (VkAccessFlags)VK_ACCESS_TRANSFER_WRITE_BIT);
    const uint32_t compute_family = vkdev->info.compute_queue_family_index();
    barrier.srcQueueFamilyIndex = weight_transfer_family == compute_family ? VK_QUEUE_FAMILY_IGNORED : (to_compute ? weight_transfer_family : compute_family);
    barrier.dstQueueFamilyIndex = weight_transfer_family == compute_family ? VK_QUEUE_FAMILY_IGNORED : (to_compute ? compute_family : weight_transfer_family);
    barrier.buffer = weight_cache->device_arena->buffer;
    barrier.offset = weight_cache->device_arena->offset + range.offset;
    barrier.size = range.size;
    const VkPipelineStageFlags src_stage = release ? (to_compute ? (VkPipelineStageFlags)VK_PIPELINE_STAGE_TRANSFER_BIT : compute_stage) : (VkPipelineStageFlags)VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    const VkPipelineStageFlags dst_stage = release ? (VkPipelineStageFlags)VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : (to_compute ? compute_stage : (VkPipelineStageFlags)VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdPipelineBarrier(command, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
#if NCNN_BENCHMARK
    if (release && weight_transfer_family != compute_family)
        weight_cache->stats.ownership_transfers++;
#endif
}

int VkComputePrivate::begin_weight_commands()
{
    VkResult ret;
    if (weight_cache->error || executing)
        return -1;
    const uint32_t compute_family = vkdev->info.compute_queue_family_index();
    compute_queue = vkdev->acquire_queue(compute_family);
    if (!compute_queue)
        return -1;
    executing = true;
    // Earlier submissions have finished before this command reuses its regions.
    for (size_t i = 0; i < weight_cache->entries.size(); i++)
        weight_cache->entries[i].compute_serial = 0;
    if (weight_transfer_family != compute_family)
        transfer_queue = vkdev->acquire_queue(weight_transfer_family);
    else if (weight_queue_mode != 2)
        transfer_queue = vkdev->try_acquire_queue(compute_family);
    if (!transfer_queue && weight_transfer_family == compute_family)
        transfer_queue = compute_queue;
    if (!transfer_queue)
        return finish_weight_commands(-1);
#if NCNN_BENCHMARK
    weight_cache->stats.separate_transfer_queue = transfer_queue != compute_queue;
    weight_cache->stats.separate_transfer_family = weight_transfer_family != compute_family;
#endif

    // carry cached-weight visibility across successive leases of compute queues
    if (weight_cache->handoff_pending)
    {
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &weight_cache->handoff;
        info.pWaitDstStageMask = &stage;
        ret = vkResetFences(vkdev->vkdevice(), 1, &handoff_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return finish_weight_commands(-1);
        }
        ret = vkQueueSubmit(compute_queue, 1, &info, handoff_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkQueueSubmit failed %d", ret);
            return finish_weight_commands(-1);
        }
#if NCNN_BENCHMARK
        weight_cache->stats.compute_submissions++;
#endif
        if (wait_weight_fence(handoff_fence, false))
            return finish_weight_commands(-1);
        weight_cache->handoff_pending = false;
    }
    return 0;
}

int VkComputePrivate::submit_weight_prefix(VkCommandBuffer command, VkFence fence)
{
    VkResult ret;
    if (!executing || prefix_fence)
        return -1;
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &command;
    ret = vkQueueSubmit(compute_queue, 1, &info, fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return -1;
    }
    prefix_fence = fence;
#if NCNN_BENCHMARK
    weight_cache->stats.compute_submissions++;
#endif
    return 0;
}

int VkComputePrivate::upload_weight_chunk(Upload& u, const VkBufferMemory* source, size_t source_offset, size_t offset, size_t size, const Range& range, bool last)
{
    VkResult ret;
    if (!(source->buffer_usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return -1;
    if (u.release_pending)
    {
        if (wait_weight_fence(u.release_fence, false))
            return -1;
        u.release_pending = false;
    }
    const bool returning = !pending_releases.empty();
    if (returning)
    {
        if (weight_begin(u.release_command))
            return -1;
        ret = vkResetFences(vkdev->vkdevice(), 1, &u.release_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return -1;
        }
        for (size_t i = 0; i < pending_releases.size(); i++)
            weight_ownership_barrier(u.release_command, pending_releases[i], false, true);
        ret = vkEndCommandBuffer(u.release_command);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
            return -1;
        }
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &u.release_command;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &u.returned;
        ret = vkQueueSubmit(compute_queue, 1, &info, u.release_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkQueueSubmit failed %d", ret);
            return -1;
        }
        u.release_pending = true;
#if NCNN_BENCHMARK
        weight_cache->stats.compute_submissions++;
#endif
    }
    if (weight_begin(u.command))
        return -1;
    ret = vkResetFences(vkdev->vkdevice(), 1, &u.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetFences failed %d", ret);
        return -1;
    }
    for (size_t i = 0; i < pending_releases.size(); i++)
        weight_ownership_barrier(u.command, pending_releases[i], false, false);
    VkBufferMemoryBarrier host_barrier = {};
    host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    host_barrier.srcAccessMask = source->mapped_ptr ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_barrier.buffer = source->buffer;
    host_barrier.offset = source_offset;
    host_barrier.size = size;
    vkCmdPipelineBarrier(u.command, source->mapped_ptr ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &host_barrier, 0, 0);
#if NCNN_BENCHMARK
    if (transfer_timestamp_bits)
    {
        vkCmdResetQueryPool(u.command, timing_pool, next_upload * 2, 2);
        vkCmdWriteTimestamp(u.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, next_upload * 2);
    }
#endif
    VkBufferCopy copy;
    copy.srcOffset = source_offset;
    copy.dstOffset = weight_cache->device_arena->offset + offset;
    copy.size = size;
    vkCmdCopyBuffer(u.command, source->buffer, weight_cache->device_arena->buffer, 1, &copy);
#if NCNN_BENCHMARK
    if (transfer_timestamp_bits)
        vkCmdWriteTimestamp(u.command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, next_upload * 2 + 1);
#endif
    if (last)
        weight_ownership_barrier(u.command, range, true, true);
    ret = vkEndCommandBuffer(u.command);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }
    VkPipelineStageFlags stage = weight_transfer_family != vkdev->info.compute_queue_family_index() ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = returning ? 1 : 0;
    info.pWaitSemaphores = &u.returned;
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &u.command;
    info.signalSemaphoreCount = last ? 1 : 0;
    info.pSignalSemaphores = &u.ready;
    ret = vkQueueSubmit(transfer_queue, 1, &info, u.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return -1;
    }
    u.pending = true;
    u.host_active = source == weight_cache->host_arena;
    u.host_offset = u.host_active ? source_offset - weight_cache->host_arena->offset : 0;
    u.host_size = u.host_active ? alignSize(size, weight_cache->alignment) : 0;
    u.offset = range.offset;
    u.size = range.size;
    pending_releases.clear();
#if NCNN_BENCHMARK
    weight_cache->stats.transfer_submissions++;
    weight_cache->stats.upload_chunks++;
    weight_cache->stats.bytes_uploaded += size;
#endif
    update_weight_peaks();
    if (last)
    {
        pending_uploads.push_back(next_upload);
        next_upload = (next_upload + 1) % weight_transfer_slots;
    }
    else
    {
        // intermediate chunks reuse this slot without signaling its ready semaphore
        if (wait_weight_fence(u.fence, true))
            return -1;
#if NCNN_BENCHMARK
        if (transfer_timestamp_bits && collect_weight_time(next_upload * 2, transfer_timestamp_bits, weight_cache->stats.upload_time))
            return -1;
#endif
        u.pending = false;
        u.host_active = false;
    }
    return 0;
}

int VkComputePrivate::upload_weight(const VkBufferMemory* source, size_t offset)
{
    const size_t size = alignSize(source->external_size, weight_cache->alignment);
    if (weight_cache->unified && needs_external_copy(source))
    {
        unsigned char* destination = (unsigned char*)weight_cache->device_arena->mapped_ptr + weight_cache->device_arena->offset + offset;
#if NCNN_BENCHMARK
        const double start = get_current_time();
#endif
        memcpy(destination, source->external_ptr, source->external_size);
#if NCNN_BENCHMARK
        if (source->storage_type == 2)
            weight_cache->stats.disk_read_time += get_current_time() - start;
#endif
        memset(destination + source->external_size, 0, size - source->external_size);
        if (weight_cache->device_allocator)
        {
            VkBufferMemory range = *weight_cache->device_arena;
            range.offset += offset;
            range.capacity = size;
            if (weight_cache->device_allocator->flush(&range))
                return -1;
        }
        else if (flush_weight_buffer(vkdev, weight_cache->device_arena, offset, size))
            return -1;
#if NCNN_BENCHMARK
        if (source->storage_type == 2)
        {
            weight_cache->stats.file_reads++;
            weight_cache->stats.bytes_read += source->external_size;
        }
        weight_cache->stats.unified_bytes_written += source->external_size;
#endif
        return 0;
    }
    Upload& u = uploads[next_upload];
    if (retire_weight_upload(u))
        return -1;
    Range range;
    range.offset = offset;
    range.size = size;
    if (!needs_external_copy(source))
    {
        const int ret = upload_weight_chunk(u, source, source->offset, offset, alignSize(source->external_size, 4), range, true);
#if NCNN_BENCHMARK
        if (!ret)
            weight_cache->stats.host_resident_bytes_uploaded += alignSize(source->external_size, 4);
#endif
        return ret;
    }
    if (size <= weight_cache->host_size)
    {
        size_t host_offset;
        if (cache_host_weight(source, host_offset))
            return -1;
        return upload_weight_chunk(u, weight_cache->host_arena, weight_cache->host_arena->offset + host_offset, offset, alignSize(source->external_size, 4), range, true);
    }
    for (size_t file_offset = 0; file_offset < source->external_size;)
    {
        const size_t chunk_size = std::min(weight_cache->host_size, source->external_size - file_offset);
        const size_t host_range_size = alignSize(chunk_size, weight_cache->alignment);
        size_t host_offset;
        if (!chunk_size || weight_host_range(host_range_size, host_offset))
            return -1;
        unsigned char* destination = (unsigned char*)weight_cache->host_arena->mapped_ptr + weight_cache->host_arena->offset + host_offset;
#if NCNN_BENCHMARK
        const double start = get_current_time();
#endif
        memcpy(destination, (const unsigned char*)source->external_ptr + file_offset, chunk_size);
#if NCNN_BENCHMARK
        if (source->storage_type == 2)
            weight_cache->stats.disk_read_time += get_current_time() - start;
#endif
        memset(destination + chunk_size, 0, host_range_size - chunk_size);
        if (flush_weight_host(host_offset, host_range_size))
            return -1;
#if NCNN_BENCHMARK
        if (source->storage_type == 2)
        {
            weight_cache->stats.file_reads++;
            weight_cache->stats.bytes_read += chunk_size;
        }
        weight_cache->stats.host_bytes_staged += chunk_size;
#endif
        const bool last = chunk_size == source->external_size - file_offset;
        if (upload_weight_chunk(u, weight_cache->host_arena, weight_cache->host_arena->offset + host_offset, offset + file_offset, alignSize(chunk_size, 4), range, last))
            return -1;
        file_offset += chunk_size;
    }
    return 0;
}

int VkComputePrivate::prepare_weights(const std::vector<const VkBufferMemory*>& weights)
{
    if (!executing || !pending_uploads.empty() || current_compute >= 0 || weights.size() > (size_t)weight_transfer_slots)
        return -1;
    size_t total = 0;
    for (size_t i = 0; i < weights.size(); i++)
    {
        const size_t size = weight_source_size(weights[i]);
        if (size > weight_cache->device_size - total)
        {
            NCNN_LOGE("simultaneous weight bindings exceed device cache budget");
            return -1;
        }
        total += size;
    }
    std::vector<const VkBufferMemory*> missing;
    for (size_t i = 0; i < weights.size(); i++)
    {
        if (weight_device_entry(weights[i]) >= 0)
            continue;
        if (place_weight(weights[i], weights) != 0)
        {
            // return fragmented ranges before compacting the next working set
            if (evict_weights())
                return -1;
            missing = weights;
            for (size_t j = 0; j < weights.size(); j++)
                if (place_weight(weights[j], weights) != 0)
                    return -1;
            break;
        }
        missing.push_back(weights[i]);
    }
    for (size_t i = 0; i < missing.size(); i++)
    {
        const VkBufferMemory* source = missing[i];
        VkWeightCache::Entry& entry = weight_cache->entries[weight_device_entry(source)];
        if (upload_weight(source, entry.offset))
            return -1;
        entry.valid = true;
    }
    current_sources = weights;
#if NCNN_BENCHMARK
    weight_cache->stats.cache_hits += weights.size() - missing.size();
    weight_cache->stats.cache_misses += missing.size();
#endif
    return 0;
}

int VkComputePrivate::begin_weight_compute(VkCommandBuffer& command)
{
    VkResult ret;
    if (!executing || current_compute >= 0)
        return -1;
    VkComputePrivate::Compute& c = computes[next_compute];
    if (c.pending)
    {
        if (wait_weight_compute(c.serial))
            return -1;
#if NCNN_BENCHMARK
        if (compute_timestamp_bits && collect_weight_time((weight_transfer_slots + next_compute) * 2, compute_timestamp_bits, weight_cache->stats.compute_time))
            return -1;
#endif
        c.pending = false;
    }
    if (weight_begin(c.command))
        return -1;
    ret = vkResetFences(vkdev->vkdevice(), 1, &c.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetFences failed %d", ret);
        return -1;
    }
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        const VkComputePrivate::Upload& u = uploads[pending_uploads[i]];
        VkComputePrivate::Range range;
        range.offset = u.offset;
        range.size = u.size;
        weight_ownership_barrier(c.command, range, true, false);
    }
    if (weight_cache->unified)
    {
        for (size_t i = 0; i < current_sources.size(); i++)
        {
            const VkBufferMemory* source = current_sources[i];
            if (!needs_external_copy(source))
                continue;
            const VkWeightCache::Entry& entry = weight_cache->entries[weight_device_entry(source)];
            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = weight_cache->device_arena->buffer;
            barrier.offset = weight_cache->device_arena->offset + entry.offset;
            barrier.size = entry.size;
            vkCmdPipelineBarrier(c.command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
#if NCNN_BENCHMARK
    if (compute_timestamp_bits)
    {
        vkCmdResetQueryPool(c.command, timing_pool, (weight_transfer_slots + next_compute) * 2, 2);
        vkCmdWriteTimestamp(c.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, (weight_transfer_slots + next_compute) * 2);
    }
#endif
    command = c.command;
    current_compute = next_compute;
    return 0;
}

int VkComputePrivate::submit_weight_compute()
{
    VkResult ret;
    if (current_compute < 0)
        return -1;
    VkComputePrivate::Compute& c = computes[current_compute];
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(c.command);
#endif
#if NCNN_BENCHMARK
    if (compute_timestamp_bits)
        vkCmdWriteTimestamp(c.command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, (weight_transfer_slots + current_compute) * 2 + 1);
#endif
    ret = vkEndCommandBuffer(c.command);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }
    VkSemaphore semaphores[weight_transfer_slots];
    VkPipelineStageFlags stages[weight_transfer_slots];
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        semaphores[i] = uploads[pending_uploads[i]].ready;
        stages[i] = weight_transfer_family != vkdev->info.compute_queue_family_index() ? (VkPipelineStageFlags)VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = (uint32_t)pending_uploads.size();
    info.pWaitSemaphores = semaphores;
    info.pWaitDstStageMask = stages;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &c.command;
    ret = vkQueueSubmit(compute_queue, 1, &info, c.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return -1;
    }
    c.serial = ++compute_serial;
    c.pending = true;
    for (size_t i = 0; i < pending_uploads.size(); i++)
        uploads[pending_uploads[i]].consumer_serial = c.serial;
    for (size_t i = 0; i < current_sources.size(); i++)
        weight_cache->entries[weight_device_entry(current_sources[i])].compute_serial = c.serial;
    pending_uploads.clear();
    current_sources.clear();
    current_compute = -1;
    next_compute = (next_compute + 1) % weight_compute_slots;
#if NCNN_BENCHMARK
    weight_cache->stats.compute_submissions++;
#endif
    update_weight_peaks();
    return 0;
}

int VkComputePrivate::finish_weight_commands(int result)
{
    if (!executing)
        return result ? result : weight_cache->error;
    for (int i = 0; i < weight_compute_slots; i++)
    {
        if (computes[i].pending && wait_weight_fence(computes[i].fence, false))
            result = -1;
#if NCNN_BENCHMARK
        if (computes[i].pending && compute_timestamp_bits && !result && collect_weight_time((weight_transfer_slots + i) * 2, compute_timestamp_bits, weight_cache->stats.compute_time))
            result = -1;
#endif
        computes[i].pending = false;
    }
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        VkComputePrivate::Upload& u = uploads[i];
        if (u.pending && wait_weight_fence(u.fence, true))
            result = -1;
        if (u.release_pending && wait_weight_fence(u.release_fence, false))
            result = -1;
#if NCNN_BENCHMARK
        if (u.pending && transfer_timestamp_bits && !result && collect_weight_time(i * 2, transfer_timestamp_bits, weight_cache->stats.upload_time))
            result = -1;
#endif
        u.pending = false;
        u.release_pending = false;
        u.host_active = false;
        u.consumer_serial = 0;
    }
    if (prefix_fence && wait_weight_fence(prefix_fence, false))
        result = -1;
    prefix_fence = VK_NULL_HANDLE;
    if (!result)
    {
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &weight_cache->handoff;
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &handoff_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            result = -1;
        }
        if (!result)
        {
            ret = vkQueueSubmit(compute_queue, 1, &info, handoff_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                result = -1;
            }
        }
        if (!result && wait_weight_fence(handoff_fence, false))
            result = -1;
        if (!result)
        {
            weight_cache->handoff_pending = true;
#if NCNN_BENCHMARK
            weight_cache->stats.compute_submissions++;
#endif
        }
    }
    if (transfer_queue && transfer_queue != compute_queue)
        vkdev->reclaim_queue(weight_transfer_family, transfer_queue);
    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
    compute_queue = VK_NULL_HANDLE;
    transfer_queue = VK_NULL_HANDLE;
    executing = false;
    current_compute = -1;
    completed_compute_serial = compute_serial;
    pending_uploads.clear();
    current_sources.clear();
    if (result)
        weight_cache->error = result;
#if NCNN_BENCHMARK
    if (weight_benchmark_callback)
    {
        weight_benchmark_callback("file_reads", weight_cache->stats.file_reads, 0.);
        weight_benchmark_callback("bytes_read", weight_cache->stats.bytes_read, 0.);
        weight_benchmark_callback("bytes_uploaded", weight_cache->stats.bytes_uploaded, 0.);
        weight_benchmark_callback("cache_hits", weight_cache->stats.cache_hits, 0.);
        weight_benchmark_callback("cache_misses", weight_cache->stats.cache_misses, 0.);
        weight_benchmark_callback("transfer_submissions", weight_cache->stats.transfer_submissions, 0.);
        weight_benchmark_callback("compute_submissions", weight_cache->stats.compute_submissions, 0.);
        weight_benchmark_callback("ownership_transfers", weight_cache->stats.ownership_transfers, 0.);
        weight_benchmark_callback("host_bytes_staged", weight_cache->stats.host_bytes_staged, 0.);
        weight_benchmark_callback("host_cache_hits", weight_cache->stats.host_cache_hits, 0.);
        weight_benchmark_callback("host_resident_bytes_uploaded", weight_cache->stats.host_resident_bytes_uploaded, 0.);
        weight_benchmark_callback("upload_chunks", weight_cache->stats.upload_chunks, 0.);
        weight_benchmark_callback("unified_bytes_written", weight_cache->stats.unified_bytes_written, 0.);
        weight_benchmark_callback("peak_pending_compute", weight_cache->stats.peak_pending_compute, 0.);
        weight_benchmark_callback("peak_pending_transfer", weight_cache->stats.peak_pending_transfer, 0.);
        weight_benchmark_callback("disk_read_time", 0, weight_cache->stats.disk_read_time);
        weight_benchmark_callback("upload_time", 0, weight_cache->stats.upload_time);
        weight_benchmark_callback("compute_time", 0, weight_cache->stats.compute_time);
        weight_benchmark_callback("transfer_wait_time", 0, weight_cache->stats.transfer_wait_time);
        weight_benchmark_callback("compute_wait_time", 0, weight_cache->stats.compute_wait_time);
        weight_benchmark_callback("upload_time_available", weight_cache->stats.upload_time_available, 0.);
        weight_benchmark_callback("host_capacity", weight_cache->stats.host_capacity, 0.);
        weight_benchmark_callback("device_capacity", weight_cache->stats.device_capacity, 0.);
        weight_benchmark_callback("host_allocation_size", weight_cache->stats.host_allocation_size, 0.);
        weight_benchmark_callback("device_allocation_size", weight_cache->stats.device_allocation_size, 0.);
        weight_benchmark_callback("separate_transfer_queue", weight_cache->stats.separate_transfer_queue, 0.);
        weight_benchmark_callback("separate_transfer_family", weight_cache->stats.separate_transfer_family, 0.);
        weight_benchmark_callback("unified_memory", weight_cache->stats.unified_memory, 0.);
    }
#endif
    return result;
}

int VkComputePrivate::resolve_weight_binding(const VkBufferBinding& binding, VkDescriptorBufferInfo& descriptor) const
{
    const VkBufferMemory* source;
    if (weight_binding_source(binding, source))
        return -1;
    descriptor = binding.descriptor;
    if (!source)
        return 0;
    const int slot = weight_device_entry(source);
    if (slot < 0 || !weight_cache->entries[slot].valid)
        return -1;
    descriptor.buffer = weight_cache->device_arena->buffer;
    descriptor.offset = weight_cache->device_arena->offset + weight_cache->entries[slot].offset + binding.view_offset;
    return 0;
}

#if NCNN_BENCHMARK
int VkComputePrivate::collect_weight_time(uint32_t query, uint32_t bits, double& elapsed)
{
    VkResult ret;
    uint64_t timestamps[2];
    ret = vkGetQueryPoolResults(vkdev->vkdevice(), timing_pool, query, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetQueryPoolResults failed %d", ret);
        return -1;
    }
    const uint64_t mask = bits == 64 ? (uint64_t)-1 : (((uint64_t)1 << bits) - 1);
    elapsed += ((timestamps[1] - timestamps[0]) & mask) * vkdev->info.timestamp_period() / 1000000.;
    return 0;
}

#endif

class WeightResidentUpload
{
public:
    explicit WeightResidentUpload(const VulkanDevice* _vkdev)
        : vkdev(_vkdev), staging(0), allocation_size(0), queue(0), pool(0), command(0), fence(0), pending(false)
    {
    }

    ~WeightResidentUpload()
    {
        if (pending)
        {
            VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
            if (ret != VK_SUCCESS)
                NCNN_LOGE("vkWaitForFences failed %d", ret);
        }
        if (fence)
            vkDestroyFence(vkdev->vkdevice(), fence, 0);
        if (pool)
            vkDestroyCommandPool(vkdev->vkdevice(), pool, 0);
        if (queue)
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), queue);
        destroy_weight_buffer(vkdev, staging);
    }

    int initialize(size_t budget, size_t max_size)
    {
        VkResult ret;
        const size_t alignment = weight_buffer_alignment(vkdev);
        const size_t size = std::min(std::min(budget, max_size), (size_t)32 * 1024 * 1024) / alignment * alignment;
        if (!size)
        {
            NCNN_LOGE("weight load staging budget %llu needs at least %llu bytes", (unsigned long long)budget, (unsigned long long)alignment);
            return -100;
        }
        if (create_weight_buffer(vkdev, size, 1, false, budget, staging, allocation_size))
            return -100;
        const uint32_t family = vkdev->info.compute_queue_family_index();
        queue = vkdev->acquire_queue(family);
        if (!queue)
            return -1;
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = family;
        ret = vkCreateCommandPool(vkdev->vkdevice(), &pool_info, 0, &pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return -1;
        }
        VkCommandBufferAllocateInfo command_info = {};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &command_info, &command);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        ret = vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
        return 0;
    }

    int upload(const VkBufferMemory* destination)
    {
        VkResult ret;
        const size_t data_size = destination->external_size;
        for (size_t offset = 0; offset < data_size;)
        {
            const size_t size = std::min(staging->capacity, data_size - offset);
            const size_t copy_size = alignSize(size, 4);
            memset((unsigned char*)staging->mapped_ptr + size, 0, copy_size - size);
            memcpy(staging->mapped_ptr, (const unsigned char*)destination->external_ptr + offset, size);
            if (flush_weight_buffer(vkdev, staging, 0, copy_size))
                return -1;
            ret = vkResetCommandBuffer(command, 0);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
                return -1;
            }
            ret = vkResetFences(vkdev->vkdevice(), 1, &fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetFences failed %d", ret);
                return -1;
            }
            VkCommandBufferBeginInfo begin = {};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            ret = vkBeginCommandBuffer(command, &begin);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
                return -1;
            }
            VkBufferMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = staging->buffer;
            barrier.offset = 0;
            barrier.size = copy_size;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
            VkBufferCopy copy;
            copy.srcOffset = 0;
            copy.dstOffset = destination->offset + offset;
            copy.size = copy_size;
            vkCmdCopyBuffer(command, staging->buffer, destination->buffer, 1, &copy);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barrier.buffer = destination->buffer;
            barrier.offset = copy.dstOffset;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
            ret = vkEndCommandBuffer(command);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
                return -1;
            }
            VkSubmitInfo submit = {};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            ret = vkQueueSubmit(queue, 1, &submit, fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                return -1;
            }
            pending = true;
            ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkWaitForFences failed %d", ret);
                return -1;
            }
            pending = false;
            offset += size;
        }
        return 0;
    }

    const VulkanDevice* vkdev;
    VkBufferMemory* staging;
    size_t allocation_size;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer command;
    VkFence fence;
    bool pending;
};

int upload_weight_buffers(const VulkanDevice* vkdev, const std::vector<VkBufferMemory>& buffers, size_t host_budget)
{
    size_t max_size = 0;
    for (size_t i = 0; i < buffers.size(); i++)
        if (!buffers[i].mapped_ptr)
            max_size = std::max(max_size, buffers[i].capacity);
    WeightResidentUpload upload(vkdev);
    if (max_size)
    {
        const int ret = upload.initialize(host_budget, max_size);
        if (ret)
            return ret;
    }
    for (size_t i = 0; i < buffers.size(); i++)
    {
        const VkBufferMemory& buffer = buffers[i];
        if (buffer.mapped_ptr)
        {
            unsigned char* destination = (unsigned char*)buffer.mapped_ptr + buffer.offset;
            memcpy(destination, buffer.external_ptr, buffer.external_size);
            memset(destination + buffer.external_size, 0, buffer.capacity - buffer.external_size);
            if (flush_weight_buffer(vkdev, &buffer, 0, buffer.capacity))
                return -1;
        }
        else if (upload.upload(&buffer))
            return -1;
    }
    return 0;
}

class VkTransferPrivate
{
public:
    VkTransferPrivate(const VulkanDevice* _vkdev);
    ~VkTransferPrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();

    const VulkanDevice* vkdev;

    uint64_t pending_upload_total;
    int upload_error;

    VkCommandPool compute_command_pool;
    VkCommandPool transfer_command_pool;

    VkCommandBuffer upload_command_buffer;
    VkCommandBuffer compute_command_buffer;

    VkSemaphore upload_compute_semaphore;

    VkFence upload_command_fence;
    VkFence compute_command_fence;

    std::vector<VkMat> upload_staging_buffers;
};

VkTransferPrivate::VkTransferPrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    pending_upload_total = 0;
    upload_error = 0;

    compute_command_pool = 0;
    transfer_command_pool = 0;

    upload_command_buffer = 0;
    compute_command_buffer = 0;

    upload_compute_semaphore = 0;

    upload_command_fence = 0;
    compute_command_fence = 0;

    upload_error = init();
}

VkTransferPrivate::~VkTransferPrivate()
{
    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);

        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);

        vkFreeCommandBuffers(vkdev->vkdevice(), transfer_command_pool, 1, &upload_command_buffer);
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    }
}

int VkTransferPrivate::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        // transfer_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            commandPoolCreateInfo.queueFamilyIndex = vkdev->info.transfer_queue_family_index();

            VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateCommandPool failed %d", ret);
                return -1;
            }
        }

        // upload_command_buffer
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = transfer_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
                return -1;
            }
        }

        // upload_compute_semaphore
        {
            VkSemaphoreCreateInfo semaphoreCreateInfo;
            semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreCreateInfo.pNext = 0;
            semaphoreCreateInfo.flags = 0;

            VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphore);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateSemaphore failed %d", ret);
                return -1;
            }
        }

        // upload_command_fence
        {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateFence failed %d", ret);
                return -1;
            }
        }
    }

    begin_command_buffer();

    return 0;
}

int VkTransferPrivate::begin_command_buffer()
{
    {
        VkCommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = 0;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        commandBufferBeginInfo.pInheritanceInfo = 0;

        VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(upload_command_buffer, &commandBufferBeginInfo);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }

    return 0;
}

int VkTransferPrivate::end_command_buffer()
{
    {
        VkResult ret = vkEndCommandBuffer(compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkResult ret = vkEndCommandBuffer(upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }

    return 0;
}

VkTransfer::VkTransfer(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkTransferPrivate(_vkdev))
{
}

VkTransfer::~VkTransfer()
{
    delete d;
}

void VkTransfer::record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten)
{
    if (d->upload_error)
        return;

    if (src.empty())
        return;

    //     NCNN_LOGE("record_upload src = %d | %d %d %d @ %d", src.dims, src.w, src.h, src.c, src.elempack);

    // NOTE keep the hack here ?
    if (src.elembits() == 32)
    {
        if (opt.use_bf16_storage || opt.use_bf16_packed)
        {
            Mat src_bf16;
            cast_float32_to_bfloat16(src, src_bf16, opt);

            record_upload(src_bf16, dst, opt, flatten);

            return;
        }
        else if (opt.use_fp16_storage || opt.use_fp16_packed)
        {
            Mat src_fp16;
            cast_float32_to_float16(src, src_fp16, opt);

            record_upload(src_fp16, dst, opt, flatten);

            return;
        }
    }

    const size_t flattened_size = (size_t)src.w * src.h * src.d * src.c;
    if (flatten && flattened_size > 2147483647)
    {
        NCNN_LOGE("prepared weight exceeds maximum flattened dimension");
        d->upload_error = -1;
        return;
    }
    Mat src_flattened = flatten ? src.reshape((int)flattened_size) : src;

    if (src_flattened.empty())
    {
        d->upload_error = -100;
        return;
    }

    // create dst
    dst.create_like(src_flattened, opt.blob_vkallocator);

    if (dst.empty())
    {
        d->upload_error = -100;
        return;
    }

    if (dst.data->storage_type == 2)
    {
        const size_t destination_offset = dst.offset;
        if (!(dst.data->access_flags & VK_ACCESS_HOST_WRITE_BIT) || !dst.data->external_ptr || destination_offset > dst.data->external_size
                || dst.buffer_capacity() > dst.data->external_size - destination_offset)
        {
            NCNN_LOGE("file buffer requires a writable external range during upload");
            d->upload_error = -1;
            return;
        }
        unsigned char* destination = (unsigned char*)dst.data->external_ptr;
        for (int b = 0; b < src_flattened.n; b++)
        {
            const Mat src_b = src_flattened.batch(b);
            const size_t offset = dst.batch(b).offset;
            const size_t size = src_b.total() * src_b.elemsize;
            const size_t end = b + 1 == src_flattened.n ? destination_offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            if (offset > end || end > dst.data->external_size || size > end - offset)
            {
                d->upload_error = -1;
                return;
            }
            memcpy(destination + offset, src_b.data, size);
            memset(destination + offset + size, 0, end - offset - size);
        }
        return;
    }

    d->pending_upload_total += dst.buffer_capacity();

    if (dst.mapped_ptr())
    {
        // memcpy src_flattened to device
        for (int b = 0; b < src_flattened.n; b++)
        {
            const Mat src_b = src_flattened.batch(b);
            VkMat dst_b = dst.batch(b);
            memcpy(dst_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
        }
        d->upload_error = dst.allocator->flush(dst.data);
        if (d->upload_error)
            return;

        // barrier device host-write @ null to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.data->offset;
            barrier.size = dst.data->capacity;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // mark device shader-readwrite @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        return;
    }

    // create staging
    VkMat dst_staging;
    dst_staging.create_like(src_flattened, opt.staging_vkallocator);

    if (dst_staging.empty())
    {
        d->upload_error = -100;
        return;
    }

    // memcpy src_flattened to staging
    for (int b = 0; b < src_flattened.n; b++)
    {
        const Mat src_b = src_flattened.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }
    d->upload_error = dst_staging.allocator->flush(dst_staging.data);
    if (d->upload_error)
        return;

    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue())
    {
        command_buffer = d->compute_command_buffer;
    }
    else
    {
        command_buffer = d->upload_command_buffer;
    }

    // barrier staging host-write @ null to transfer-read @ queue
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst_staging.buffer();
        barrier.offset = dst_staging.data->offset;
        barrier.size = dst_staging.data->capacity;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.srcOffset = dst_staging.buffer_offset();
        region.dstOffset = dst.buffer_offset();
        region.size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

        vkCmdCopyBuffer(command_buffer, dst_staging.buffer(), dst.buffer(), 1, &region);
    }

    if (vkdev->info.unified_compute_transfer_queue())
    {
        // barrier device transfer-write @ compute to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.data->offset;
            barrier.size = dst.data->capacity;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    else
    {
        // queue ownership transfer transfer-write @ transfer to shader-read @ compute

        // release
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index();
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
            barrier.buffer = dst.buffer();
            barrier.offset = dst.data->offset;
            barrier.size = dst.data->capacity;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(d->upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // acquire
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index();
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
            barrier.buffer = dst.buffer();
            barrier.offset = dst.data->offset;
            barrier.size = dst.data->capacity;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }

    // mark device shader-readwrite @ compute
    dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

int VkTransfer::submit_and_wait()
{
    if (d->upload_error)
        return d->upload_error;
    if (d->pending_upload_total == 0)
        return 0;

    //     NCNN_LOGE("submit_and_wait");

    // end command buffer
    {
        int ret = d->end_command_buffer();
        if (ret != 0)
            return ret;
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
    if (compute_queue == 0)
    {
        NCNN_LOGE("out of compute queue");
        return -1;
    }

    if (vkdev->info.unified_compute_transfer_queue())
    {
        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &d->compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, d->compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return -1;
            }
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index());
        if (transfer_queue == 0)
        {
            NCNN_LOGE("out of transfer queue");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
            return -1;
        }

        // submit upload compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &d->upload_command_buffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &d->upload_compute_semaphore;

            VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, d->upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return -1;
            }
        }
        {
            VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // FIXME

            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &d->upload_compute_semaphore;
            submitInfo.pWaitDstStageMask = &wait_dst_stage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &d->compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, d->compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return -1;
            }
        }

        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);

    // wait
    if (vkdev->info.unified_compute_transfer_queue())
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &d->compute_command_fence, VK_TRUE, (uint64_t)-1);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkWaitForFences failed %d", ret);
            return -1;
        }
    }
    else
    {
        VkFence fences[2] = {d->upload_command_fence, d->compute_command_fence};

        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 2, fences, VK_TRUE, (uint64_t)-1);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkWaitForFences failed %d", ret);
            return -1;
        }
    }

    d->pending_upload_total = 0;

    return 0;
}

int VkTransfer::reset()
{
    if (d->upload_error)
        return d->upload_error;

    d->upload_staging_buffers.clear();

    d->pending_upload_total = 0;

    // reset command buffer and fence
    {
        VkResult ret = vkResetCommandBuffer(d->compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            return -1;
        }
    }
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &d->compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkResult ret = vkResetCommandBuffer(d->upload_command_buffer, 0);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
                return -1;
            }
        }
        {
            VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &d->upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetFences failed %d", ret);
                return -1;
            }
        }
    }

    d->upload_error = d->begin_command_buffer();

    return d->upload_error;
}

int VkTransfer::error() const
{
    return d->upload_error;
}

uint64_t VkTransfer::pending_upload_total() const
{
    return d->pending_upload_total;
}

} // namespace ncnn

#endif // NCNN_VULKAN
