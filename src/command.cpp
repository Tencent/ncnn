// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"

#if NCNN_VULKAN

#include "option.h"
#include "pipeline.h"
#include <limits.h>

#if !defined _WIN32
#include <time.h>
#endif

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

namespace ncnn {

size_t weight_buffer_alignment(const VulkanDevice* vkdev);
int fit_weight_capacity(const VulkanDevice* vkdev, size_t memory_limit, bool automatic, bool host, size_t& capacity);
VkBufferMemory* create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t memory_limit, size_t& allocation_size, bool concurrent);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

static void wait_for_pending_commands()
{
#if defined _WIN32
    Sleep(1);
#else
    struct timespec delay = {0, 1000000};
    nanosleep(&delay, 0);
#endif
}

static bool is_deferred_buffer(const VkBufferMemory* memory)
{
    return memory && memory->storage_type != 0 && memory->source_size != 0;
}

static bool needs_host_copy(const VkMat& source)
{
    return source.data->storage_type == 2 || !source.buffer();
}

// Cache views retain their backing VkMat. Model allocators must outlive recorded work.
class VkWeightCacheAllocator : public VkAllocator
{
public:
    struct Region : VkBufferMemory
    {
        Region()
            : allocation(0)
        {
        }
        VkMat backing;
        VkBufferMemory* allocation;
    };
    explicit VkWeightCacheAllocator(const VulkanDevice* device)
        : VkAllocator(device)
    {
    }
    virtual VkBufferMemory* fastMalloc(size_t)
    {
        return new Region;
    }
    virtual void fastFree(VkBufferMemory* memory)
    {
        Region* region = static_cast<Region*>(memory);
        if (region->allocation)
            destroy_weight_buffer(vkdev, region->allocation);
        delete region;
    }
    virtual VkImageMemory* fastMalloc(int, int, int, size_t, int)
    {
        return 0;
    }
    virtual void fastFree(VkImageMemory*)
    {
    }
    VkMat adopt(VkBufferMemory* allocation)
    {
        if (!allocation)
            return VkMat();
        Region* region = new Region;
        static_cast<VkBufferMemory&>(*region) = *allocation;
        region->allocation = allocation;
        region->refcount = 1;
        VkMat owner(1, region, allocation->capacity, 1, this);
        owner.refcount = &region->refcount;
        return owner;
    }
};

// Append complete working sets in execution order. On wrap, retire the previous
// pass as a whole; no interval splitting, victim selection or cross-reset hits.
class VkWeightCache
{
public:
    struct weight_entry
    {
        VkMat target;
        uintptr_t source_identity;
        size_t last_task;
        bool valid;
    };
    struct buffer_range
    {
        size_t offset;
        size_t size;
    };
    struct weight_bindings
    {
        std::vector<VkMat> weights;
        std::vector<VkMat> targets;
        std::vector<bool> missing;
    };
    VkWeightCache(const VulkanDevice* device, size_t device_cache_size, size_t host_cache_size, size_t device_memory_limit, size_t host_memory_limit)
        : vkdev(device), weight_device_cache_size(device_cache_size), weight_host_cache_size(host_cache_size), weight_device_memory_limit(device_memory_limit), weight_host_memory_limit(host_memory_limit), automatic_device_size(!device_cache_size), automatic_host_size(!host_cache_size), view_allocator(device), host_size(0), device_size(0), alignment(weight_buffer_alignment(device)), device_head(0), host_head(0), device_reader(0), host_reader(0), unified(false), initialized(false)
    {
    }
    ~VkWeightCache()
    {
        clear();
    }
    void clear()
    {
        clear_entries();
        host_buffer.release();
        device_buffer.release();
        initialized = false;
    }
    void clear_entries()
    {
        device_entries.clear();
        device_head = host_head = 0;
        device_reader = host_reader = 0;
    }
    int initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size);
    int validate_source(const VkMat& source) const;
    size_t aligned_size(const VkMat& source) const
    {
        return alignSize(source.data->source_size, alignment);
    }
    bool has_reader(size_t task) const
    {
        return task && device_reader == task;
    }
    void reset_readers()
    {
        device_reader = host_reader = 0;
    }
    int prepare_bindings(weight_bindings& bindings, std::vector<weight_entry>& evicted);
    void mark_bindings_ready(const weight_bindings& bindings)
    {
        for (size_t i = 0; i < bindings.weights.size(); i++)
            device_entries[find_device_entry(bindings.weights[i])].valid = true;
    }
    void mark_bindings_reader(const weight_bindings& bindings, size_t task)
    {
        if (!bindings.weights.empty())
            device_reader = task;
    }
    VkMat allocate_host(size_t size, size_t& reader)
    {
        reader = 0;
        if (!size || size > host_size)
            return VkMat();
        if (size > host_size - host_head)
        {
            reader = host_reader;
            host_head = 0;
        }
        VkMat result = backing_span(true, host_head, size);
        host_head += size;
        return result;
    }
    void mark_host_reader(const VkMat& reservation, size_t task)
    {
        if (!host_buffer.empty() && reservation.buffer() == host_buffer.buffer())
            host_reader = task;
    }
    const VkMat& get_device_buffer() const
    {
        return device_buffer;
    }
    const VkMat& get_host_buffer() const
    {
        return host_buffer;
    }
    size_t get_host_size() const
    {
        return host_size;
    }
    size_t get_alignment() const
    {
        return alignment;
    }
    bool is_unified() const
    {
        return unified;
    }

private:
    int find_device_entry(const VkMat& source) const
    {
        for (size_t i = 0; i < device_entries.size(); i++)
            if (device_entries[i].source_identity == (uintptr_t)source.data)
                return (int)i;
        return -1;
    }
    VkMat backing_span(bool host, size_t offset, size_t size) const
    {
        const VkMat& owner = host ? host_buffer : device_buffer;
        VkWeightCacheAllocator::Region* memory = new VkWeightCacheAllocator::Region;
        static_cast<VkBufferMemory&>(*memory) = *owner.data;
        memory->backing = owner;
        memory->offset = owner.buffer_offset() + offset;
        memory->capacity = size;
        memory->source_ptr = 0;
        memory->source_size = 0;
        memory->refcount = 1;
        VkMat view(1, memory, size, 1, &view_allocator);
        view.refcount = &memory->refcount;
        return view;
    }
    int allocate()
    {
        size_t allocation_size;
        if (device_size && device_buffer.empty())
        {
            device_buffer = view_allocator.adopt(create_weight_buffer(vkdev, device_size, 0, unified, weight_device_memory_limit, allocation_size, true));
            if (device_buffer.empty())
                return -100;
        }
        if (host_size && host_buffer.empty())
        {
            host_buffer = view_allocator.adopt(create_weight_buffer(vkdev, host_size, 1, false, weight_host_memory_limit, allocation_size, true));
            if (host_buffer.empty())
                return -100;
        }
        return 0;
    }
    const VulkanDevice* vkdev;
    size_t weight_device_cache_size;
    size_t weight_host_cache_size;
    size_t weight_device_memory_limit;
    size_t weight_host_memory_limit;
    bool automatic_device_size;
    bool automatic_host_size;
    mutable VkWeightCacheAllocator view_allocator;
    VkMat host_buffer;
    VkMat device_buffer;
    size_t host_size;
    size_t device_size;
    size_t alignment;
    size_t device_head;
    size_t host_head;
    size_t device_reader;
    size_t host_reader;
    std::vector<weight_entry> device_entries;
    bool unified;
    bool initialized;
};

int VkWeightCache::validate_source(const VkMat& source) const
{
    const VkBufferMemory* src = source.data;
    if (source.allocator && source.allocator->vkdev != vkdev)
    {
        NCNN_LOGE("weight source and command must use the same Vulkan device");
        return -1;
    }
    if (src->storage_type == 2 && !(src->access_flags & VK_ACCESS_HOST_READ_BIT))
    {
        NCNN_LOGE("file weights must be finalized before submission");
        return -1;
    }
    if (!is_deferred_buffer(src) || source.offset > src->source_size
            || source.buffer_capacity() > src->source_size - source.offset
            || src->source_size > (size_t)-1 - alignment
            || (src->storage_type != 2 && src->storage_type != 1)
            || (needs_host_copy(source) && !src->source_ptr)
            || (!needs_host_copy(source) && alignSize(src->source_size, 4) > src->capacity))
    {
        NCNN_LOGE("invalid deferred buffer binding");
        return -1;
    }
    return 0;
}

int VkWeightCache::prepare_bindings(weight_bindings& bindings, std::vector<weight_entry>& evicted)
{
    bindings.targets.clear();
    bindings.missing.clear();
    evicted.clear();
    size_t required = 0;
    size_t missing = 0;
    for (size_t i = 0; i < bindings.weights.size(); i++)
    {
        const size_t size = aligned_size(bindings.weights[i]);
        if (size > device_size - required)
        {
            NCNN_LOGE("weight bindings exceed cache capacity");
            return -1;
        }
        required += size;
        if (find_device_entry(bindings.weights[i]) < 0)
            missing += size;
    }
    if (missing > device_size - device_head)
    {
        weight_entry previous;
        previous.target = backing_span(false, 0, device_head);
        previous.last_task = device_reader;
        previous.source_identity = 0;
        previous.valid = true;
        evicted.push_back(previous);
        device_entries.clear();
        device_head = 0;
    }
    for (size_t i = 0; i < bindings.weights.size(); i++)
    {
        const VkMat& source = bindings.weights[i];
        int index = find_device_entry(source);
        if (index < 0)
        {
            const size_t size = aligned_size(source);
            weight_entry entry;
            entry.target = backing_span(false, device_head, size);
            entry.source_identity = (uintptr_t)source.data;
            entry.last_task = 0;
            entry.valid = false;
            device_head += size;
            index = (int)device_entries.size();
            device_entries.push_back(entry);
        }
        bindings.targets.push_back(device_entries[index].target);
        bindings.missing.push_back(!device_entries[index].valid);
    }
    return 0;
}

int VkWeightCache::initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size)
{
    if (!minimum_device_size && !max_source_size && !max_file_size)
        return 0;

    size_t device_memory_limit = weight_device_memory_limit;
    size_t host_memory_limit = weight_host_memory_limit;
    unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    if (unified)
    {
        device_memory_limit = device_memory_limit && host_memory_limit ? std::min(device_memory_limit, host_memory_limit) : std::max(device_memory_limit, host_memory_limit);
        host_memory_limit = 0;
    }
    if (!initialized)
    {
        // Submission has collected the largest simultaneous working set.
        // Once allocated, this buffer never grows until the cache is cleared.
        size_t capacity = weight_device_cache_size;
        if (!capacity)
            capacity = std::max((size_t)128 * 1024 * 1024, std::max(minimum_device_size, max_source_size));
        if (capacity > (size_t)-1 - alignment)
            return -1;
        capacity = alignSize(capacity, alignment);
        int ret = fit_weight_capacity(vkdev, device_memory_limit, automatic_device_size, false, capacity);
        if (ret != 0)
            return ret;
        if (capacity < minimum_device_size || capacity < max_source_size)
        {
            NCNN_LOGE("weight cache exceeds command capacity or memory limit");
            return -1;
        }
        weight_device_memory_limit = device_memory_limit;
        weight_host_memory_limit = host_memory_limit;
        device_size = capacity;
        host_size = 0;
        ret = allocate();
        if (ret != 0)
            return ret;
        initialized = true;
    }
    else if (minimum_device_size > device_size || max_source_size > device_size)
    {
        NCNN_LOGE("recorded weight bindings exceed the fixed command cache; clear and configure a larger cache");
        return -1;
    }

    // Host-only sources do not need staging. A later submission may add file
    // sources without changing the retained device allocation.
    if (max_file_size && !unified && host_buffer.empty())
    {
        size_t capacity = weight_host_cache_size ? weight_host_cache_size : (size_t)16 * 1024 * 1024;
        if (capacity > (size_t)-1 - alignment)
            return -1;
        capacity = alignSize(capacity, alignment);
        int ret = fit_weight_capacity(vkdev, host_memory_limit, automatic_host_size, true, capacity);
        if (ret != 0)
            return ret;
        if (capacity < alignment)
        {
            NCNN_LOGE("weight host staging exceeds memory limit");
            return -1;
        }
        host_size = capacity;
        return allocate();
    }
    return 0;
}

class VkComputePrivate
{
public:
    explicit VkComputePrivate(const VulkanDevice* _vkdev);
    ~VkComputePrivate();

    int init();
    int begin_command_buffer();

    bool deferred() const
    {
        return weight_deferred || !vkdev->info.support_VK_KHR_push_descriptor();
    }
    void record_weight(const VkMat& mat);
    typedef VkWeightCache::buffer_range buffer_range;
    typedef VkWeightCache::weight_bindings weight_bindings;
    VkCommandPool transfer_command_pool;
    uint32_t transfer_queue_family_index;
    std::vector<VkSemaphore> compute_waits;
    std::vector<VkSemaphore> ready_uploads;
    std::vector<VkCommandBuffer> upload_command_buffers;
    std::vector<VkFence> upload_fences;
    std::vector<VkSemaphore> upload_compute_semaphores;
    std::vector<bool> upload_pending;
    std::vector<VkMat> upload_sources;
    std::vector<VkMat> upload_targets;
    std::vector<VkSemaphore> upload_queue_signals;
#if NCNN_BENCHMARK
    uint64_t bytes_read;
    uint64_t bytes_uploaded;
    double weight_start_time;
#endif
    int allocate_upload_command_buffer(size_t task);
    int begin_upload_command_buffer(size_t task);
    int record_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last);
    int submit_upload(int task, const VkSubmitInfo& info);
    int upload_source_to_device_cache(const VkMat& source, const VkMat& target);
    int record_device_upload(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last);
    int wait_fence(VkFence fence);
    int create_upload_task();
    int allocate_host_range(size_t size, VkMat& reservation);
    int fill_host_cache(const VkMat& source, VkMat& target);
    void record_upload_release_to_compute(int task, const buffer_range& range);
    int submit_immediate();
    int begin_weight_execution();
    int create_compute_task(bool ordinary);
    int begin_compute_batch();
    void record_weight_barriers(int task);
    int end_compute_task(int task, bool release);
    int submit_compute_command(int task, const VkSubmitInfo& info, bool ordered);
    int wait_compute(size_t task);
    int finish_execution(int result = 0);
    void drain_execution();
    void clear_execution();
    void record_compute_acquire_from_upload(int task, const buffer_range& range);
    weight_bindings current_bindings;
    int active_compute_task;
    bool weight_recording;
    std::vector<VkSemaphore> active_upload_waits;
    struct execution_action
    {
        enum
        {
            FILL,
            WAIT_COMPUTE,
            WAIT_UPLOAD,
            UPLOAD,
            COMPUTE
        } type;
        execution_action()
            : type(FILL), task(-1), signal(false), source_offset(0), size(0)
        {
        }
        int task;
        bool signal;
        std::vector<VkSemaphore> waits;
        VkMat source;
        VkMat target;
        size_t source_offset;
        size_t size;
    };
    std::vector<execution_action> execution_actions;
    int execute_actions();
    void record_fill(const VkMat& source, const VkMat& target, size_t offset, size_t size);
    int close_compute_task();

    VkCommandPool batch_command_pool;
    std::vector<VkCommandBuffer> compute_command_buffers;
    std::vector<VkFence> compute_fences;
    std::vector<VkSemaphore> compute_upload_semaphores;
    std::vector<bool> compute_release_pending;
    std::vector<bool> compute_pending;
    std::vector<VkSemaphore> compute_queue_signals;
    bool executing;
    int execution_error;
    int submit_managed();
    void clear_records();

    int resolve_weight(const VkMat& weight, VkDescriptorBufferInfo& descriptor) const;

    size_t weight_device_cache_size;
    size_t weight_host_cache_size;
    size_t weight_device_memory_limit;
    size_t weight_host_memory_limit;
    VkWeightCache* weight_cache;
    bool weight_deferred;

    int create_descriptorset(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template,
                             const unsigned char* descriptor_infos, VkDescriptorSet* descriptorset);

    const VulkanDevice* vkdev;

    VkCommandPool compute_command_pool;

    VkCommandBuffer compute_command_buffer;

    VkFence compute_command_fence;
    bool compute_command_pending;

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
        union
        {
            struct
            {
                VkBuffer src;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferCopy* regions;
                const VkMat* weight;
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
                const VkMat* weight;
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
                VkDescriptorSetLayout descriptorset_layout;
                VkDescriptorUpdateTemplateKHR update_template;
                uint32_t binding_count;
                const int* binding_types;
                unsigned char* data;
                uint32_t weight_count;
                const VkMat* weights;
                const size_t* weight_offsets;
            } bind_descriptorsets;
            struct
            {
                VkDescriptorUpdateTemplateKHR update_template;
                VkPipelineLayout pipeline_layout;
                unsigned char* data;
                uint32_t weight_count;
                const VkMat* weights;
                const size_t* weight_offsets;
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

    int resolve_weight_descriptors(uint32_t count, const VkMat* weights, const size_t* offsets, unsigned char* data) const;
    int record_deferred(VkCommandBuffer command_buffer, const record& r);
    int record_sources(const record& r, std::vector<VkMat>& weights);
    int prepare_records(size_t begin, size_t end);

    std::vector<record> deferred_records;

    uint64_t pending_dispatch_total;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
    bool query_results_ready;
#endif // NCNN_BENCHMARK
};

int VkComputePrivate::allocate_upload_command_buffer(size_t task)
{
    const VkDevice device = vkdev->vkdevice();
    const VkCommandPool pool = transfer_command_pool;
    VkCommandBuffer* command_buffer = &upload_command_buffers[task];
    VkResult ret;
    VkCommandBufferAllocateInfo commandBufferAllocateInfo;
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.pNext = 0;
    commandBufferAllocateInfo.commandBufferCount = 1;
    commandBufferAllocateInfo.commandPool = pool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ret = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, command_buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
        return -1;
    }
    return 0;
}

int VkComputePrivate::begin_upload_command_buffer(size_t task)
{
    const VkCommandBuffer command_buffer = upload_command_buffers[task];
    VkResult ret;
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;
    ret = vkBeginCommandBuffer(command_buffer, &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
        return -1;
    }
    return 0;
}

int VkComputePrivate::record_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last)
{
    const VkBufferMemory* src = source.data;
    int ret;
    if (!(src->usage_flags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return -1;
    ret = begin_upload_command_buffer(task);
    if (ret != 0)
        return ret;
    VkResult vkret;
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    if (src->mapped_ptr)
    {
        src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    }
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = src->buffer;
    barrier.offset = src_offset;
    barrier.size = size;
    vkCmdPipelineBarrier(upload_command_buffers[task], src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
    VkBufferCopy copy;
    copy.size = size;
    copy.srcOffset = src_offset;
    copy.dstOffset = target.buffer_offset() + target_offset;
    vkCmdCopyBuffer(upload_command_buffers[task], source.buffer(), target.buffer(), 1, &copy);
    if (last)
    {
        buffer_range range = {target.buffer_offset(), target.buffer_capacity()};
        record_upload_release_to_compute(task, range);
    }
    vkret = vkEndCommandBuffer(upload_command_buffers[task]);
    if (vkret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", vkret);
        return -1;
    }
    return 0;
}

int VkComputePrivate::submit_upload(int task, const VkSubmitInfo& info)
{
    VkSemaphore signal = 0;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
    if (vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &signal) != VK_SUCCESS)
        return -1;
    // Retain even an unsubmitted signal until execution cleanup.
    const VkSemaphore previous = upload_queue_signals.empty() ? 0 : upload_queue_signals.back();
    upload_queue_signals.push_back(signal);
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    for (uint32_t i = 0; i < info.waitSemaphoreCount; i++)
    {
        waits.push_back(info.pWaitSemaphores[i]);
        stages.push_back(info.pWaitDstStageMask[i]);
    }
    if (previous)
    {
        waits.push_back(previous);
        stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    std::vector<VkSemaphore> signals;
    for (uint32_t i = 0; i < info.signalSemaphoreCount; i++)
        signals.push_back(info.pSignalSemaphores[i]);
    signals.push_back(signal);
    VkSubmitInfo submitInfo = info;
    submitInfo.waitSemaphoreCount = (uint32_t)waits.size();
    submitInfo.pWaitSemaphores = waits.data();
    submitInfo.pWaitDstStageMask = stages.data();
    submitInfo.signalSemaphoreCount = (uint32_t)signals.size();
    submitInfo.pSignalSemaphores = signals.data();

    const uint32_t family = transfer_queue_family_index;
    VkQueue queue = vkdev->acquire_queue(family);
    if (!queue)
    {
        NCNN_LOGE("out of weight %s queue", "transfer");
        return -1;
    }
    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, upload_fences[task]);
    vkdev->reclaim_queue(family, queue);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return ret;
    }
    return 0;
}

int VkComputePrivate::upload_source_to_device_cache(const VkMat& source, const VkMat& target)
{
    const size_t size = target.buffer_capacity();
    if (weight_cache->is_unified() && needs_host_copy(source))
    {
        record_fill(source, target, 0, source.data->source_size);
        return 0;
    }
    if (!needs_host_copy(source))
    {
        int task = create_upload_task();
        if (task < 0) return task;
        return record_device_upload(task, source, target, source.buffer_offset(), 0, alignSize(source.data->source_size, 4), true);
    }
    if (size <= weight_cache->get_host_size())
    {
        VkMat host;
        int ret = fill_host_cache(source, host);
        if (ret) return ret;
        int task = create_upload_task();
        if (task < 0) return task;
        return record_device_upload(task, host, target, host.buffer_offset(), 0, alignSize(source.data->source_size, 4), true);
    }
    size_t alignment = weight_cache->get_alignment();
    size_t chunk_capacity = std::min((size_t)INT_MAX / alignment * alignment, std::max(alignment, weight_cache->get_host_size() / 2 / alignment * alignment));
    for (size_t offset = 0; offset < source.data->source_size;)
    {
        size_t size = std::min(chunk_capacity, source.data->source_size - offset);
        VkMat host;
        int ret = allocate_host_range(alignSize(size, alignment), host);
        if (ret) return ret;
        record_fill(source, host, offset, size);
        int task = create_upload_task();
        if (task < 0) return task;
        bool last = size == source.data->source_size - offset;
        ret = record_device_upload(task, host, target, host.buffer_offset(), offset, alignSize(size, 4), last);
        if (ret) return ret;
        offset += size;
    }
    return 0;
}

int VkComputePrivate::record_device_upload(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last)
{
    int ret = record_device_cache_copy(task, source, target, src_offset, target_offset, size, last);
    if (ret) return ret;
    execution_action action;
    action.type = execution_action::UPLOAD;
    action.task = task;
    action.signal = last;
    action.waits = compute_waits;
    execution_actions.push_back(action);
    compute_waits.clear();
    weight_cache->mark_host_reader(source, task + 1);
    upload_sources[task] = source;
    upload_targets[task] = target;
    if (last) ready_uploads.push_back(upload_compute_semaphores[task]);
#if NCNN_BENCHMARK
    bytes_uploaded += size;
#endif
    return 0;
}

int VkComputePrivate::wait_fence(VkFence fence)
{
    VkResult ret = vkGetFenceStatus(vkdev->vkdevice(), fence);
    if (ret == VK_SUCCESS)
        return 0;
    if (ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetFenceStatus failed %d", ret);
        return ret;
    }
    ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkWaitForFences failed %d", ret);
        return ret;
    }
    return 0;
}

int VkComputePrivate::create_upload_task()
{
    if (upload_command_buffers.size() >= (size_t)INT_MAX)
        return -1;
    const int task = (int)upload_command_buffers.size();
    upload_command_buffers.push_back(0);
    upload_fences.push_back(0);
    upload_compute_semaphores.push_back(0);
    upload_pending.push_back(false);
    upload_sources.push_back(VkMat());
    upload_targets.push_back(VkMat());
    int ret = allocate_upload_command_buffer(task);
    if (ret != 0)
        return ret;
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
    if (vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &upload_fences[task]) != VK_SUCCESS)
        return -1;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
    if (vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &upload_compute_semaphores[task]) != VK_SUCCESS)
        return -1;
    return task;
}

int VkComputePrivate::allocate_host_range(size_t size, VkMat& reservation)
{
    size_t reader;
    reservation = weight_cache->allocate_host(size, reader);
    if (reservation.empty()) return -1;
    if (reader)
    {
        execution_action action;
        action.type = execution_action::WAIT_UPLOAD;
        action.task = (int)reader - 1;
        execution_actions.push_back(action);
    }
    return 0;
}

int VkComputePrivate::fill_host_cache(const VkMat& source, VkMat& target)
{
    int ret = allocate_host_range(weight_cache->aligned_size(source), target);
    if (ret) return ret;
    record_fill(source, target, 0, source.data->source_size);
    return 0;
}

void VkComputePrivate::record_upload_release_to_compute(int task, const buffer_range& range)
{
    const VkCommandBuffer command_buffer = upload_command_buffers[task];
    // release device transfer-write @ transfer to null @ compute
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = weight_cache->get_device_buffer().buffer();
    barrier.offset = weight_cache->get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkComputePrivate::record_compute_acquire_from_upload(int task, const buffer_range& range)
{
    const VkCommandBuffer command_buffer = compute_command_buffers[task];
    // acquire device null @ transfer to shader-read @ compute
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = weight_cache->get_device_buffer().buffer();
    barrier.offset = weight_cache->get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

int VkComputePrivate::wait_compute(size_t task)
{
    if (!task || task > compute_fences.size()) return 0;
    if (!compute_pending[task - 1]) return 0;
    int ret = wait_fence(compute_fences[task - 1]);
    if (ret == 0 || ret == VK_ERROR_DEVICE_LOST) compute_pending[task - 1] = false;
    return ret;
}

int VkComputePrivate::submit_compute_command(int task, const VkSubmitInfo& info, bool ordered)
{
    VkSemaphore signal = 0;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
    if (ordered && vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &signal) != VK_SUCCESS)
        return -1;
    // Retain even an unsubmitted signal until execution cleanup.
    const VkSemaphore previous = compute_queue_signals.empty() ? 0 : compute_queue_signals.back();
    if (ordered)
        compute_queue_signals.push_back(signal);
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    for (uint32_t i = 0; i < info.waitSemaphoreCount; i++)
    {
        waits.push_back(info.pWaitSemaphores[i]);
        stages.push_back(info.pWaitDstStageMask[i]);
    }
    if (previous)
    {
        waits.push_back(previous);
        stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    std::vector<VkSemaphore> signals;
    for (uint32_t i = 0; i < info.signalSemaphoreCount; i++)
        signals.push_back(info.pSignalSemaphores[i]);
    if (ordered)
        signals.push_back(signal);
    VkSubmitInfo submitInfo = info;
    submitInfo.waitSemaphoreCount = (uint32_t)waits.size();
    submitInfo.pWaitSemaphores = waits.data();
    submitInfo.pWaitDstStageMask = stages.data();
    submitInfo.signalSemaphoreCount = (uint32_t)signals.size();
    submitInfo.pSignalSemaphores = signals.data();

    const uint32_t family = vkdev->info.compute_queue_family_index();
    VkQueue queue = vkdev->acquire_queue(family);
    if (!queue)
    {
        NCNN_LOGE("out of weight %s queue", "compute");
        return -1;
    }
    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, compute_fences[task]);
    vkdev->reclaim_queue(family, queue);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return ret;
    }
    return 0;
}

int VkComputePrivate::create_compute_task(bool ordinary)
{
    if (compute_command_buffers.size() >= (size_t)INT_MAX)
        return -1;
    const int task = (int)compute_command_buffers.size();
    compute_command_buffers.push_back(ordinary ? compute_command_buffer : 0);
    compute_fences.push_back(ordinary ? compute_command_fence : 0);
    compute_upload_semaphores.push_back(0);
    compute_release_pending.push_back(false);
    compute_pending.push_back(false);
    if (ordinary)
        return task;
    VkCommandBufferAllocateInfo allocate = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, batch_command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (vkAllocateCommandBuffers(vkdev->vkdevice(), &allocate, &compute_command_buffers[task]) != VK_SUCCESS)
        return -1;
    VkFenceCreateInfo fence = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
    if (vkCreateFence(vkdev->vkdevice(), &fence, 0, &compute_fences[task]) != VK_SUCCESS)
        return -1;
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};
    if (vkBeginCommandBuffer(compute_command_buffers[task], &begin) != VK_SUCCESS)
        return -1;
    return task;
}

int VkComputePrivate::begin_compute_batch()
{
    const bool first = compute_command_buffers.empty();
    int task = create_compute_task(first && vkdev->info.support_VK_KHR_push_descriptor());
#if NCNN_BENCHMARK
    if (task >= 0 && first && query_pool && !vkdev->info.support_VK_KHR_push_descriptor())
        vkCmdResetQueryPool(compute_command_buffers[task], query_pool, 0, query_count);
#endif
    return task;
}

void VkComputePrivate::record_weight_barriers(int task)
{
    for (size_t i = 0; i < current_bindings.weights.size(); i++)
    {
        const VkMat& source = current_bindings.weights[i];
        const VkMat& target = current_bindings.targets[i];
        if (!current_bindings.missing[i]) continue;
        if (!weight_cache->is_unified() || !needs_host_copy(source))
        {
            buffer_range range = {target.buffer_offset(), target.buffer_capacity()};
            record_compute_acquire_from_upload(task, range);
            continue;
        }
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, 0,
                                         VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                                         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, target.buffer(), target.buffer_offset(), target.buffer_capacity()
                                        };
        vkCmdPipelineBarrier(compute_command_buffers[task], VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
    }
}

int VkComputePrivate::end_compute_task(int task, bool release)
{
#if NCNN_COVERAGE
    if (!weight_recording)
        vkdev->record_shader_coverage_barrier(compute_command_buffers[task]);
#endif
    VkResult ret = vkEndCommandBuffer(compute_command_buffers[task]);
    if (ret != VK_SUCCESS) return ret;
    if (release)
    {
        VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
        ret = vkCreateSemaphore(vkdev->vkdevice(), &info, 0, &compute_upload_semaphores[task]);
    }
    return ret;
}

int VkComputePrivate::begin_weight_execution()
{
    weight_recording = true;
#if NCNN_COVERAGE
    if (vkdev->info.support_VK_KHR_push_descriptor())
        vkdev->record_shader_coverage_barrier(compute_command_buffer);
#endif
    weight_cache->reset_readers();
    transfer_queue_family_index = weight_cache->is_unified() ? vkdev->info.compute_queue_family_index() : vkdev->info.transfer_queue_family_index();
#if NCNN_BENCHMARK
    bytes_read = bytes_uploaded = 0;
    weight_start_time = get_current_time();
#endif
    VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, transfer_queue_family_index};
    VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &pool, 0, &transfer_command_pool);
    if (ret != VK_SUCCESS) return ret;
    pool.queueFamilyIndex = vkdev->info.compute_queue_family_index();
    return vkCreateCommandPool(vkdev->vkdevice(), &pool, 0, &batch_command_pool);
}

int VkComputePrivate::finish_execution(int result)
{
    if (result == VK_ERROR_DEVICE_LOST) execution_error = result;
    if (!executing) return result ? result : execution_error;
    bool incomplete = false;
    if (compute_command_pending)
    {
        int ret = wait_fence(compute_command_fence);
        if (ret == VK_ERROR_DEVICE_LOST) execution_error = ret;
        if (ret && result != VK_ERROR_DEVICE_LOST) result = ret;
        if (!ret || ret == VK_ERROR_DEVICE_LOST)
            compute_command_pending = false;
        else
            incomplete = true;
    }
    for (size_t i = 0; i < compute_fences.size(); i++)
    {
        if (!compute_pending[i]) continue;
        int ret = wait_fence(compute_fences[i]);
        if (ret == VK_ERROR_DEVICE_LOST) execution_error = ret;
        if (ret && result != VK_ERROR_DEVICE_LOST) result = ret;
        if (!ret || ret == VK_ERROR_DEVICE_LOST)
            compute_pending[i] = false;
        else
            incomplete = true;
    }
    for (size_t i = 0; i < upload_fences.size(); i++)
    {
        if (!upload_pending[i]) continue;
        int ret = wait_fence(upload_fences[i]);
        if (ret == VK_ERROR_DEVICE_LOST) execution_error = ret;
        if (ret && result != VK_ERROR_DEVICE_LOST) result = ret;
        if (!ret || ret == VK_ERROR_DEVICE_LOST)
            upload_pending[i] = false;
        else
            incomplete = true;
    }
    if (result && !execution_error) execution_error = result;
    if (incomplete) return execution_error;
    executing = false;
    if (!execution_error && weight_cache)
        weight_cache->reset_readers();
    if (weight_recording)
        clear_execution();
    return execution_error;
}

void VkComputePrivate::drain_execution()
{
    while (executing)
    {
        finish_execution(execution_error);
        if (executing)
            wait_for_pending_commands();
    }
}

void VkComputePrivate::clear_execution()
{
    // All compute producers/consumers and uploads are proven complete.
    for (size_t i = 0; i < compute_fences.size(); i++)
    {
        if (compute_fences[i] != compute_command_fence)
            vkDestroyFence(vkdev->vkdevice(), compute_fences[i], 0);
        vkDestroySemaphore(vkdev->vkdevice(), compute_upload_semaphores[i], 0);
    }
    for (size_t i = 0; i < compute_queue_signals.size(); i++)
        vkDestroySemaphore(vkdev->vkdevice(), compute_queue_signals[i], 0);
    if (batch_command_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), batch_command_pool, 0);
    batch_command_pool = 0;
    compute_command_buffers.clear();
    compute_fences.clear();
    compute_upload_semaphores.clear();
    compute_release_pending.clear();
    compute_pending.clear();
    compute_queue_signals.clear();

    for (size_t i = 0; i < upload_fences.size(); i++)
    {
        vkDestroyFence(vkdev->vkdevice(), upload_fences[i], 0);
        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphores[i], 0);
    }
    for (size_t i = 0; i < upload_queue_signals.size(); i++)
        vkDestroySemaphore(vkdev->vkdevice(), upload_queue_signals[i], 0);
    if (transfer_command_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    transfer_command_pool = 0;
    upload_command_buffers.clear();
    upload_fences.clear();
    upload_compute_semaphores.clear();
    upload_pending.clear();
    upload_sources.clear();
    upload_targets.clear();
    upload_queue_signals.clear();
    compute_waits.clear();
    ready_uploads.clear();
    execution_actions.clear();
    current_bindings = weight_bindings();
    active_upload_waits.clear();
    active_compute_task = -1;
    weight_recording = false;
#if NCNN_BENCHMARK
    if (weight_start_time)
        NCNN_LOGE("weight cache read=%llu upload=%llu time=%.2fms", (unsigned long long)bytes_read, (unsigned long long)bytes_uploaded, get_current_time() - weight_start_time);
    weight_start_time = 0;
#endif
}

int VkComputePrivate::submit_immediate()
{
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(compute_command_buffer);
#endif
    VkResult ret = vkEndCommandBuffer(compute_command_buffer);
    if (ret != VK_SUCCESS)
        return ret;
    const uint32_t family = vkdev->info.compute_queue_family_index();
    VkQueue queue = vkdev->acquire_queue(family);
    if (!queue)
        return -1;
    VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &compute_command_buffer, 0, 0};
    ret = vkQueueSubmit(queue, 1, &info, compute_command_fence);
    vkdev->reclaim_queue(family, queue);
    if (ret != VK_SUCCESS)
        return ret;
    compute_command_pending = true;
    executing = true;
    return finish_execution();
}

VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    weight_device_cache_size = 0;
    weight_host_cache_size = 0;
    weight_device_memory_limit = 0;
    weight_host_memory_limit = 0;
    weight_cache = 0;
    weight_deferred = false;
    batch_command_pool = 0;
    transfer_command_pool = 0;
    transfer_queue_family_index = vkdev->info.transfer_queue_family_index();
#if NCNN_BENCHMARK
    bytes_read = bytes_uploaded = 0;
    weight_start_time = 0;
#endif
    active_compute_task = -1;
    weight_recording = false;
    executing = false;
    execution_error = 0;

    compute_command_pool = 0;
    compute_command_buffer = 0;
    compute_command_fence = 0;
    compute_command_pending = false;

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
    drain_execution();
    if (compute_command_buffer)
        vkResetCommandBuffer(compute_command_buffer, 0);
    clear_execution();
    clear_records();
    delete weight_cache;

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
        for (size_t i = 0; i < descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), descriptor_pools[i], 1, &descriptorsets[i]);
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

    if (compute_command_buffer)
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
        commandBufferAllocateInfo.commandBufferCount = 1;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

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
        int ret = begin_command_buffer();
        if (ret != 0)
            return ret;

#if NCNN_BENCHMARK
        if (query_pool)
        {
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
        }
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

void VkComputePrivate::record_weight(const VkMat& mat)
{
    if (!is_deferred_buffer(mat.data))
        return;
    if (!weight_cache)
        weight_cache = new VkWeightCache(vkdev, weight_device_cache_size, weight_host_cache_size, weight_device_memory_limit, weight_host_memory_limit);
    weight_deferred = true;
}

void VkComputePrivate::clear_records()
{
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        const record& r = deferred_records[i];
        switch (r.type)
        {
        case record::TYPE_copy_buffer:
            delete r.copy_buffer.weight;
            delete[] r.copy_buffer.regions;
            break;
        case record::TYPE_copy_image:
            delete[] r.copy_image.regions;
            break;
        case record::TYPE_copy_buffer_to_image:
            delete r.copy_buffer_to_image.weight;
            delete[] r.copy_buffer_to_image.regions;
            break;
        case record::TYPE_copy_image_to_buffer:
            delete[] r.copy_image_to_buffer.regions;
            break;
        case record::TYPE_bind_descriptorsets:
            delete[] r.bind_descriptorsets.binding_types;
            delete[] r.bind_descriptorsets.data;
            delete[] r.bind_descriptorsets.weights;
            delete[] r.bind_descriptorsets.weight_offsets;
            break;
        case record::TYPE_push_descriptors:
            delete[] r.push_descriptors.data;
            delete[] r.push_descriptors.weights;
            delete[] r.push_descriptors.weight_offsets;
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
    deferred_records.clear();
    if (weight_cache)
        weight_cache->clear_entries();
}

int VkComputePrivate::create_descriptorset(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template,
        const unsigned char* descriptor_infos, VkDescriptorSet* descriptorset)
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

    // VUID-VkDescriptorPoolSize-descriptorCount-00302: each
    // descriptorCount must be > 0. Skip unused descriptor types
    // (common on buffer-only compute shaders; Mesa v3dv asserts).
    // See https://github.com/Tencent/ncnn/issues/6951
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

    ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, descriptorset);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
        vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pool, 0);
        return -1;
    }
    descriptor_pools.push_back(descriptor_pool);
    descriptorsets.push_back(*descriptorset);

    if (vkdev->info.support_VK_KHR_descriptor_update_template())
    {
        vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), *descriptorset, descriptor_update_template, descriptor_infos);
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
            writeDescriptorSets[i].dstSet = *descriptorset;
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

int VkComputePrivate::resolve_weight_descriptors(uint32_t count, const VkMat* weights, const size_t* offsets, unsigned char* data) const
{
    for (uint32_t i = 0; i < count; i++)
    {
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight(weights[i], descriptor);
        if (ret != 0)
            return ret;
        memcpy(data + offsets[i], &descriptor, sizeof(descriptor));
    }
    return 0;
}

int VkComputePrivate::record_sources(const record& r, std::vector<VkMat>& weights)
{
    const VkMat* sources = 0;
    size_t count = 0;
    switch (r.type)
    {
    case record::TYPE_bind_descriptorsets:
        sources = r.bind_descriptorsets.weights;
        count = r.bind_descriptorsets.weight_count;
        break;
    case record::TYPE_push_descriptors:
        sources = r.push_descriptors.weights;
        count = r.push_descriptors.weight_count;
        break;
    case record::TYPE_copy_buffer:
        sources = r.copy_buffer.weight;
        count = sources ? 1 : 0;
        break;
    case record::TYPE_copy_buffer_to_image:
        sources = r.copy_buffer_to_image.weight;
        count = sources ? 1 : 0;
        break;
    default:
        return 0;
    }
    for (size_t i = 0; i < count; i++)
    {
        int ret = weight_cache->validate_source(sources[i]);
        if (ret != 0)
            return ret;
        bool found = false;
        for (size_t j = 0; j < weights.size(); j++)
            if (weights[j].data == sources[i].data)
                found = true;
        if (!found)
        {
            weights.push_back(sources[i]);
            // Materialize the complete allocation; the record retains its view.
#if NCNN_BATCH
            weights.back().offset = 0;
#endif
        }
    }
    return 0;
}

int VkComputePrivate::submit_managed()
{
    size_t minimum_device_size = 0;
    size_t max_source_size = 0;
    size_t max_file_size = 0;
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        std::vector<VkMat> weights;
        int ret = record_sources(deferred_records[i], weights);
        if (ret != 0)
            return ret;
        size_t total = 0;
        for (size_t j = 0; j < weights.size(); j++)
        {
            const size_t size = weight_cache->aligned_size(weights[j]);
            if (size > (size_t)-1 - total)
                return -1;
            total += size;
            max_source_size = std::max(max_source_size, size);
            if (needs_host_copy(weights[j]))
                max_file_size = std::max(max_file_size, size);
        }
        minimum_device_size = std::max(minimum_device_size, total);
    }

    int ret = weight_cache->initialize(minimum_device_size, max_source_size, max_file_size);
    if (ret != 0)
        return ret;
    ret = begin_weight_execution();
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < deferred_records.size();)
    {
        size_t end = i + 1;
        if (deferred_records[i].type == record::TYPE_bind_pipeline)
        {
            // Keep pipeline state and its dispatch in the same command buffer.
            while (end < deferred_records.size() && deferred_records[end - 1].type != record::TYPE_dispatch)
                end++;
        }
        ret = prepare_records(i, end);
        if (ret != 0)
            return ret;
        i = end;
    }
    ret = close_compute_task();
    if (ret != 0)
        return ret;

    executing = true;
    ret = execute_actions();
    return finish_execution(ret);
}

int VkComputePrivate::close_compute_task()
{
    if (active_compute_task < 0) return 0;
    int ret = end_compute_task(active_compute_task, true);
    if (ret) return ret;
    execution_action action;
    action.type = execution_action::COMPUTE;
    action.task = active_compute_task;
    action.waits = active_upload_waits;
    action.signal = true;
    action.target = weight_cache->get_device_buffer();
    execution_actions.push_back(action);
    compute_release_pending[active_compute_task] = true;
    active_compute_task = -1;
    active_upload_waits.clear();
    return 0;
}

int VkComputePrivate::prepare_records(size_t begin, size_t end)
{
    const record& r = deferred_records[begin];
    if (r.type == record::TYPE_post_download || r.type == record::TYPE_post_cast_float16_to_float32 || r.type == record::TYPE_post_cast_bfloat16_to_float32) return 0;
    current_bindings = weight_bindings();
    int ret;
    for (size_t i = begin; i < end; i++)
    {
        ret = record_sources(deferred_records[i], current_bindings.weights);
        if (ret) return ret;
    }
    const bool prior_reader = active_compute_task >= 0 && weight_cache->has_reader(active_compute_task + 1);
    std::vector<VkWeightCache::weight_entry> previous;
    ret = weight_cache->prepare_bindings(current_bindings, previous);
    if (ret) return ret;
    bool gpu_upload = false;
    bool cpu_upload = false;
    for (size_t i = 0; i < current_bindings.weights.size(); i++)
    {
        if (!current_bindings.missing[i]) continue;
        if (weight_cache->is_unified() && needs_host_copy(current_bindings.weights[i]))
            cpu_upload = true;
        else
            gpu_upload = true;
    }
    const size_t reader = previous.empty() ? 0 : previous[0].last_task;
    if (active_compute_task >= 0 && (reader == (size_t)active_compute_task + 1 || (gpu_upload && prior_reader)))
    {
        ret = close_compute_task();
        if (ret) return ret;
    }
    if (cpu_upload && reader)
    {
        execution_action action;
        action.type = execution_action::WAIT_COMPUTE;
        action.task = (int)reader - 1;
        execution_actions.push_back(action);
    }
    const size_t upload_producer = gpu_upload ? reader : 0;
    ready_uploads.clear();
    compute_waits.clear();
    if (gpu_upload && upload_producer && compute_release_pending[upload_producer - 1])
        compute_waits.push_back(compute_upload_semaphores[upload_producer - 1]);
    for (size_t i = 0; i < current_bindings.weights.size(); i++)
    {
        if (!current_bindings.missing[i]) continue;
        ret = upload_source_to_device_cache(current_bindings.weights[i], current_bindings.targets[i]);
        if (ret) return ret;
    }
    // A consumed producer is covered by the ordered upload stream. Do not
    // consume unrelated signals: free-range uploads may overlap old readers.
    if (!ready_uploads.empty() && upload_producer)
        compute_release_pending[upload_producer - 1] = false;
    active_upload_waits.insert(active_upload_waits.end(), ready_uploads.begin(), ready_uploads.end());
    if (active_compute_task < 0) active_compute_task = begin_compute_batch();
    if (active_compute_task < 0) return active_compute_task;
    record_weight_barriers(active_compute_task);
    for (size_t i = begin; i < end; i++)
    {
        ret = record_deferred(compute_command_buffers[active_compute_task], deferred_records[i]);
        if (ret) return ret;
    }
    weight_cache->mark_bindings_ready(current_bindings);
    weight_cache->mark_bindings_reader(current_bindings, active_compute_task + 1);
    return 0;
}

void VkComputePrivate::record_fill(const VkMat& source, const VkMat& target, size_t offset, size_t size)
{
    execution_action action;
    action.type = execution_action::FILL;
    action.source = source;
    action.target = target;
    action.source_offset = offset;
    action.size = size;
    execution_actions.push_back(action);
}

int VkComputePrivate::execute_actions()
{
    for (size_t i = 0; i < execution_actions.size(); i++)
    {
        const execution_action& action = execution_actions[i];
        int ret = 0;
        if (action.type == execution_action::FILL)
        {
            unsigned char* dst = (unsigned char*)action.target.mapped_ptr();
            memcpy(dst, (const unsigned char*)action.source.data->source_ptr + action.source_offset, action.size);
            memset(dst + action.size, 0, action.target.buffer_capacity() - action.size);
            ret = flush_weight_buffer(vkdev, action.target.data, 0, action.target.buffer_capacity());
#if NCNN_BENCHMARK
            if (action.source.data->storage_type == 2) bytes_read += action.size;
#endif
        }
        else if (action.type == execution_action::WAIT_COMPUTE)
            ret = wait_compute(action.task + 1);
        else if (action.type == execution_action::WAIT_UPLOAD)
        {
            if (upload_pending[action.task])
            {
                ret = wait_fence(upload_fences[action.task]);
                if (!ret || ret == VK_ERROR_DEVICE_LOST) upload_pending[action.task] = false;
            }
        }
        else
        {
            const int task = action.task;
            std::vector<VkPipelineStageFlags> stages(action.waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            const bool compute = action.type == execution_action::COMPUTE;
            VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, (uint32_t)action.waits.size(), action.waits.data(), stages.data(),
                                 1, compute ? &compute_command_buffers[task] : &upload_command_buffers[task], action.signal ? 1u : 0u,
                                 action.signal ? (compute ? &compute_upload_semaphores[task] : &upload_compute_semaphores[task]) : 0
                                };
            if (compute)
            {
                ret = submit_compute_command(task, info, true);
                if (!ret) compute_pending[task] = true;
            }
            else
            {
                ret = submit_upload(task, info);
                if (!ret) upload_pending[task] = true;
            }
        }
        if (ret) return ret;
    }

    return 0;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev))
{
}

VkCompute::~VkCompute()
{
    delete d;
}

int VkCompute::set_weight_memory_limit(size_t device_memory_limit, size_t host_memory_limit, size_t device_cache_size, size_t host_cache_size)
{
    if (d->weight_cache || d->executing || d->execution_error == VK_ERROR_DEVICE_LOST || !d->deferred_records.empty() || d->pending_dispatch_total)
    {
        NCNN_LOGE("configure weight memory before recording; reset and clear the weight cache before reconfiguration");
        return -1;
    }

    d->weight_device_memory_limit = device_memory_limit;
    d->weight_host_memory_limit = host_memory_limit;
    d->weight_device_cache_size = device_cache_size;
    d->weight_host_cache_size = host_cache_size;
    return 0;
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

    if (src_fp16.empty())
        return;

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
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
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
    d->record_weight(src);

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
    if (dst_staging.empty())
        return;

    Mat dst_fp16;
    record_clone(dst_staging, dst_fp16, opt);
    if (dst_fp16.empty())
        return;

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

            if (dst.empty())
                return;

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32;
            r.post_cast_bfloat16_to_float32.download_post_mat_bf16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_bfloat16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_bfloat16_to_float32.num_threads = opt.num_threads;
            d->deferred_records.push_back(r);
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

            if (dst.empty())
                return;

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_float16_to_float32;
            r.post_cast_float16_to_float32.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_float16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_float16_to_float32.num_threads = opt.num_threads;
            d->deferred_records.push_back(r);
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
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
        return;

    // memcpy src to device
    for (int b = 0; b < src.n; b++)
    {
        const Mat src_b = src.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }

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
    if (dst_staging.empty())
        return;

    // staging to image
    record_clone(dst_staging, dst, opt);

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_clone(const VkMat& src, Mat& dst, const Option& opt)
{
    d->record_weight(src);

    //     NCNN_LOGE("record_clone buffer to host");

    VkMat src_staging;
    if (!src.mapped_ptr() || is_deferred_buffer(src.data))
    {
        Option opt_staging = opt;
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
        record_clone(src, src_staging, opt_staging);
        if (src_staging.empty())
            return;
    }
    else
    {
        src_staging = src;
    }
    if (!src_staging.mapped_ptr())
        return;

    // create dst
    dst.create_like(src_staging, opt.blob_allocator);
    if (dst.empty())
        return;

    // barrier device any @ compute to host-read @ compute
    if (src_staging.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || src_staging.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src_staging.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src_staging.buffer();
        barriers[0].offset = src_staging.data->offset;
        barriers[0].size = src_staging.data->capacity;

        VkPipelineStageFlags src_stage = src_staging.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
        }

        // mark device host-read @ any
        src_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        src_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    // stash download post buffer and mat
    d->download_post_buffers.push_back(src_staging);
    d->download_post_mats_fp16.push_back(dst);

    // post memcpy device to dst
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_post_download;
        r.post_download.download_post_buffer_mat_offset = d->download_post_buffers.size() - 1;
        r.post_download.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
        d->deferred_records.push_back(r);
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
    if (src_staging.empty())
        return;

    // staging to host
    record_clone(src_staging, dst, opt);
}

void VkCompute::record_clone(const VkMat& src, VkMat& dst, const Option& opt)
{
    d->record_weight(src);

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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
        {
            vkCmdCopyBuffer(d->compute_command_buffer, src.buffer(), dst.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer;
            r.copy_buffer.src = src.buffer();
            r.copy_buffer.weight = is_deferred_buffer(src.data) ? new VkMat(src) : 0;
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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
    d->record_weight(src);

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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
        {
            vkCmdCopyBufferToImage(d->compute_command_buffer, src.buffer(), dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer_to_image;
            r.copy_buffer_to_image.src = src.buffer();
            r.copy_buffer_to_image.weight = is_deferred_buffer(src.data) ? new VkMat(src) : 0;
            r.copy_buffer_to_image.dst = dst.image();
            r.copy_buffer_to_image.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            r.copy_buffer_to_image.region_count = region_count;
            r.copy_buffer_to_image.regions = regions;
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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
        d->record_weight(buffer_bindings[i]);
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

    // record bind pipeline
    {
        if (d->deferred())
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_bind_pipeline;
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline();
#if NCNN_COVERAGE
            r.bind_pipeline.pipeline_layout = pipeline->pipeline_layout();
#endif
            d->deferred_records.push_back(r);
        }
        else
        {
            vkCmdBindPipeline(d->compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
            vkdev->bind_shader_coverage(d->compute_command_buffer, pipeline->pipeline_layout());
#endif
        }
    }

    // record update bindings
    if (binding_count > 0)
    {
        uint32_t weight_count = 0;
        for (int i = 0; i < buffer_binding_count; i++)
            if (is_deferred_buffer(buffer_bindings[i].data))
                weight_count++;
        VkMat* weights = weight_count ? new VkMat[weight_count] : 0;
        size_t* weight_offsets = weight_count ? new size_t[weight_count] : 0;

        std::vector<unsigned char> descriptorInfos(sizeof(VkDescriptorBufferInfo) * buffer_binding_count + sizeof(VkDescriptorImageInfo) * image_binding_count);
        unsigned char* p_descriptorInfos = descriptorInfos.data();
        int bi = 0;
        int ii = 0;
        uint32_t wi = 0;
        for (int i = 0; i < binding_count; i++)
        {
            int binding_type = shader_info.binding_types[i];
            if (binding_type == 1)
            {
                const VkMat& binding = buffer_bindings[bi].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[bi];
                bi++;
                VkDescriptorBufferInfo descriptorBufferInfo;
                descriptorBufferInfo.buffer = binding.buffer();
                descriptorBufferInfo.offset = binding.buffer_offset();
                descriptorBufferInfo.range = binding.buffer_capacity();
                if (is_deferred_buffer(binding.data))
                {
                    weights[wi] = binding;
                    weight_offsets[wi] = p_descriptorInfos - descriptorInfos.data();
                    wi++;
                }
                memcpy(p_descriptorInfos, &descriptorBufferInfo, sizeof(descriptorBufferInfo));
                p_descriptorInfos += sizeof(descriptorBufferInfo);
            }
            else // if (binding_type == 2 || binding_type == 3)
            {
                const VkImageMat& binding = image_bindings[ii].empty() ? vkdev->get_dummy_image() : image_bindings[ii];
                ii++;
                // The descriptor layout supplies the immutable nearest sampler.
                VkDescriptorImageInfo descriptorImageInfo;
                descriptorImageInfo.sampler = 0;
                descriptorImageInfo.imageView = binding.imageview();
                descriptorImageInfo.imageLayout = binding.data->image_layout;
                memcpy(p_descriptorInfos, &descriptorImageInfo, sizeof(descriptorImageInfo));
                p_descriptorInfos += sizeof(descriptorImageInfo);
            }
        }

        if (d->deferred())
        {
            unsigned char* data = new unsigned char[descriptorInfos.size()];
            memcpy(data, descriptorInfos.data(), descriptorInfos.size());
            VkComputePrivate::record r;
            if (vkdev->info.support_VK_KHR_push_descriptor())
            {
                r.type = VkComputePrivate::record::TYPE_push_descriptors;
                r.push_descriptors.update_template = pipeline->descriptor_update_template();
                r.push_descriptors.pipeline_layout = pipeline->pipeline_layout();
                r.push_descriptors.data = data;
                r.push_descriptors.weight_count = weight_count;
                r.push_descriptors.weights = weights;
                r.push_descriptors.weight_offsets = weight_offsets;
            }
            else
            {
                int* binding_types = new int[binding_count];
                memcpy(binding_types, shader_info.binding_types, binding_count * sizeof(int));
                r.type = VkComputePrivate::record::TYPE_bind_descriptorsets;
                r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
                r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout();
                r.bind_descriptorsets.descriptorset_layout = pipeline->descriptorset_layout();
                r.bind_descriptorsets.update_template = pipeline->descriptor_update_template();
                r.bind_descriptorsets.binding_count = binding_count;
                r.bind_descriptorsets.binding_types = binding_types;
                r.bind_descriptorsets.data = data;
                r.bind_descriptorsets.weight_count = weight_count;
                r.bind_descriptorsets.weights = weights;
                r.bind_descriptorsets.weight_offsets = weight_offsets;
            }
            d->deferred_records.push_back(r);
        }
        else
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffer, pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, descriptorInfos.data());
        }
    }

    // record push constants
    if (constant_count > 0)
    {
        if (d->deferred())
        {
            const size_t size = constant_count * sizeof(vk_constant_type);
            unsigned char* values = new unsigned char[size];
            memcpy(values, constants.data(), size);
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_push_constants;
            r.push_constants.pipeline_layout = pipeline->pipeline_layout();
            r.push_constants.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
            r.push_constants.size = size;
            r.push_constants.values = values;
            d->deferred_records.push_back(r);
        }
        else
        {
            vkCmdPushConstants(d->compute_command_buffer, pipeline->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dispatcher.h * (dispatcher.d ? dispatcher.d : 1) + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        if (d->deferred())
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_dispatch;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            d->deferred_records.push_back(r);
        }
        else
        {
            vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }

        d->pending_dispatch_total += (uint64_t)group_count_x * group_count_y * group_count_z;
    }
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (!d->deferred())
    {
        if (d->query_pool)
        {
            vkCmdWriteTimestamp(d->compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->query_pool, query);
        }
    }
    else
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_write_timestamp;
        r.write_timestamp.query = query;
        d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
        }
    }

    VkDescriptorImageInfo descriptorImageInfo;
    descriptorImageInfo.sampler = pipeline->sampler;
    descriptorImageInfo.imageView = src.imageview();
    descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo descriptorBufferInfo;
    descriptorBufferInfo.buffer = dst.buffer();
    descriptorBufferInfo.offset = dst.buffer_offset();
    descriptorBufferInfo.range = dst.total() * dst.elemsize;

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

    const uint32_t group_count_x = (dst.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
    const uint32_t group_count_y = (dst.h + pipeline->local_size_y() - 1) / pipeline->local_size_y();
    const uint32_t group_count_z = (dst.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();
    if (d->deferred())
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_bind_pipeline;
        r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        r.bind_pipeline.pipeline = pipeline->pipeline();
#if NCNN_COVERAGE
        r.bind_pipeline.pipeline_layout = pipeline->pipeline_layout();
#endif
        d->deferred_records.push_back(r);

        unsigned char* data = new unsigned char[sizeof(info)];
        memcpy(data, &info, sizeof(info));
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            r.type = VkComputePrivate::record::TYPE_push_descriptors;
            r.push_descriptors.update_template = pipeline->descriptor_update_template();
            r.push_descriptors.pipeline_layout = pipeline->pipeline_layout();
            r.push_descriptors.data = data;
            r.push_descriptors.weight_count = 0;
            r.push_descriptors.weights = 0;
            r.push_descriptors.weight_offsets = 0;
        }
        else
        {
            r.type = VkComputePrivate::record::TYPE_bind_descriptorsets;
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout();
            r.bind_descriptorsets.descriptorset_layout = pipeline->descriptorset_layout();
            r.bind_descriptorsets.update_template = pipeline->descriptor_update_template();
            r.bind_descriptorsets.binding_count = 3;
            r.bind_descriptorsets.binding_types = new int[3] {3, 1, 1};
            r.bind_descriptorsets.data = data;
            r.bind_descriptorsets.weight_count = 0;
            r.bind_descriptorsets.weights = 0;
            r.bind_descriptorsets.weight_offsets = 0;
        }
        d->deferred_records.push_back(r);

        r.type = VkComputePrivate::record::TYPE_dispatch;
        r.dispatch.group_count_x = group_count_x;
        r.dispatch.group_count_y = group_count_y;
        r.dispatch.group_count_z = group_count_z;
        d->deferred_records.push_back(r);
        return;
    }

    vkCmdBindPipeline(d->compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
    vkdev->bind_shader_coverage(d->compute_command_buffer, pipeline->pipeline_layout());
#endif
    vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffer, pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, &info);
    vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

int VkComputePrivate::record_deferred(VkCommandBuffer command_buffer, const record& r)
{
    switch (r.type)
    {
    case VkComputePrivate::record::TYPE_copy_buffer:
    {
        if (!r.copy_buffer.weight)
        {
            vkCmdCopyBuffer(command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight(*r.copy_buffer.weight, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferCopy> regions(r.copy_buffer.region_count);
        for (uint32_t i = 0; i < r.copy_buffer.region_count; i++)
        {
            regions[i] = r.copy_buffer.regions[i];
            regions[i].srcOffset = descriptor.offset + regions[i].srcOffset - r.copy_buffer.weight->buffer_offset();
        }
        vkCmdCopyBuffer(command_buffer, descriptor.buffer, r.copy_buffer.dst, r.copy_buffer.region_count, regions.data());
        break;
    }
    case VkComputePrivate::record::TYPE_copy_image:
    {
        vkCmdCopyImage(command_buffer, r.copy_image.src, r.copy_image.src_layout, r.copy_image.dst, r.copy_image.dst_layout, r.copy_image.region_count, r.copy_image.regions);
        break;
    }
    case VkComputePrivate::record::TYPE_copy_buffer_to_image:
    {
        if (!r.copy_buffer_to_image.weight)
        {
            vkCmdCopyBufferToImage(command_buffer, r.copy_buffer_to_image.src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, r.copy_buffer_to_image.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight(*r.copy_buffer_to_image.weight, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferImageCopy> regions(r.copy_buffer_to_image.region_count);
        for (uint32_t i = 0; i < r.copy_buffer_to_image.region_count; i++)
        {
            regions[i] = r.copy_buffer_to_image.regions[i];
            regions[i].bufferOffset = descriptor.offset + regions[i].bufferOffset - r.copy_buffer_to_image.weight->buffer_offset();
        }
        vkCmdCopyBufferToImage(command_buffer, descriptor.buffer, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, regions.data());
        break;
    }
    case VkComputePrivate::record::TYPE_copy_image_to_buffer:
    {
        vkCmdCopyImageToBuffer(command_buffer, r.copy_image_to_buffer.src, r.copy_image_to_buffer.layout, r.copy_image_to_buffer.dst, r.copy_image_to_buffer.region_count, r.copy_image_to_buffer.regions);
        break;
    }
    case VkComputePrivate::record::TYPE_bind_pipeline:
    {
        vkCmdBindPipeline(command_buffer, r.bind_pipeline.bind_point, r.bind_pipeline.pipeline);
#if NCNN_COVERAGE
        vkdev->bind_shader_coverage(command_buffer, r.bind_pipeline.pipeline_layout);
#endif
        break;
    }
    case VkComputePrivate::record::TYPE_bind_descriptorsets:
    {
        int ret = resolve_weight_descriptors(r.bind_descriptorsets.weight_count, r.bind_descriptorsets.weights, r.bind_descriptorsets.weight_offsets, r.bind_descriptorsets.data);
        if (ret != 0)
            return ret;
        VkDescriptorSet descriptorset;
        ret = create_descriptorset(r.bind_descriptorsets.binding_count, r.bind_descriptorsets.binding_types, r.bind_descriptorsets.descriptorset_layout,
                                   r.bind_descriptorsets.update_template, r.bind_descriptorsets.data, &descriptorset);
        if (ret != 0)
            return ret;
        vkCmdBindDescriptorSets(command_buffer, r.bind_descriptorsets.bind_point, r.bind_descriptorsets.pipeline_layout, 0, 1, &descriptorset, 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_push_descriptors:
    {
        int ret = resolve_weight_descriptors(r.push_descriptors.weight_count, r.push_descriptors.weights, r.push_descriptors.weight_offsets, r.push_descriptors.data);
        if (ret != 0)
            return ret;
        vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, r.push_descriptors.update_template, r.push_descriptors.pipeline_layout, 0, r.push_descriptors.data);
        break;
    }
    case VkComputePrivate::record::TYPE_push_constants:
    {
        vkCmdPushConstants(command_buffer, r.push_constants.pipeline_layout, r.push_constants.stage_flags, 0, r.push_constants.size, r.push_constants.values);
        break;
    }
    case VkComputePrivate::record::TYPE_dispatch:
    {
        vkCmdDispatch(command_buffer, r.dispatch.group_count_x, r.dispatch.group_count_y, r.dispatch.group_count_z);
#if NCNN_COVERAGE
        if (weight_recording)
            vkdev->record_shader_coverage_barrier(command_buffer);
#endif
        break;
    }
    case VkComputePrivate::record::TYPE_memory_barrers:
    {
        vkCmdPipelineBarrier(command_buffer, r.memory_barrers.src_stage, r.memory_barrers.dst_stage, 0, r.memory_barrers.barrier_count, r.memory_barrers.barriers, 0, 0, 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_buffer_barrers:
    {
        vkCmdPipelineBarrier(command_buffer, r.buffer_barrers.src_stage, r.buffer_barrers.dst_stage, 0, 0, 0, r.buffer_barrers.barrier_count, r.buffer_barrers.barriers, 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_image_barrers:
    {
        vkCmdPipelineBarrier(command_buffer, r.image_barrers.src_stage, r.image_barrers.dst_stage, 0, 0, 0, 0, 0, r.image_barrers.barrier_count, r.image_barrers.barriers);
        break;
    }
#if NCNN_BENCHMARK
    case VkComputePrivate::record::TYPE_write_timestamp:
    {
        if (query_pool)
        {
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, r.write_timestamp.query);
        }
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

    if (d->executing || d->execution_error)
        return d->execution_error ? d->execution_error : -1;
    for (size_t i = 0; i < d->upload_staging_buffers.size(); i++)
    {
        const VkMat& staging = d->upload_staging_buffers[i];
        int ret = staging.allocator->flush(staging.data);
        if (ret != 0)
            return ret;
    }
    if (d->weight_deferred)
    {
        int ret = d->submit_managed();
        if (ret != 0)
            return ret;
    }
    else
    {
        if (!vkdev->info.support_VK_KHR_push_descriptor())
        {
            int ret = d->begin_command_buffer();
            if (ret != 0)
                return ret;

#if NCNN_BENCHMARK
            if (d->query_pool)
            {
                vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
            }
#endif // NCNN_BENCHMARK

            for (size_t i = 0; i < d->deferred_records.size(); i++)
            {
                ret = d->record_deferred(d->compute_command_buffer, d->deferred_records[i]);
                if (ret != 0)
                    return ret;
            }
        }

        int ret = d->submit_immediate();
        if (ret != 0)
            return ret;
    }

    // handle deferred post records
    for (size_t i = 0; i < d->deferred_records.size(); i++)
    {
        const VkComputePrivate::record& r = d->deferred_records[i];

        switch (r.type)
        {
        case VkComputePrivate::record::TYPE_post_download:
        {
            const VkMat& src = d->download_post_buffers[r.post_download.download_post_buffer_mat_offset];
            Mat& dst = d->download_post_mats_fp16[r.post_download.download_post_mat_fp16_offset];

            // NCNN_LOGE("post_download  %p +%d ~%d  -> %p", src.buffer(), src.buffer_offset(), src.buffer_capacity(), dst.data);

            if (src.allocator->invalidate(src.data))
                return -1;
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
#endif // NCNN_BENCHMARK
    d->clear_records();

    d->pending_dispatch_total = 0;

    return 0;
}

int VkCompute::reset()
{
    if (d->executing || d->execution_error == VK_ERROR_DEVICE_LOST)
        return -1;
#if NCNN_BENCHMARK
    d->query_results_ready = false;
#endif // NCNN_BENCHMARK
    // Discard command references before releasing their source and exchange views.
    {
        VkResult ret = vkResetCommandBuffer(d->compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            return -1;
        }
    }

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
        for (size_t i = 0; i < d->descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), d->descriptor_pools[i], 1, &d->descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), d->descriptor_pools[i], 0);
        }
        d->descriptor_pools.clear();
        d->descriptorsets.clear();
    }

    d->clear_execution();
    d->clear_records();
    // A failed execution may discard its allocation; reset always discards contents.
    if (d->weight_cache && d->execution_error)
    {
        d->weight_cache->clear();
    }
    d->weight_deferred = false;
    d->execution_error = 0;

    d->pending_dispatch_total = 0;

    // reset command buffer and fence

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
        {
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
        }
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkCompute::clear_weight_cache()
{
    if ((d->executing || d->execution_error == VK_ERROR_DEVICE_LOST) || !d->deferred_records.empty() || d->pending_dispatch_total)
    {
        NCNN_LOGE("cannot clear a weight cache while recording or executing");
        return -1;
    }
    delete d->weight_cache;
    d->weight_cache = 0;
    d->weight_deferred = false;
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
        {
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
        }
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
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

        if (!d->deferred())
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
            d->deferred_records.push_back(r);
        }

        // mark image shader-readonly-optimal @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        binding.data->image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

int VkComputePrivate::resolve_weight(const VkMat& weight, VkDescriptorBufferInfo& descriptor) const
{
    const weight_bindings& batch = current_bindings;
    for (size_t i = 0; i < batch.weights.size(); i++)
    {
        if (batch.weights[i].data != weight.data)
            continue;
        descriptor.buffer = batch.targets[i].buffer();
        descriptor.offset = batch.targets[i].buffer_offset() + weight.offset;
        descriptor.range = weight.buffer_capacity();
        return 0;
    }
    return -1;
}

class VkTransferPrivate
{
public:
    VkTransferPrivate(const VulkanDevice* _vkdev);
    ~VkTransferPrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();
    int finish_upload(int error = 0);
    int submit_and_wait();
    int reset();
    int upload_buffers();
    void record_upload(const Mat& src, const VkMat& dst, const Option& opt);
    void record_copy(const VkBufferMemory& src, const VkBufferMemory& dst, size_t size);
    void record_host_upload(const VkBufferMemory& dst);

    const VulkanDevice* vkdev;

    uint64_t pending_upload_total;
    int upload_error;
    bool compute_pending;
    bool upload_pending;

    VkCommandPool compute_command_pool;
    VkCommandPool transfer_command_pool;

    VkCommandBuffer upload_command_buffer;
    VkCommandBuffer compute_command_buffer;

    VkSemaphore upload_compute_semaphore;

    VkFence upload_command_fence;
    VkFence compute_command_fence;

    struct upload_record
    {
        Mat source;
        VkMat destination;
        size_t host_memory_limit;
        VkAllocator* staging_allocator;
    };
    std::vector<upload_record> uploads;
    uint64_t queued_upload_total;
    VkMat pending_staging_mat;
    VkBufferMemory* pending_staging_buffer;
};

VkTransferPrivate::VkTransferPrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    pending_upload_total = 0;
    queued_upload_total = 0;
    pending_staging_buffer = 0;
    upload_error = 0;
    compute_pending = false;
    upload_pending = false;

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
    do
    {
        finish_upload(upload_error);
        if (compute_pending || upload_pending)
            wait_for_pending_commands();
    } while (compute_pending || upload_pending);
    destroy_weight_buffer(vkdev, pending_staging_buffer);

    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    if (compute_command_buffer)
        vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);

        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);

        if (upload_command_buffer)
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
        commandBufferAllocateInfo.commandBufferCount = 1;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

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
            commandBufferAllocateInfo.commandBufferCount = 1;
            commandBufferAllocateInfo.commandPool = transfer_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

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

    return begin_command_buffer();
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
        NCNN_LOGE("weight exceeds maximum flattened dimension");
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

    // Preserve the allocating overload's snapshot-at-recording behavior.
    if (dst.data->storage_type == 2)
        d->record_upload(src_flattened, dst, opt);
    else
    {
        Mat snapshot = src_flattened.clone();
        if (snapshot.empty())
        {
            d->upload_error = -100;
            return;
        }
        d->record_upload(snapshot, dst, opt);
    }
}

void VkTransferPrivate::record_upload(const Mat& src, const VkMat& dst, const Option& opt)
{
    if (upload_error || src.empty())
        return;
    if (dst.empty() || !dst.allocator || dst.allocator->vkdev != vkdev || src.n != dst.n
            || dst.offset > dst.data->capacity || dst.buffer_capacity() > dst.data->capacity - dst.offset)
    {
        upload_error = -1;
        return;
    }
    for (int b = 0; b < src.n; b++)
    {
        const size_t offset = dst.batch(b).offset;
        const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
        if (offset > end || src.batch(b).total() * src.elemsize > end - offset)
        {
            upload_error = -1;
            return;
        }
    }
    if (dst.data->storage_type == 2)
    {
        if (!(dst.data->access_flags & VK_ACCESS_HOST_WRITE_BIT) || !dst.data->source_ptr
                || dst.offset > dst.data->source_size || dst.buffer_capacity() > dst.data->source_size - dst.offset)
        {
            NCNN_LOGE("disk weight requires a writable source range during upload");
            upload_error = -1;
            return;
        }
        for (int b = 0; b < src.n; b++)
        {
            const Mat src_b = src.batch(b);
            const size_t offset = dst.batch(b).offset;
            const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            const size_t size = src_b.total() * src_b.elemsize;
            unsigned char* ptr = (unsigned char*)dst.data->source_ptr + offset;
            memcpy(ptr, src_b.data, size);
            memset(ptr + size, 0, end - offset - size);
        }
        return;
    }
    if (dst.buffer_capacity() > (uint64_t)-1 - queued_upload_total)
    {
        upload_error = -1;
        return;
    }
    VkTransferPrivate::upload_record r;
    r.source = src;
    r.destination = dst;
    r.host_memory_limit = opt.weight_host_memory_limit;
    r.staging_allocator = opt.staging_vkallocator;
    uploads.push_back(r);
    queued_upload_total += dst.buffer_capacity();
}

void VkTransferPrivate::record_copy(const VkBufferMemory& src, const VkBufferMemory& dst, size_t size)
{
    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue())
    {
        command_buffer = compute_command_buffer;
    }
    else
    {
        command_buffer = upload_command_buffer;
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
        barrier.buffer = src.buffer;
        barrier.offset = src.offset;
        barrier.size = size;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.size = size;
        region.srcOffset = src.offset;
        region.dstOffset = dst.offset;

        vkCmdCopyBuffer(command_buffer, src.buffer, dst.buffer, 1, &region);
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
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

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
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
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
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
}

void VkTransferPrivate::record_host_upload(const VkBufferMemory& dst)
{
    // barrier device host-write @ null to shader-read @ compute
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst.buffer;
        barrier.offset = dst.offset;
        barrier.size = dst.capacity;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }
}

int VkTransferPrivate::upload_buffers()
{
    if (upload_error)
        return upload_error;
    size_t capacity = 0;
    size_t host_memory_limit = 0;
    const size_t target = 8 * 1024 * 1024;
    for (size_t i = 0; i < uploads.size(); i++)
    {
        const upload_record& r = uploads[i];
        if (!r.destination.mapped_ptr())
        {
            capacity += std::min(alignSize(r.destination.buffer_capacity(), 4), target - capacity);
            if (r.host_memory_limit)
                host_memory_limit = host_memory_limit ? std::min(host_memory_limit, r.host_memory_limit) : r.host_memory_limit;
        }
    }
    VkBufferMemory* staging = 0;
    VkMat staging_mat;
    VkAllocator* staging_allocator = 0;
    int ret = 0;
    if (capacity)
        ret = fit_weight_capacity(vkdev, host_memory_limit, true, true, capacity);
    size_t used = 0;
    for (size_t i = 0; ret == 0 && i < uploads.size(); i++)
    {
        const upload_record& r = uploads[i];
        const VkMat& dst = r.destination;
        if (!dst.mapped_ptr() && (!staging || staging_allocator != r.staging_allocator))
        {
            if (used)
            {
                ret = staging_allocator ? staging_allocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
                if (ret == 0)
                    ret = submit_and_wait();
                if (ret == 0)
                    ret = reset();
                used = 0;
            }
            if (ret != 0)
                break;
            if (staging && !staging_allocator)
                destroy_weight_buffer(vkdev, staging);
            staging_mat.release();
            staging = 0;
            staging_allocator = r.staging_allocator;
            if (staging_allocator)
            {
                if (staging_allocator->vkdev != vkdev)
                {
                    ret = -1;
                    break;
                }
                staging_mat.create((int)capacity, (size_t)1u, staging_allocator);
                staging = staging_mat.data;
            }
            else
            {
                size_t allocation_size;
                const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
                staging = create_weight_buffer(vkdev, capacity, 1, unified, host_memory_limit, allocation_size, true);
            }
            if (!staging || !staging->mapped_ptr)
            {
                ret = -100;
                break;
            }
            if (host_memory_limit)
            {
                VkMemoryRequirements requirements;
                vkGetBufferMemoryRequirements(vkdev->vkdevice(), staging->buffer, &requirements);
                if (requirements.size > host_memory_limit)
                {
                    ret = -1;
                    break;
                }
            }
        }
        for (int b = 0; ret == 0 && b < r.source.n; b++)
        {
            const Mat src_b = r.source.batch(b);
            const VkMat dst_b = dst.batch(b);
            const size_t source_size = src_b.total() * src_b.elemsize;
            const size_t end = b + 1 == r.source.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            const size_t size = end - dst_b.offset;
            if (dst.mapped_ptr())
            {
                unsigned char* ptr = (unsigned char*)dst_b.mapped_ptr();
                memcpy(ptr, src_b.data, source_size);
                memset(ptr + source_size, 0, size - source_size);
                continue;
            }
            for (size_t offset = 0; ret == 0 && offset < size;)
            {
                if (used == capacity)
                {
                    ret = staging_allocator ? staging_allocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
                    if (ret == 0)
                        ret = submit_and_wait();
                    if (ret == 0)
                        ret = reset();
                    used = 0;
                    if (ret != 0)
                        break;
                }
                const size_t n = std::min(capacity - used, size - offset);
                const size_t copy_size = alignSize(n, 4);
                if (offset + copy_size > dst.data->capacity - dst_b.offset)
                {
                    ret = -1;
                    break;
                }
                const size_t source_bytes = offset < source_size ? std::min(n, source_size - offset) : 0;
                unsigned char* ptr = (unsigned char*)staging->mapped_ptr + staging->offset + used;
                if (source_bytes)
                    memcpy(ptr, (const unsigned char*)src_b.data + offset, source_bytes);
                memset(ptr + source_bytes, 0, copy_size - source_bytes);
                VkBufferMemory src_buffer = *staging;
                src_buffer.offset += used;
                VkBufferMemory dst_buffer = *dst.data;
                dst_buffer.offset = dst_b.buffer_offset() + offset;
                record_copy(src_buffer, dst_buffer, copy_size);
                pending_upload_total += copy_size;
                used += copy_size;
                offset += n;
            }
        }
        if (ret == 0 && dst.mapped_ptr())
        {
            ret = dst.allocator->flush(dst.data);
            if (ret == 0 && dst.data->storage_type != 1)
            {
                record_host_upload(*dst.data);
                pending_upload_total += dst.buffer_capacity();
            }
        }
        if (ret == 0)
        {
            dst.data->access_flags = dst.data->storage_type == 1 ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            dst.data->stage_flags = dst.data->storage_type == 1 ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
    }
    if (ret == 0 && used)
        ret = staging_allocator ? staging_allocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
    if (ret == 0)
        ret = submit_and_wait();
    // A failed fence wait does not prove completion. Keep staging alive for
    // finish_upload() to drain again during command destruction.
    if (compute_pending || upload_pending)
    {
        if (staging_allocator)
            pending_staging_mat = staging_mat;
        else
            pending_staging_buffer = staging;
    }
    else if (staging && !staging_allocator)
    {
        destroy_weight_buffer(vkdev, staging);
    }
    if (ret != 0)
        upload_error = ret;
    if (ret == 0)
    {
        uploads.clear();
        queued_upload_total = 0;
    }
    return ret;
}

int VkTransferPrivate::finish_upload(int error)
{
    VkFence fences[2];
    uint32_t count = 0;
    if (upload_pending)
        fences[count++] = upload_command_fence;
    if (compute_pending)
        fences[count++] = compute_command_fence;
    if (count)
    {
        const VkResult ret = vkWaitForFences(vkdev->vkdevice(), count, fences, VK_TRUE, (uint64_t)-1);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkWaitForFences failed %d", ret);
            error = -1;
        }
        if (ret == VK_SUCCESS || ret == VK_ERROR_DEVICE_LOST)
        {
            compute_pending = false;
            upload_pending = false;
        }
    }
    if (error)
        upload_error = error;
    else
        pending_upload_total = 0;
    return error;
}

int VkTransfer::submit_and_wait()
{
    return d->upload_buffers();
}

int VkTransfer::reset()
{
    const int ret = d->reset();
    if (ret == 0)
    {
        d->uploads.clear();
        d->queued_upload_total = 0;
    }
    return ret;
}

int VkTransferPrivate::submit_and_wait()
{
    if (upload_error)
        return upload_error;
    if (pending_upload_total == 0)
        return 0;

    //     NCNN_LOGE("submit_and_wait");

    // end command buffer
    {
        int ret = end_command_buffer();
        if (ret != 0)
            return finish_upload(ret);
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
    if (compute_queue == 0)
    {
        NCNN_LOGE("out of compute queue");
        return finish_upload(-1);
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
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            compute_pending = true;
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index());
        if (transfer_queue == 0)
        {
            NCNN_LOGE("out of transfer queue");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
            return finish_upload(-1);
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
            submitInfo.pCommandBuffers = &upload_command_buffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &upload_compute_semaphore;

            VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            upload_pending = true;
        }
        {
            VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // FIXME

            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &upload_compute_semaphore;
            submitInfo.pWaitDstStageMask = &wait_dst_stage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            compute_pending = true;
        }

        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);

    return finish_upload();
}

int VkTransferPrivate::reset()
{
    if (upload_error)
        return upload_error;

    pending_upload_total = 0;

    // reset command buffer and fence
    {
        VkResult ret = vkResetCommandBuffer(compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            upload_error = -1;
            return upload_error;
        }
    }
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            upload_error = -1;
            return upload_error;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkResult ret = vkResetCommandBuffer(upload_command_buffer, 0);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
                upload_error = -1;
                return upload_error;
            }
        }
        {
            VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetFences failed %d", ret);
                upload_error = -1;
                return upload_error;
            }
        }
    }

    upload_error = begin_command_buffer();

    return upload_error;
}

uint64_t VkTransfer::pending_upload_total() const
{
    return d->pending_upload_total + d->queued_upload_total;
}

} // namespace ncnn

#endif // NCNN_VULKAN
