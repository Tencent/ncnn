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
int fit_weight_capacity(const VulkanDevice* vkdev, size_t memory_limit, bool automatic, bool concurrent, size_t& capacity);
VkBufferMemory* create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t memory_limit, size_t& allocation_size, bool allocate);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

int get_weight_source_generation(const VkAllocator* allocator, uint64_t& source_generation, uint64_t* source_lifetime = 0);
void release_weight_source(VkMat& source, uint64_t lifetime);

struct buffer_binding;

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

struct buffer_binding
{
    uintptr_t source_identity;
    size_t view_offset;
    VkDescriptorBufferInfo descriptor;
};

// Metadata and backing allocation ownership stay below the VkMat resource interface.
class VkWeightCacheAllocator : public VkAllocator
{
public:
    struct Region : VkBufferMemory
    {
        Region()
            : allocation(0), source_lifetime(0)
        {
        }
        VkMat backing;
        VkBufferMemory* allocation;
        uint64_t source_lifetime;
    };

    explicit VkWeightCacheAllocator(const VulkanDevice* vkdev)
        : VkAllocator(vkdev)
    {
    }
    virtual VkBufferMemory* fastMalloc(size_t)
    {
        return new Region;
    }
    virtual void fastFree(VkBufferMemory* memory)
    {
        Region* region = static_cast<Region*>(memory);
        if (region->source_lifetime)
            release_weight_source(region->backing, region->source_lifetime);
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
        // Opaque byte capacity, not a tensor: never apply dtype/shape operations.
        VkMat owner(1, region, allocation->capacity, 1, this);
        owner.refcount = &region->refcount;
        return owner;
    }

    VkMat retain_source(const VkMat& source, uint64_t lifetime)
    {
        Region* region = new Region;
        static_cast<VkBufferMemory&>(*region) = *source.data;
        region->source_lifetime = lifetime;
        region->backing = source;
        region->refcount = 1;
        VkMat result(source.w, source.h, source.d, source.c, region, source.elemsize, source.elempack, this);
        result.dims = source.dims;
        result.cstep = source.cstep;
#if NCNN_BATCH
        result.n = source.n;
        result.nstep = source.nstep;
        result.offset = 0;
#endif
        result.refcount = &region->refcount;
        return result;
    }

    uintptr_t identity(const VkMat& source) const
    {
        if (source.allocator == this)
            return (uintptr_t) static_cast<const Region*>(source.data)->backing.data;
        return (uintptr_t)source.data;
    }
    const VkAllocator* source_allocator(const VkMat& source) const
    {
        return source.allocator == this ? static_cast<const Region*>(source.data)->backing.allocator : source.allocator;
    }
};

class VkWeightCache
{
public:
    struct weight_entry
    {
        VkMat target;
        // Weak identity only: never dereferenced; the generation guards reuse.
        uintptr_t source_identity;
        uint64_t source_generation;
        size_t last_batch;
        bool valid;
    };

    struct buffer_range
    {
        size_t offset;
        size_t size;
    };

    struct weight_batch
    {
        size_t begin;
        size_t end;
        std::vector<VkMat> weights;
        std::vector<VkMat> targets;
        std::vector<bool> missing;
        std::vector<buffer_range> releases;
        std::vector<buffer_range> acquires;
        size_t wait_batch;
    };

    struct device_plan
    {
        std::vector<weight_batch> batches;
        std::vector<buffer_range> initial_releases;
        std::vector<weight_entry> device_entries;
        size_t device_head;
    };

    VkWeightCache(const VulkanDevice* vkdev, size_t device_cache_size, size_t host_cache_size, size_t device_memory_limit, size_t host_memory_limit);
    ~VkWeightCache();
    void clear();
    void clear_sources();
    int record_source(const VkMat& mat);
    int validate_sources() const;
    int find_source_entry(const VkMat& source) const;
    int find_device_entry(const VkMat& source) const;
    int find_host_entry(const VkMat& source) const;
    int initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size);
    int binding_source(const buffer_binding& binding, VkMat& source) const;
    size_t aligned_size(const VkMat& source) const;
    int plan_batches(const std::vector<VkMat>& dependencies, const std::vector<size_t>& dependency_offsets, device_plan& plan) const;
    void publish_device_plan(device_plan& plan);
    int select_host_range(size_t size, size_t& offset) const;
    void reserve_host_range(size_t offset, size_t size);
    int publish_host_entry(const VkMat& source, size_t offset, size_t size);
    VkMat host_range(size_t offset, size_t size) const;
    const VulkanDevice* device() const
    {
        return vkdev;
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
    bool valid() const
    {
        return error == 0;
    }
    void invalidate(int result)
    {
        if (!error) error = result;
    }
    const VkMat& source(int index) const
    {
        return source_entries[index];
    }
    const weight_entry& host_entry(int index) const
    {
        return host_entries[index];
    }

private:
    VkMat cache_view(bool host, size_t offset, size_t size) const;
    int allocate();
    int find_planned_device_entry(const std::vector<weight_entry>& planned_device_entries, const VkMat& source) const;
    bool is_pinned(uintptr_t identity, const std::vector<VkMat>& pinned) const;
    int layout_device_batch(weight_batch& batch, std::vector<weight_entry>& planned_device_entries, size_t& head, std::vector<weight_entry>& evicted) const;
    void commit_device_batch(weight_batch& batch, std::vector<weight_entry>& planned_device_entries, const std::vector<weight_entry>& evicted, std::vector<weight_batch>& batches,
                             std::vector<buffer_range>& initial_releases) const;
    int plan_device_cache_range(const VkMat& source, const std::vector<VkMat>& pinned, std::vector<weight_entry>& planned_device_entries, size_t& head,
                                std::vector<weight_entry>& evicted) const;
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
    size_t host_head;
    size_t device_head;
    std::vector<VkMat> source_entries;
    std::vector<uint64_t> source_generations;
    std::vector<weight_entry> device_entries;
    std::vector<weight_entry> host_entries;
    bool unified;
    bool initialized;
    int error;
};

class VkWeightCacheTransfer
{
public:
    explicit VkWeightCacheTransfer(VkWeightCache& cache);
    ~VkWeightCacheTransfer();
    int begin_execution(const std::vector<VkMat>& dependencies, const std::vector<size_t>& dependency_offsets);
    int submit_prefix(VkCommandBuffer command_buffer, VkFence fence);
    int prepare(size_t batch_index);
    int begin_compute(VkCommandBuffer* command_buffer);
    int submit_compute(int task);
    int finish(int result = 0);
    void drain();
    bool pending() const
    {
        return executing;
    }
    int status() const
    {
        return error;
    }
    const std::vector<VkWeightCache::weight_batch>& batches() const
    {
        return plan.batches;
    }
    int resolve_binding(const buffer_binding& binding, VkDescriptorBufferInfo& descriptor) const;

private:
    typedef VkWeightCache::buffer_range buffer_range;
    typedef VkWeightCache::weight_batch weight_batch;
    int release_ranges(const std::vector<buffer_range>& ranges);
    int submit_command(bool transfer, const VkSubmitInfo& info, VkFence fence);
    enum command_type
    {
        UPLOAD_COMMAND,
        COMPUTE_COMMAND,
        RELEASE_COMMAND
    };
    int allocate_command_buffer(command_type type, size_t task = 0);
    int begin_command_buffer(command_type type, size_t task = 0);
    int create_commands();
    void destroy_commands();
    int upload_source_to_device_cache(const VkMat& source, const VkMat& target);
    int record_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last);
    int submit_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last);
    int wait_fence(VkFence fence);
    int wait_compute(uint64_t serial);
    int create_upload();
    int allocate_host_range(size_t size, size_t& offset);
    int fill_host_cache(const VkMat& source, VkMat& target);
    int flush_host(size_t offset, size_t size);
    void record_upload_release_to_compute(int task, const buffer_range& range);
    void record_compute_acquire_from_upload(int task, const buffer_range& range);
    void record_compute_release_to_upload(int task, const buffer_range& range);
    void record_initial_release_to_upload(const buffer_range& range);
    void record_upload_acquire_from_compute(int task, const buffer_range& range);

    VkWeightCache& cache;
    const VulkanDevice* vkdev;
    VkWeightCache::device_plan plan;
    uint64_t execution_serial_base;
    std::vector<buffer_range> pending_acquires;
    std::vector<int> pending_uploads;
    const weight_batch* current_batch;
    VkCommandPool transfer_command_pool;
    VkFence prefix_fence;
    VkCommandBuffer release_command_buffer;
    VkFence release_fence;
    bool release_pending;
    uint32_t transfer_queue_family_index;
    VkSemaphore release_semaphore;
    bool release_signal_pending;
    std::vector<VkCommandBuffer> upload_command_buffers;
    std::vector<VkFence> upload_fences;
    std::vector<VkSemaphore> upload_compute_semaphores;
    std::vector<bool> upload_pending;
    std::vector<VkMat> upload_sources;
    std::vector<VkMat> upload_targets;
    std::vector<VkCommandBuffer> compute_command_buffers;
    std::vector<VkFence> compute_fences;
    std::vector<VkSemaphore> compute_upload_semaphores;
    std::vector<bool> compute_release_pending;
    std::vector<bool> compute_pending;
    std::vector<uint64_t> compute_serials;
    uint64_t compute_serial;
    uint64_t completed_compute_serial;
    bool commands_initialized;
    bool executing;

    VkCommandPool compute_command_pool;

    std::vector<VkSemaphore> queue_signals[2];
    int error;
#if NCNN_BENCHMARK
    uint64_t bytes_read;
    uint64_t bytes_uploaded;
    double start_time;
#endif
};

int VkWeightCacheTransfer::allocate_command_buffer(command_type type, size_t task)
{
    const VkDevice device = vkdev->vkdevice();
    const VkCommandPool pool = type == UPLOAD_COMMAND ? transfer_command_pool : compute_command_pool;
    VkCommandBuffer* command_buffer = type == UPLOAD_COMMAND ? &upload_command_buffers[task] : type == COMPUTE_COMMAND ? &compute_command_buffers[task] : &release_command_buffer;
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

int VkWeightCacheTransfer::begin_command_buffer(command_type type, size_t task)
{
    const VkCommandBuffer command_buffer = type == UPLOAD_COMMAND ? upload_command_buffers[task] : type == COMPUTE_COMMAND ? compute_command_buffers[task] : release_command_buffer;
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

VkWeightCache::VkWeightCache(const VulkanDevice* _vkdev, size_t device_cache_size, size_t host_cache_size, size_t device_memory_limit, size_t host_memory_limit)
    : vkdev(_vkdev), view_allocator(_vkdev)
{
    weight_device_cache_size = device_cache_size;
    weight_host_cache_size = host_cache_size;
    weight_device_memory_limit = device_memory_limit;
    weight_host_memory_limit = host_memory_limit;
    automatic_device_size = weight_device_cache_size == 0;
    automatic_host_size = weight_host_cache_size == 0;
    host_buffer.release();
    device_buffer.release();
    host_size = weight_host_cache_size;
    device_size = weight_device_cache_size;
    alignment = weight_buffer_alignment(vkdev);
    host_head = 0;
    device_head = 0;
    unified = false;
    initialized = false;
    error = 0;
}

VkWeightCache::~VkWeightCache()
{
    clear();
}

void VkWeightCache::publish_device_plan(device_plan& plan)
{
    std::swap(device_entries, plan.device_entries);
    device_head = plan.device_head;
}

int VkWeightCache::select_host_range(size_t size, size_t& offset) const
{
    if (size > host_size)
        return -1;
    offset = size > host_size - host_head ? 0 : host_head;
    return 0;
}

void VkWeightCache::reserve_host_range(size_t offset, size_t size)
{
    for (size_t i = 0; i < host_entries.size(); i++)
    {
        weight_entry& entry = host_entries[i];
        if (entry.source_identity && offset < entry.target.buffer_offset() + entry.target.buffer_capacity() && entry.target.buffer_offset() < offset + size)
            entry.source_identity = 0;
    }
    host_head = offset + size;
}

VkMat VkWeightCache::cache_view(bool host, size_t offset, size_t size) const
{
    const VkMat& owner = host ? host_buffer : device_buffer;
    const VkBufferMemory* backing = owner.data;
    VkBufferMemory* memory = view_allocator.fastMalloc(size);
    *memory = *backing;
    static_cast<VkWeightCacheAllocator::Region*>(memory)->backing = owner;
    memory->offset = backing->offset + offset;
    memory->capacity = size;
    memory->source_ptr = 0;
    memory->source_size = 0;
    memory->refcount = 1;
    // A cache region is an opaque byte span, not the source tensor shape.
    VkMat view(1, memory, size, 1, &view_allocator);
    view.refcount = &memory->refcount;
    return view;
}

VkMat VkWeightCache::host_range(size_t offset, size_t size) const
{
    return cache_view(true, offset, size);
}

int VkWeightCache::publish_host_entry(const VkMat& source, size_t offset, size_t size)
{
    const int recorded = find_source_entry(source);
    if (recorded < 0)
        return -1;
    int slot = -1;
    for (size_t i = 0; i < host_entries.size(); i++)
        if (!host_entries[i].source_identity)
            slot = (int)i;
    weight_entry entry;
    entry.target = cache_view(true, offset, size);
    entry.source_identity = view_allocator.identity(source);
    entry.source_generation = source_generations[recorded];
    entry.last_batch = 0;
    entry.valid = true;
    if (slot < 0)
        host_entries.push_back(entry);
    else
        host_entries[slot] = entry;
    return 0;
}

void VkWeightCache::clear()
{
    device_entries.clear();
    host_entries.clear();
    host_buffer.release();
    device_buffer.release();
    host_head = 0;
    device_head = 0;
    initialized = false;
    error = 0;
}

void VkWeightCache::clear_sources()
{
    source_entries.clear();
    source_generations.clear();
    // Custom allocators have no generation registry, so do not reuse their identity.
    for (size_t i = 0; i < device_entries.size(); i++)
        if (!device_entries[i].source_generation)
            device_entries[i].source_identity = 0;
    for (size_t i = 0; i < host_entries.size(); i++)
        if (!host_entries[i].source_generation)
            host_entries[i].source_identity = 0;
}

int VkWeightCache::find_source_entry(const VkMat& source) const
{
    for (size_t i = 0; i < source_entries.size(); i++)
        if (view_allocator.identity(source_entries[i]) == view_allocator.identity(source))
            return (int)i;
    return -1;
}

int VkWeightCache::record_source(const VkMat& mat)
{
    const int recorded = find_source_entry(mat);
    if (recorded >= 0)
        return view_allocator.source_allocator(source_entries[recorded]) == mat.allocator ? 0 : -1;
    uint64_t generation = 0;
    uint64_t lifetime = 0;
    if (get_weight_source_generation(mat.allocator, generation, &lifetime) < 0)
    {
        NCNN_LOGE("weight source must be finalized before recording");
        return -1;
    }
    const int device_entry = find_device_entry(mat);
    if (device_entry >= 0 && device_entries[device_entry].source_generation != generation)
        device_entries[device_entry].source_identity = 0;
    const int host_entry = find_host_entry(mat);
    if (host_entry >= 0 && host_entries[host_entry].source_generation != generation)
        host_entries[host_entry].source_identity = 0;
    source_entries.push_back(generation ? view_allocator.retain_source(mat, lifetime) : mat);
    // Bindings retain their view offset; uploads cover the complete allocation.
#if NCNN_BATCH
    source_entries.back().offset = 0;
#endif
    source_generations.push_back(generation);
    return 0;
}

int VkWeightCache::validate_sources() const
{
    // Check allocator versions before dereferencing any retained source metadata.
    for (size_t i = 0; i < source_entries.size(); i++)
    {
        if (!source_generations[i])
            continue;
        uint64_t generation;
        if (get_weight_source_generation(view_allocator.source_allocator(source_entries[i]), generation) < 0 || generation != source_generations[i])
        {
            NCNN_LOGE("weight source was cleared or changed while recording");
            return -1;
        }
    }
    return 0;
}

int VkWeightCache::initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size)
{
    if (error != 0)
        return error;
    if (max_source_size > (size_t)-1 / 2 || max_file_size > (size_t)-1 / 2)
    {
        NCNN_LOGE("weight cache size overflow");
        return -1;
    }

    size_t device_memory_limit = weight_device_memory_limit;
    size_t host_memory_limit = weight_host_memory_limit;
    unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    if (unified)
    {
        if (device_memory_limit && host_memory_limit)
            device_memory_limit = std::min(device_memory_limit, host_memory_limit);
        else
            device_memory_limit = std::max(device_memory_limit, host_memory_limit);
        host_memory_limit = 0;
    }
    size_t new_device_size = device_size;
    if (!new_device_size)
        new_device_size = std::max(max_source_size * 2, minimum_device_size);
    else if (automatic_device_size)
        new_device_size = std::max(new_device_size, minimum_device_size);
    size_t new_host_size = host_size ? host_size : weight_host_cache_size;
    if (unified || !max_file_size)
        new_host_size = 0;
    else if (max_file_size && !new_host_size)
        new_host_size = max_file_size * 2;
    if (new_device_size > (size_t)-1 - alignment || new_host_size > (size_t)-1 - alignment)
    {
        NCNN_LOGE("weight cache alignment overflow");
        return -1;
    }
    new_device_size = alignSize(new_device_size, alignment);
    new_host_size = alignSize(new_host_size, alignment);
    const bool validate_capacity = !initialized || new_device_size != device_size || new_host_size != host_size;
    if (validate_capacity)
    {
        int ret = fit_weight_capacity(vkdev, device_memory_limit, automatic_device_size, false, new_device_size);
        if (ret != 0)
            return ret;
        ret = fit_weight_capacity(vkdev, host_memory_limit, automatic_host_size, true, new_host_size);
        if (ret != 0)
            return ret;
    }
    if (new_device_size < minimum_device_size || (max_file_size && !unified && new_host_size < alignment))
    {
        NCNN_LOGE("weight cache exceeds command capacity or memory limit");
        return -1;
    }
    const bool resize = new_device_size != device_size || new_host_size != host_size;
    if (initialized && resize)
    {
        // release the old buffers before growing the cache
        clear();
    }
    weight_device_memory_limit = device_memory_limit;
    weight_host_memory_limit = host_memory_limit;
    device_size = new_device_size;
    host_size = new_host_size;
    if (!initialized)
    {
        error = allocate();
        initialized = true;
    }
    if (error)
        return error;
    return error;
}

int VkWeightCache::binding_source(const buffer_binding& binding, VkMat& source) const
{
    source.release();
    if (!binding.source_identity)
        return 0;
    for (size_t i = 0; i < source_entries.size(); i++)
        if (view_allocator.identity(source_entries[i]) == binding.source_identity)
            source = source_entries[i];
    if (source.empty())
        return -1;
    const VkBufferMemory* src = source.data;
    if (!is_deferred_buffer(src) || binding.view_offset > src->source_size
            || binding.descriptor.range > src->source_size - binding.view_offset
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

size_t VkWeightCache::aligned_size(const VkMat& source) const
{
    return alignSize(source.data->source_size, alignment);
}

int VkWeightCache::layout_device_batch(weight_batch& batch, std::vector<weight_entry>& planned_device_entries, size_t& head, std::vector<weight_entry>& evicted) const
{
    for (size_t j = 0; j < batch.weights.size(); j++)
    {
        if (find_planned_device_entry(planned_device_entries, batch.weights[j]) >= 0)
            continue;
        if (plan_device_cache_range(batch.weights[j], batch.weights, planned_device_entries, head, evicted) >= 0)
            continue;
        // Compact only when pinned ranges prevent a contiguous placement.
        for (size_t k = 0; k < planned_device_entries.size(); k++)
            if (planned_device_entries[k].valid)
                evicted.push_back(planned_device_entries[k]);
        planned_device_entries.clear();
        head = 0;
        for (size_t k = 0; k < batch.weights.size(); k++)
            if (plan_device_cache_range(batch.weights[k], batch.weights, planned_device_entries, head, evicted) < 0)
                return -1;
        break;
    }
    batch.targets.clear();
    batch.missing.clear();
    for (size_t j = 0; j < batch.weights.size(); j++)
    {
        const weight_entry& entry = planned_device_entries[find_planned_device_entry(planned_device_entries, batch.weights[j])];
        batch.targets.push_back(entry.target);
        batch.missing.push_back(!entry.valid);
    }
    return 0;
}

void VkWeightCache::commit_device_batch(weight_batch& batch, std::vector<weight_entry>& planned_device_entries, const std::vector<weight_entry>& evicted,
                                        std::vector<weight_batch>& batches, std::vector<buffer_range>& initial_releases) const
{
    batch.wait_batch = 0;
    for (size_t j = 0; j < evicted.size(); j++)
    {
        const weight_entry& entry = evicted[j];
        buffer_range range;
        range.offset = entry.target.buffer_offset();
        range.size = entry.target.buffer_capacity();
        if (entry.last_batch)
            batches[entry.last_batch - 1].releases.push_back(range);
        else
            initial_releases.push_back(range);
        batch.acquires.push_back(range);
        batch.wait_batch = std::max(batch.wait_batch, entry.last_batch);
    }
    for (size_t j = 0; j < batch.weights.size(); j++)
    {
        weight_entry& entry = planned_device_entries[find_planned_device_entry(planned_device_entries, batch.weights[j])];
        entry.valid = true;
        entry.last_batch = batches.size() + 1;
    }
    batches.push_back(batch);
}

int VkWeightCache::plan_batches(const std::vector<VkMat>& dependencies, const std::vector<size_t>& dependency_offsets, device_plan& plan) const
{
    plan.batches.clear();
    plan.initial_releases.clear();
    std::vector<weight_batch>& batches = plan.batches;
    std::vector<buffer_range>& initial_releases = plan.initial_releases;
    std::vector<weight_entry> working_device_entries = device_entries;
    for (size_t i = 0; i < working_device_entries.size(); i++)
        working_device_entries[i].last_batch = 0;
    size_t head = device_head;
    weight_batch batch;
    batch.begin = batch.end = batch.wait_batch = 0;
    std::vector<weight_entry> candidate_device_entries = working_device_entries;
    std::vector<weight_entry> planned_evicted;
    size_t planned_head = head;
    size_t batch_size = 0;
    const size_t record_count = dependency_offsets.size() - 1;
    for (size_t i = 0; i < record_count;)
    {
        weight_batch candidate = batch;
        size_t candidate_size = batch_size;
        for (size_t j = dependency_offsets[i]; j < dependency_offsets[i + 1]; j++)
        {
            const VkMat& source = dependencies[j];
            if (is_pinned(view_allocator.identity(source), candidate.weights))
                continue;
            const size_t size = aligned_size(source);
            if (size > (size_t)-1 - candidate_size)
                return -1;
            candidate_size += size;
            candidate.weights.push_back(source);
        }
        if (candidate.weights.size() == batch.weights.size())
        {
            batch.end = ++i;
            continue;
        }
        std::vector<weight_entry> trial_device_entries = working_device_entries;
        std::vector<weight_entry> evicted;
        size_t trial_head = head;
        size_t missing_size = 0;
        int ret = -1;
        if (candidate_size <= device_size)
        {
            ret = layout_device_batch(candidate, trial_device_entries, trial_head, evicted);
            if (ret != 0)
                return ret;
            for (size_t j = 0; j < candidate.weights.size(); j++)
            {
                if (!candidate.missing[j])
                    continue;
                missing_size += aligned_size(candidate.weights[j]);
            }
        }
        // Batch only for real device-cache capacity and overlap opportunities.
        if (!batch.weights.empty() && (candidate_size > device_size || missing_size > device_size / 2))
        {
            commit_device_batch(batch, candidate_device_entries, planned_evicted, batches, initial_releases);
            std::swap(working_device_entries, candidate_device_entries);
            head = planned_head;
            batch = weight_batch();
            batch.begin = batch.end = i;
            batch_size = 0;
            candidate_device_entries = working_device_entries;
            planned_evicted.clear();
            continue;
        }
        if (ret != 0)
        {
            NCNN_LOGE("weight bindings exceed cache capacity");
            return -1;
        }
        batch = candidate;
        batch.end = ++i;
        batch_size = candidate_size;
        std::swap(candidate_device_entries, trial_device_entries);
        std::swap(planned_evicted, evicted);
        planned_head = trial_head;
    }
    commit_device_batch(batch, candidate_device_entries, planned_evicted, batches, initial_releases);
    std::swap(plan.device_entries, candidate_device_entries);
    plan.device_head = planned_head;
    return 0;
}

int VkWeightCache::find_device_entry(const VkMat& source) const
{
    return find_planned_device_entry(device_entries, source);
}

int VkWeightCache::find_planned_device_entry(const std::vector<weight_entry>& planned_device_entries, const VkMat& source) const
{
    for (size_t i = 0; i < planned_device_entries.size(); i++)
        if (planned_device_entries[i].source_identity == view_allocator.identity(source))
            return (int)i;
    return -1;
}

int VkWeightCache::find_host_entry(const VkMat& source) const
{
    for (size_t i = 0; i < host_entries.size(); i++)
        if (host_entries[i].source_identity == view_allocator.identity(source))
            return (int)i;
    return -1;
}

int VkWeightCache::allocate()
{
    if (!device_size)
        return 0;

    size_t allocation_size;
    device_buffer = view_allocator.adopt(create_weight_buffer(vkdev, device_size, 0, unified, weight_device_memory_limit, allocation_size, true));
    if (device_buffer.empty())
        return -100;

    if (host_size)
    {
        host_buffer = view_allocator.adopt(create_weight_buffer(vkdev, host_size, 1, false, weight_host_memory_limit, allocation_size, true));
        if (host_buffer.empty())
            return -100;
    }

    return 0;
}

bool VkWeightCache::is_pinned(uintptr_t identity, const std::vector<VkMat>& pinned) const
{
    for (size_t i = 0; i < pinned.size(); i++)
        if (view_allocator.identity(pinned[i]) == identity)
            return true;
    return false;
}

int VkWeightCache::plan_device_cache_range(const VkMat& source, const std::vector<VkMat>& pinned, std::vector<VkWeightCache::weight_entry>& planned_device_entries, size_t& head,
        std::vector<VkWeightCache::weight_entry>& evicted) const
{
    const int recorded = find_source_entry(source);
    if (recorded < 0)
        return -1;
    const size_t size = aligned_size(source);
    size_t offset = 0;
    size_t best_batch = (size_t)-1;
    size_t best_bytes = (size_t)-1;
    bool found = false;
    for (size_t candidate = 0; candidate < planned_device_entries.size() + 2; candidate++)
    {
        const size_t start = candidate == 0 ? head : candidate == 1 ? 0 : planned_device_entries[candidate - 2].target.buffer_offset() + planned_device_entries[candidate - 2].target.buffer_capacity();
        if (start > device_size || size > device_size - start)
            continue;
        size_t last_batch = 0;
        size_t bytes = 0;
        bool pinned_overlap = false;
        for (size_t i = 0; i < planned_device_entries.size(); i++)
        {
            const weight_entry& entry = planned_device_entries[i];
            if ((!entry.source_identity && !entry.valid) || start >= entry.target.buffer_offset() + entry.target.buffer_capacity() || entry.target.buffer_offset() >= start + size)
                continue;
            if (is_pinned(entry.source_identity, pinned))
            {
                pinned_overlap = true;
                break;
            }
            last_batch = std::max(last_batch, entry.last_batch);
            bytes += entry.target.buffer_capacity();
        }
        if (!pinned_overlap && (!found || last_batch < best_batch || (last_batch == best_batch && bytes < best_bytes)))
        {
            found = true;
            offset = start;
            best_batch = last_batch;
            best_bytes = bytes;
            if (!bytes)
                break;
        }
    }
    if (!found)
        return -1;

    int slot = -1;
    for (size_t i = 0; i < planned_device_entries.size(); i++)
    {
        VkWeightCache::weight_entry& e = planned_device_entries[i];
        if ((e.source_identity || e.valid) && offset < e.target.buffer_offset() + e.target.buffer_capacity() && e.target.buffer_offset() < offset + size)
        {
            if (e.valid)
                evicted.push_back(e);
            e.source_identity = 0;
            e.valid = false;
        }
        if (!e.source_identity && !e.valid)
            slot = (int)i;
    }
    VkWeightCache::weight_entry entry;
    entry.target = cache_view(false, offset, size);
    entry.source_identity = view_allocator.identity(source);
    entry.source_generation = source_generations[recorded];
    entry.last_batch = 0;
    entry.valid = false;
    if (slot == -1)
    {
        slot = (int)planned_device_entries.size();
        planned_device_entries.push_back(entry);
    }
    else
    {
        planned_device_entries[slot] = entry;
    }
    head = offset + size;
    return slot;
}

VkWeightCacheTransfer::VkWeightCacheTransfer(VkWeightCache& _cache)
    : cache(_cache), vkdev(_cache.device())
{
    error = 0;
    execution_serial_base = 0;
    compute_command_pool = 0;
    transfer_command_pool = 0;
    prefix_fence = 0;
    release_command_buffer = 0;
    release_fence = 0;
    release_pending = false;
    transfer_queue_family_index = vkdev->info.transfer_queue_family_index();
    release_semaphore = 0;
    release_signal_pending = false;
    current_batch = 0;
    compute_serial = 0;
    completed_compute_serial = 0;
    commands_initialized = false;
    executing = false;
}

VkWeightCacheTransfer::~VkWeightCacheTransfer()
{
    drain();
    destroy_commands();
}

int VkWeightCacheTransfer::record_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last)
{
    const VkBufferMemory* src = source.data;
    int ret;
    if (!(src->usage_flags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return -1;
    ret = begin_command_buffer(UPLOAD_COMMAND, task);
    if (ret != 0)
        return ret;
    VkResult vkret;
    for (size_t i = 0; i < pending_acquires.size(); i++)
        record_upload_acquire_from_compute(task, pending_acquires[i]);
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

int VkWeightCacheTransfer::begin_execution(const std::vector<VkMat>& dependencies, const std::vector<size_t>& dependency_offsets)
{
    if (error || executing)
        return -1;
    error = cache.plan_batches(dependencies, dependency_offsets, plan);
    if (error != 0)
        return error;
    transfer_queue_family_index = cache.is_unified() ? vkdev->info.compute_queue_family_index() : vkdev->info.transfer_queue_family_index();
    commands_initialized = true;
    error = create_commands();
    if (error != 0)
        return error;
    execution_serial_base = compute_serial;
    executing = true;
#if NCNN_BENCHMARK
    bytes_read = 0;
    bytes_uploaded = 0;
    start_time = get_current_time();
#endif // NCNN_BENCHMARK
    return release_ranges(plan.initial_releases);
}

int VkWeightCacheTransfer::submit_prefix(VkCommandBuffer command_buffer, VkFence fence)
{
    if (!executing || prefix_fence)
        return -1;
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = 0;
    int ret = submit_command(false, submitInfo, fence);
    if (ret != 0)
        return ret;
    prefix_fence = fence;
    return 0;
}

int VkWeightCacheTransfer::submit_command(bool transfer, const VkSubmitInfo& info, VkFence fence)
{
    const int stream = transfer ? 1 : 0;
    VkSemaphore signal = 0;
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
    if (vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &signal) != VK_SUCCESS)
        return -1;
    // Retain even an unsubmitted signal until execution cleanup.
    const VkSemaphore previous = queue_signals[stream].empty() ? 0 : queue_signals[stream].back();
    queue_signals[stream].push_back(signal);
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
    if (transfer)
    {
        // Carry prior compute releases into this transfer stream. Later chunks
        // inherit the dependency through the stream's semaphore chain.
        if (release_signal_pending)
        {
            waits.push_back(release_semaphore);
            stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        for (size_t i = 0; i < compute_upload_semaphores.size(); i++)
        {
            if (!compute_release_pending[i])
                continue;
            waits.push_back(compute_upload_semaphores[i]);
            stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
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

    const uint32_t family = transfer ? transfer_queue_family_index : vkdev->info.compute_queue_family_index();
    VkQueue queue = vkdev->acquire_queue(family);
    if (!queue)
    {
        NCNN_LOGE("out of weight %s queue", transfer ? "transfer" : "compute");
        return -1;
    }
    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, fence);
    vkdev->reclaim_queue(family, queue);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return -1;
    }
    if (transfer)
    {
        release_signal_pending = false;
        for (size_t i = 0; i < compute_release_pending.size(); i++)
            compute_release_pending[i] = false;
    }
    return 0;
}

int VkWeightCacheTransfer::prepare(size_t batch_index)
{
    const weight_batch& batch = plan.batches[batch_index];
    const uint64_t serial_base = execution_serial_base;
    // wait for the last reader before overwriting its ranges
    if (batch.wait_batch)
    {
        int ret = wait_compute(serial_base + batch.wait_batch);
        if (ret != 0)
            return ret;
    }
    if (!cache.is_unified() && transfer_queue_family_index != vkdev->info.compute_queue_family_index())
        pending_acquires = batch.acquires;
    for (size_t i = 0; i < batch.weights.size(); i++)
    {
        if (!batch.missing[i])
            continue;

        int ret = upload_source_to_device_cache(batch.weights[i], batch.targets[i]);
        if (ret != 0)
            return ret;
    }
    current_batch = &batch;
    return 0;
}

int VkWeightCacheTransfer::release_ranges(const std::vector<buffer_range>& ranges)
{
    if (ranges.empty() || cache.is_unified() || transfer_queue_family_index == vkdev->info.compute_queue_family_index())
        return 0;
    int ret = begin_command_buffer(RELEASE_COMMAND);
    if (ret != 0)
        return ret;

    VkResult vkret;
    for (size_t i = 0; i < ranges.size(); i++)
        record_initial_release_to_upload(ranges[i]);
    vkret = vkEndCommandBuffer(release_command_buffer);
    if (vkret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", vkret);
        return -1;
    }
    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
    if (vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &release_semaphore) != VK_SUCCESS)
        return -1;
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &release_command_buffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &release_semaphore;
    ret = submit_command(false, submitInfo, release_fence);
    if (ret != 0)
        return ret;
    release_signal_pending = true;
    release_pending = true;
    return 0;
}

int VkWeightCacheTransfer::begin_compute(VkCommandBuffer* command_buffer)
{
    if (compute_command_buffers.size() >= (size_t)INT_MAX)
        return -1;
    const int task = (int)compute_command_buffers.size();
    compute_command_buffers.push_back(0);
    compute_fences.push_back(0);
    compute_upload_semaphores.push_back(0);
    compute_release_pending.push_back(false);
    compute_pending.push_back(false);
    compute_serials.push_back(0);
    int ret = allocate_command_buffer(COMPUTE_COMMAND, task);
    if (ret != 0)
        return ret;
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
    if (vkCreateFence(vkdev->vkdevice(), &fence_info, 0, &compute_fences[task]) != VK_SUCCESS)
        return -1;
    ret = begin_command_buffer(COMPUTE_COMMAND, task);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        const VkMat& target = upload_targets[pending_uploads[i]];
        const buffer_range range = {target.buffer_offset(), target.buffer_capacity()};
        record_compute_acquire_from_upload(task, range);
    }
    if (cache.is_unified())
    {
        for (size_t i = 0; i < current_batch->weights.size(); i++)
        {
            const VkMat& source = current_batch->weights[i];
            if (!needs_host_copy(source))
                continue;
            // barrier device host-write @ host to shader-read @ compute
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = current_batch->targets[i].buffer();
            barrier.offset = current_batch->targets[i].buffer_offset();
            barrier.size = cache.aligned_size(source);
            vkCmdPipelineBarrier(compute_command_buffers[task], VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    *command_buffer = compute_command_buffers[task];
    return task;
}

int VkWeightCacheTransfer::submit_compute(int slot)
{
    int ret;
    if (!cache.is_unified() && transfer_queue_family_index != vkdev->info.compute_queue_family_index())
        for (size_t i = 0; i < current_batch->releases.size(); i++)
            record_compute_release_to_upload(slot, current_batch->releases[i]);
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(compute_command_buffers[slot]);
#endif
    VkResult vkret = vkEndCommandBuffer(compute_command_buffers[slot]);
    if (vkret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", vkret);
        return -1;
    }
    std::vector<VkSemaphore> semaphores(pending_uploads.size());
    std::vector<VkPipelineStageFlags> stages(pending_uploads.size());
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        semaphores[i] = upload_compute_semaphores[pending_uploads[i]];
        stages[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
            stages[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    bool release = false;
    if (!cache.is_unified() && transfer_queue_family_index != vkdev->info.compute_queue_family_index() && !current_batch->releases.empty())
    {
        VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0};
        if (vkCreateSemaphore(vkdev->vkdevice(), &semaphore_info, 0, &compute_upload_semaphores[slot]) != VK_SUCCESS)
            return -1;
        release = true;
    }
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = (uint32_t)pending_uploads.size();
    submitInfo.pWaitSemaphores = semaphores.data();
    submitInfo.pWaitDstStageMask = stages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &compute_command_buffers[slot];
    submitInfo.signalSemaphoreCount = release ? 1 : 0;
    submitInfo.pSignalSemaphores = release ? &compute_upload_semaphores[slot] : 0;
    ret = submit_command(false, submitInfo, compute_fences[slot]);
    if (ret != 0)
        return ret;
    if (release)
        compute_release_pending[slot] = true;
    compute_serials[slot] = ++compute_serial;
    compute_pending[slot] = true;
    pending_uploads.clear();
    current_batch = 0;
    return 0;
}

int VkWeightCacheTransfer::finish(int result)
{
    if (result == VK_ERROR_DEVICE_LOST)
        error = result;
    if (!executing)
        return result ? result : error;
    bool incomplete = false;
    for (size_t i = 0; i < compute_fences.size(); i++)
    {
        if (!compute_pending[i])
            continue;
        const int ret = wait_fence(compute_fences[i]);
        if (ret == VK_ERROR_DEVICE_LOST)
            error = ret;
        if (ret != 0)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
        {
            compute_pending[i] = false;
            if (ret == 0)
                completed_compute_serial = std::max(completed_compute_serial, compute_serials[i]);
        }
        else
            incomplete = true;
    }
    for (size_t i = 0; i < upload_fences.size(); i++)
    {
        if (!upload_pending[i])
            continue;
        const int ret = wait_fence(upload_fences[i]);
        if (ret == VK_ERROR_DEVICE_LOST)
            error = ret;
        if (ret != 0)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
            upload_pending[i] = false;
        else
            incomplete = true;
    }
    if (prefix_fence)
    {
        const int ret = wait_fence(prefix_fence);
        if (ret == VK_ERROR_DEVICE_LOST)
            error = ret;
        if (ret != 0)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
            prefix_fence = 0;
        else
            incomplete = true;
    }
    if (release_pending)
    {
        const int ret = wait_fence(release_fence);
        if (ret == VK_ERROR_DEVICE_LOST)
            error = ret;
        if (ret != 0)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
            release_pending = false;
        else
            incomplete = true;
    }
    if (result != 0 && !error)
        error = result;
    if (error)
    {
        cache.invalidate(error);
        result = error;
    }
    if (incomplete)
        return result;
    if (!error)
        cache.publish_device_plan(plan);
    executing = false;
    pending_uploads.clear();
    current_batch = 0;
    // All producers and all consumers have completed, including semaphore waits.
    destroy_commands();
#if NCNN_BENCHMARK
    NCNN_LOGE("weight cache read=%llu upload=%llu time=%.2fms", (unsigned long long)bytes_read, (unsigned long long)bytes_uploaded, get_current_time() - start_time);
#endif
    return result;
}

void VkWeightCacheTransfer::drain()
{
    while (executing)
    {
        finish(error);
        if (executing)
            wait_for_pending_commands();
    }
}

int VkWeightCacheTransfer::create_commands()
{
    int ret;
    VkCommandPoolCreateInfo commandPoolCreateInfo;
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = 0;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = transfer_queue_family_index;
    VkResult vkret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
    if (vkret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateCommandPool failed %d", vkret);
        return -1;
    }
    commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();
    vkret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
    if (vkret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateCommandPool failed %d", vkret);
        return -1;
    }
    VkFenceCreateInfo fenceCreateInfo;
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = 0;
    fenceCreateInfo.flags = 0;
    if (!cache.is_unified() && transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        ret = allocate_command_buffer(RELEASE_COMMAND);
        if (ret != 0)
            return ret;
        vkret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &release_fence);
        if (vkret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", vkret);
            return -1;
        }
    }
    return 0;
}

void VkWeightCacheTransfer::destroy_commands()
{
    if (!commands_initialized)
        return;
    if (executing)
    {
        drain();
        return;
    }
    for (size_t i = 0; i < upload_fences.size(); i++)
    {
        vkDestroyFence(vkdev->vkdevice(), upload_fences[i], 0);
        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphores[i], 0);
    }
    for (size_t i = 0; i < compute_fences.size(); i++)
    {
        vkDestroyFence(vkdev->vkdevice(), compute_fences[i], 0);
        vkDestroySemaphore(vkdev->vkdevice(), compute_upload_semaphores[i], 0);
    }
    vkDestroyFence(vkdev->vkdevice(), release_fence, 0);
    vkDestroySemaphore(vkdev->vkdevice(), release_semaphore, 0);
    release_command_buffer = 0;
    release_fence = 0;
    release_semaphore = 0;
    release_pending = release_signal_pending = false;
    for (int stream = 0; stream < 2; stream++)
    {
        for (size_t i = 0; i < queue_signals[stream].size(); i++)
            vkDestroySemaphore(vkdev->vkdevice(), queue_signals[stream][i], 0);
        queue_signals[stream].clear();
    }
    if (transfer_command_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    if (compute_command_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
    compute_command_pool = transfer_command_pool = 0;
    prefix_fence = 0;
    upload_command_buffers.clear();
    upload_fences.clear();
    upload_compute_semaphores.clear();
    upload_pending.clear();
    upload_sources.clear();
    upload_targets.clear();
    compute_command_buffers.clear();
    compute_fences.clear();
    compute_upload_semaphores.clear();
    compute_release_pending.clear();
    compute_pending.clear();
    compute_serials.clear();
    commands_initialized = false;
    pending_acquires.clear();
    pending_uploads.clear();
    current_batch = 0;
    plan.batches.clear();
    plan.initial_releases.clear();
    plan.device_entries.clear();
}

int VkWeightCacheTransfer::upload_source_to_device_cache(const VkMat& source, const VkMat& target)
{
    // Cache the full allocation shared by recorded tensor views.
    const VkBufferMemory* src = source.data;
    const size_t size = target.buffer_capacity();
    if (cache.is_unified() && needs_host_copy(source))
    {
        unsigned char* dst = (unsigned char*)target.mapped_ptr();
        memcpy(dst, src->source_ptr, src->source_size);
#if NCNN_BENCHMARK
        if (src->storage_type == 2)
            bytes_read += src->source_size;
#endif
        memset(dst + src->source_size, 0, size - src->source_size);
        return flush_weight_buffer(vkdev, target.data, 0, size);
    }
    if (!needs_host_copy(source))
    {
        const int task = create_upload();
        if (task < 0)
            return task;
        return submit_device_cache_copy(task, source, target, source.buffer_offset(), 0, alignSize(src->source_size, 4), true);
    }
    if (size <= cache.get_host_size())
    {
        VkMat host;
        int ret = fill_host_cache(source, host);
        if (ret != 0)
            return ret;
        const int task = create_upload();
        if (task < 0)
            return task;
        return submit_device_cache_copy(task, host, target, host.buffer_offset(), 0, alignSize(src->source_size, 4), true);
    }
    const size_t alignment = cache.get_alignment();
    const size_t chunk_capacity = std::min((size_t)INT_MAX / alignment * alignment, std::max(alignment, cache.get_host_size() / 2 / alignment * alignment));
    for (size_t file_offset = 0; file_offset < src->source_size;)
    {
        const size_t chunk_size = std::min(chunk_capacity, src->source_size - file_offset);
        const size_t host_range_size = alignSize(chunk_size, alignment);
        size_t host_offset;
        int ret = allocate_host_range(host_range_size, host_offset);
        if (ret != 0)
            return ret;
        VkMat host = cache.host_range(host_offset, host_range_size);
        if (host.empty())
            return -1;
        unsigned char* dst = (unsigned char*)host.mapped_ptr();
        memcpy(dst, (const unsigned char*)src->source_ptr + file_offset, chunk_size);
#if NCNN_BENCHMARK
        if (src->storage_type == 2)
            bytes_read += chunk_size;
#endif
        memset(dst + chunk_size, 0, host_range_size - chunk_size);
        ret = flush_weight_buffer(vkdev, host.data, 0, host_range_size);
        if (ret != 0)
            return ret;
        const int task = create_upload();
        if (task < 0)
            return task;
        const bool last = chunk_size == src->source_size - file_offset;
        ret = submit_device_cache_copy(task, host, target, host.buffer_offset(), file_offset, alignSize(chunk_size, 4), last);
        if (ret != 0)
            return ret;
        file_offset += chunk_size;
    }
    return 0;
}

int VkWeightCacheTransfer::submit_device_cache_copy(int task, const VkMat& source, const VkMat& target, size_t src_offset, size_t target_offset, size_t size, bool last)
{
    int ret = record_device_cache_copy(task, source, target, src_offset, target_offset, size, last);
    if (ret != 0)
        return ret;
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &upload_command_buffers[task];
    submitInfo.signalSemaphoreCount = 0;
    if (last)
        submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &upload_compute_semaphores[task];
    ret = submit_command(true, submitInfo, upload_fences[task]);
    if (ret != 0)
        return ret;
#if NCNN_BENCHMARK
    bytes_uploaded += size;
#endif // NCNN_BENCHMARK
    upload_pending[task] = true;
    upload_sources[task] = source;
    upload_targets[task] = target;
    pending_acquires.clear();
    if (last)
        pending_uploads.push_back(task);
    return 0;
}

int VkWeightCacheTransfer::wait_fence(VkFence fence)
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

int VkWeightCacheTransfer::wait_compute(uint64_t serial)
{
    if (serial <= completed_compute_serial)
        return 0;
    for (size_t i = 0; i < compute_fences.size(); i++)
    {
        if (compute_pending[i] && compute_serials[i] == serial)
        {
            int ret = wait_fence(compute_fences[i]);
            if (ret != 0)
                return ret;
            completed_compute_serial = serial;
            return 0;
        }
    }
    return -1;
}

int VkWeightCacheTransfer::create_upload()
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
    int ret = allocate_command_buffer(UPLOAD_COMMAND, task);
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

int VkWeightCacheTransfer::allocate_host_range(size_t size, size_t& offset)
{
    int ret = cache.select_host_range(size, offset);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < upload_fences.size(); i++)
    {
        VkMat& source = upload_sources[i];
        if (!source.empty() && source.buffer() == cache.get_host_buffer().buffer()
                && offset < source.buffer_offset() + source.buffer_capacity() && source.buffer_offset() < offset + size)
        {
            int ret = wait_fence(upload_fences[i]);
            if (ret != 0)
                return ret;
            source.release();
        }
    }
    cache.reserve_host_range(offset, size);
    return 0;
}

int VkWeightCacheTransfer::fill_host_cache(const VkMat& source, VkMat& target)
{
    const int cached = cache.find_host_entry(source);
    if (cached >= 0)
    {
        target = cache.host_entry(cached).target;
        return 0;
    }
    const VkBufferMemory* src = source.data;
    const size_t size = cache.aligned_size(source);
    size_t offset;
    int ret = allocate_host_range(size, offset);
    if (ret != 0)
        return ret;
    unsigned char* dst = (unsigned char*)cache.get_host_buffer().mapped_ptr() + offset;
    memcpy(dst, src->source_ptr, src->source_size);
#if NCNN_BENCHMARK
    if (src->storage_type == 2)
        bytes_read += src->source_size;
#endif
    memset(dst + src->source_size, 0, size - src->source_size);
    ret = flush_host(offset, size);
    if (ret != 0)
        return ret;
    ret = cache.publish_host_entry(source, offset, size);
    if (ret == 0)
        target = cache.host_entry(cache.find_host_entry(source)).target;
    return ret;
}

int VkWeightCacheTransfer::flush_host(size_t offset, size_t size)
{
    return flush_weight_buffer(vkdev, cache.get_host_buffer().data, offset, size);
}

void VkWeightCacheTransfer::record_upload_release_to_compute(int task, const buffer_range& range)
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
    if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        barrier.srcQueueFamilyIndex = transfer_queue_family_index;
        barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
    }
    barrier.buffer = cache.get_device_buffer().buffer();
    barrier.offset = cache.get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCacheTransfer::record_compute_acquire_from_upload(int task, const buffer_range& range)
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
    if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        barrier.srcQueueFamilyIndex = transfer_queue_family_index;
        barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
    }
    barrier.buffer = cache.get_device_buffer().buffer();
    barrier.offset = cache.get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCacheTransfer::record_compute_release_to_upload(int task, const buffer_range& range)
{
    const VkCommandBuffer command_buffer = compute_command_buffers[task];
    // release device shader-read @ compute to null @ transfer
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        barrier.srcQueueFamilyIndex = vkdev->info.compute_queue_family_index();
        barrier.dstQueueFamilyIndex = transfer_queue_family_index;
    }
    barrier.buffer = cache.get_device_buffer().buffer();
    barrier.offset = cache.get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCacheTransfer::record_initial_release_to_upload(const buffer_range& range)
{
    const VkCommandBuffer command_buffer = release_command_buffer;
    // release device shader-read @ compute to null @ transfer
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        barrier.srcQueueFamilyIndex = vkdev->info.compute_queue_family_index();
        barrier.dstQueueFamilyIndex = transfer_queue_family_index;
    }
    barrier.buffer = cache.get_device_buffer().buffer();
    barrier.offset = cache.get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCacheTransfer::record_upload_acquire_from_compute(int task, const buffer_range& range)
{
    const VkCommandBuffer command_buffer = upload_command_buffers[task];
    // acquire device null @ compute to transfer-write @ transfer
    VkBufferMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = 0;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        barrier.srcQueueFamilyIndex = vkdev->info.compute_queue_family_index();
        barrier.dstQueueFamilyIndex = transfer_queue_family_index;
    }
    barrier.buffer = cache.get_device_buffer().buffer();
    barrier.offset = cache.get_device_buffer().buffer_offset() + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

class VkComputePrivate
{
public:
    VkComputePrivate(const VulkanDevice* _vkdev, const Option* opt);
    ~VkComputePrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();

    bool deferred() const
    {
        return weight_deferred || !vkdev->info.support_VK_KHR_push_descriptor();
    }
    int ensure_weight_cache(const VkMat& mat);
    int submit_managed();
    void clear_records();

    int resolve_weight_binding(const buffer_binding& binding, VkDescriptorBufferInfo& descriptor) const;

    size_t weight_device_cache_size;
    size_t weight_host_cache_size;
    size_t weight_device_memory_limit;
    size_t weight_host_memory_limit;
    VkWeightCache* weight_cache;
    VkWeightCacheTransfer* weight_transfer;
    bool weight_deferred;
    bool command_buffer_has_work;
    int record_error;

    struct pipeline_record
    {
        VkPipeline pipeline;
        VkPipelineLayout pipeline_layout;
        VkDescriptorSetLayout descriptorset_layout;
        VkDescriptorUpdateTemplateKHR descriptor_update_template;
        ShaderInfo shader_info;
        std::vector<buffer_binding> buffers;
        std::vector<VkDescriptorImageInfo> images;
        std::vector<vk_constant_type> constants;
        uint32_t group_count_x;
        uint32_t group_count_y;
        uint32_t group_count_z;
    };
    std::vector<pipeline_record> pipelines;
    int create_descriptorset(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template,
                             const unsigned char* descriptor_infos, VkDescriptorSet* descriptorset);
    int emit_pipeline(const pipeline_record& r, VkCommandBuffer command_buffer);

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
        union
        {
            size_t pipeline_index;
            struct
            {
                VkBuffer src;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferCopy* regions;
                buffer_binding binding;
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
                buffer_binding binding;
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

    int replay_record(const record& r, VkCommandBuffer command_buffer);
    int record_sources(const record& r, std::vector<VkMat>& weights);

    std::vector<record> deferred_records;

    uint64_t pending_dispatch_total;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
    bool query_results_ready;
#endif // NCNN_BENCHMARK
};

VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev, const Option* opt)
    : vkdev(_vkdev)
{
    weight_device_cache_size = opt ? opt->weight_device_cache_size : 0;
    weight_host_cache_size = opt ? opt->weight_host_cache_size : 0;
    weight_device_memory_limit = opt ? opt->weight_device_memory_limit : 0;
    weight_host_memory_limit = opt ? opt->weight_host_memory_limit : 0;
    weight_cache = 0;
    weight_transfer = 0;
    weight_deferred = false;
    command_buffer_has_work = false;
    record_error = 0;

    compute_command_pool = 0;
    compute_command_buffer = 0;
    compute_command_fence = 0;

    pending_dispatch_total = 0;

#if NCNN_BENCHMARK
    query_count = 0;
    query_pool = 0;
    query_results_ready = false;
#endif // NCNN_BENCHMARK

    record_error = init();
}

VkComputePrivate::~VkComputePrivate()
{
    delete weight_transfer;
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
            command_buffer_has_work = true;
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

    command_buffer_has_work = false;
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

int VkComputePrivate::ensure_weight_cache(const VkMat& mat)
{
    if (record_error != 0)
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
        weight_cache = new VkWeightCache(vkdev, weight_device_cache_size, weight_host_cache_size, weight_device_memory_limit, weight_host_memory_limit);
    if (!weight_transfer)
        weight_transfer = new VkWeightCacheTransfer(*weight_cache);
    weight_deferred = true;
    record_error = weight_cache->record_source(mat);
    return record_error;
}

void VkComputePrivate::clear_records()
{
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        const record& r = deferred_records[i];
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
    deferred_records.clear();
    pipelines.clear();
    if (weight_cache)
        weight_cache->clear_sources();
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

int VkComputePrivate::emit_pipeline(const pipeline_record& r, VkCommandBuffer command_buffer)
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
            const buffer_binding& binding = r.buffers[bi++];
            descriptor = binding.descriptor;
            int ret = resolve_weight_binding(binding, descriptor);
            if (ret != 0)
                return ret;
            memcpy(p, &descriptor, sizeof(descriptor));
            p += sizeof(descriptor);
        }
        else
        {
            memcpy(p, &r.images[ii++], sizeof(VkDescriptorImageInfo));
            p += sizeof(VkDescriptorImageInfo);
        }
    }
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline);
#if NCNN_COVERAGE
    vkdev->bind_shader_coverage(command_buffer, r.pipeline_layout);
#endif
    if (binding_count > 0)
    {
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, r.descriptor_update_template, r.pipeline_layout, 0, descriptorInfos.data());
        }
        else
        {
            VkDescriptorSet descriptorset;
            int ret = create_descriptorset(binding_count, shader_info.binding_types, r.descriptorset_layout, r.descriptor_update_template, descriptorInfos.data(), &descriptorset);
            if (ret != 0)
                return ret;

            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipeline_layout, 0, 1, &descriptorset, 0, 0);
        }
    }
    if (!r.constants.empty())
        vkCmdPushConstants(command_buffer, r.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, r.constants.size() * sizeof(vk_constant_type), r.constants.data());
    vkCmdDispatch(command_buffer, r.group_count_x, r.group_count_y, r.group_count_z);
    return 0;
}

int VkComputePrivate::record_sources(const record& r, std::vector<VkMat>& weights)
{
    if (r.type != record::TYPE_pipeline && r.type != record::TYPE_copy_buffer && r.type != record::TYPE_copy_buffer_to_image)
        return 0;

    const size_t count = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers.size() : 1;
    for (size_t i = 0; i < count; i++)
    {
        const buffer_binding& binding = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers[i] : r.type == record::TYPE_copy_buffer ? r.copy_buffer.binding : r.copy_buffer_to_image.binding;
        VkMat src;
        int ret = weight_cache->binding_source(binding, src);
        if (ret != 0)
            return ret;
        if (src.empty())
            continue;
        bool found = false;
        for (size_t j = 0; j < weights.size(); j++)
            if (weights[j].data == src.data)
                found = true;
        if (!found)
            weights.push_back(src);
    }
    return 0;
}

int VkComputePrivate::submit_managed()
{
    if (weight_transfer->pending() || weight_transfer->status())
        return -1;
    int ret = weight_cache->validate_sources();
    if (ret != 0)
        return ret;
    size_t minimum_device_size = 0;
    size_t max_source_size = 0;
    size_t max_file_size = 0;
    std::vector<VkMat> dependencies;
    std::vector<size_t> dependency_offsets;
    std::vector<VkMat> sources;
    dependency_offsets.push_back(0);
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        sources.clear();
        ret = record_sources(deferred_records[i], sources);
        if (ret != 0)
            return ret;
        size_t total = 0;
        for (size_t j = 0; j < sources.size(); j++)
        {
            const size_t size = weight_cache->aligned_size(sources[j]);
            if (size > (size_t)-1 - total)
            {
                NCNN_LOGE("weight binding size overflow");
                return -1;
            }
            total += size;
            max_source_size = std::max(max_source_size, size);
            if (needs_host_copy(sources[j]))
                max_file_size = std::max(max_file_size, size);
            dependencies.push_back(sources[j]);
        }
        dependency_offsets.push_back(dependencies.size());
        minimum_device_size = std::max(minimum_device_size, total);
    }
    ret = weight_cache->initialize(minimum_device_size, max_source_size, max_file_size);
    if (ret != 0)
        return ret;
    ret = weight_transfer->begin_execution(dependencies, dependency_offsets);
    if (ret != 0)
        return weight_transfer->finish(ret);
    const std::vector<VkWeightCache::weight_batch>& batches = weight_transfer->batches();
    // submit ordinary commands recorded before the first deferred weight
    if (ret == 0 && vkdev->info.support_VK_KHR_push_descriptor() && command_buffer_has_work)
    {
        ret = end_command_buffer();
        if (ret == 0)
            ret = weight_transfer->submit_prefix(compute_command_buffer, compute_command_fence);
    }
    for (size_t i = 0; i < batches.size() && ret == 0; i++)
    {
        ret = weight_transfer->prepare(i);
        if (ret != 0)
            break;
        VkCommandBuffer command_buffer;
        int slot = weight_transfer->begin_compute(&command_buffer);
        if (slot < 0)
        {
            ret = slot;
            break;
        }
#if NCNN_BENCHMARK
        if (i == 0 && query_pool && !vkdev->info.support_VK_KHR_push_descriptor())
            vkCmdResetQueryPool(command_buffer, query_pool, 0, query_count);
#endif // NCNN_BENCHMARK
        for (size_t j = batches[i].begin; j < batches[i].end && ret == 0; j++)
            ret = replay_record(deferred_records[j], command_buffer);
        if (ret == 0)
        {
            ret = weight_transfer->submit_compute(slot);
        }
    }
    ret = weight_transfer->finish(ret);
    return ret;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev, 0))
{
}

VkCompute::VkCompute(const VulkanDevice* _vkdev, const Option& opt)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev, &opt))
{
}

VkCompute::~VkCompute()
{
    delete d;
}

void VkCompute::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

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
    {
        d->record_error = -100;
        return;
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
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
    {
        d->record_error = -100;
        return;
    }

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
    if (dst_staging.allocator->flush(dst_staging.data))
    {
        d->record_error = -1;
        return;
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
    if (dst.empty())
        d->record_error = -100;
}

void VkCompute::record_download(const VkMat& src, Mat& dst, const Option& opt)
{
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    if (d->ensure_weight_cache(src))
        return;

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

    Mat dst_fp16;
    record_clone(dst_staging, dst_fp16, opt);
    if (d->record_error)
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
            {
                d->record_error = -100;
                return;
            }

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
            {
                d->record_error = -100;
                return;
            }

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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    //     NCNN_LOGE("record_clone host to buffer");

    // host to staging
    VkMat dst_staging;
    dst_staging.create_like(src, opt.staging_vkallocator);
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
    {
        d->record_error = -100;
        return;
    }

    // memcpy src to device
    for (int b = 0; b < src.n; b++)
    {
        const Mat src_b = src.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }
    if (dst_staging.allocator->flush(dst_staging.data))
    {
        d->record_error = -1;
        return;
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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    if (d->ensure_weight_cache(src))
        return;

    //     NCNN_LOGE("record_clone buffer to host");

    VkMat src_staging;
    if (!src.mapped_ptr() || is_deferred_buffer(src.data))
    {
        Option opt_staging = opt;
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
        record_clone(src, src_staging, opt_staging);
        if (d->record_error)
            return;
    }
    else
    {
        src_staging = src;
    }
    if (!src_staging.mapped_ptr())
    {
        d->record_error = -100;
        return;
    }

    // create dst
    dst.create_like(src_staging, opt.blob_allocator);
    if (dst.empty())
    {
        d->record_error = -100;
        return;
    }

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
            d->command_buffer_has_work = true;
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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    if (d->ensure_weight_cache(src))
        return;

    //     NCNN_LOGE("record_clone buffer to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
    {
        d->record_error = -100;
        return;
    }

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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
            vkCmdCopyBuffer(d->compute_command_buffer, src.buffer(), dst.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer;
            r.copy_buffer.src = src.buffer();
            r.copy_buffer.binding.source_identity = 0;
            if (is_deferred_buffer(src.data))
                r.copy_buffer.binding.source_identity = (uintptr_t)src.data;
            r.copy_buffer.binding.view_offset = src.offset;
            r.copy_buffer.binding.descriptor.buffer = src.buffer();
            r.copy_buffer.binding.descriptor.offset = src.buffer_offset();
            r.copy_buffer.binding.descriptor.range = src.buffer_capacity();
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            d->deferred_records.push_back(r);
        }
    }
}

void VkCompute::record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt)
{
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    //     NCNN_LOGE("record_clone image to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
    {
        d->record_error = -100;
        return;
    }

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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    if (d->ensure_weight_cache(src))
        return;

    //     NCNN_LOGE("record_clone buffer to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
    {
        d->record_error = -100;
        return;
    }

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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
            vkCmdCopyBufferToImage(d->compute_command_buffer, src.buffer(), dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer_to_image;
            r.copy_buffer_to_image.src = src.buffer();
            r.copy_buffer_to_image.binding.source_identity = 0;
            if (is_deferred_buffer(src.data))
                r.copy_buffer_to_image.binding.source_identity = (uintptr_t)src.data;
            r.copy_buffer_to_image.binding.view_offset = src.offset;
            r.copy_buffer_to_image.binding.descriptor.buffer = src.buffer();
            r.copy_buffer_to_image.binding.descriptor.offset = src.buffer_offset();
            r.copy_buffer_to_image.binding.descriptor.range = src.buffer_capacity();
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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -1;
        return;
    }

    //     NCNN_LOGE("record_clone image to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
    {
        d->record_error = -100;
        return;
    }

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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants,
                                const VkMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants,
                                const VkImageMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants,
                                const Mat& dispatcher)
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

    if (d->deferred())
    {
        VkComputePrivate::pipeline_record p;
        p.pipeline = pipeline->pipeline();
        p.pipeline_layout = pipeline->pipeline_layout();
        p.descriptorset_layout = pipeline->descriptorset_layout();
        p.descriptor_update_template = pipeline->descriptor_update_template();
        p.shader_info = shader_info;
        for (int i = 0; i < buffer_binding_count; i++)
        {
            const VkMat& binding = buffer_bindings[i].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[i];
            buffer_binding b;
            b.source_identity = 0;
            if (is_deferred_buffer(binding.data))
                b.source_identity = (uintptr_t)binding.data;
            b.view_offset = binding.offset;
            b.descriptor.buffer = binding.buffer();
            b.descriptor.offset = binding.buffer_offset();
            b.descriptor.range = binding.buffer_capacity();
            p.buffers.push_back(b);
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
        d->deferred_records.push_back(r);
        return;
    }

    // record bind pipeline
    {
        d->command_buffer_has_work = true;
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

        d->command_buffer_has_work = true;
        vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffer, pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, descriptorInfos.data());
    }

    // record push constants
    if (constant_count > 0)
    {
        d->command_buffer_has_work = true;
        vkCmdPushConstants(d->compute_command_buffer, pipeline->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dispatcher.h * (dispatcher.d ? dispatcher.d : 1) + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        d->command_buffer_has_work = true;
        vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);

        d->pending_dispatch_total += group_count_x * group_count_y * group_count_z;
    }
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (!d->deferred())
    {
        if (d->query_pool)
        {
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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

    // record bind pipeline
    {
        if (!d->deferred())
        {
            d->command_buffer_has_work = true;
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
            d->deferred_records.push_back(r);
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

            if (d->weight_deferred)
            {
                VkComputePrivate::record r;
                r.type = VkComputePrivate::record::TYPE_push_descriptors;
                r.push_descriptors.update_template = pipeline->descriptor_update_template();
                r.push_descriptors.pipeline_layout = pipeline->pipeline_layout();
                unsigned char* data = new unsigned char[sizeof(info)];
                memcpy(data, &info, sizeof(info));
                r.push_descriptors.data = data;
                d->deferred_records.push_back(r);
            }
            else
            {
                d->command_buffer_has_work = true;
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
                    d->record_error = -1;
                    return;
                }
            }
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
                    vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pool, 0);
                    d->record_error = -1;
                    return;
                }
            }
            d->descriptor_pools.push_back(descriptor_pool);
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
            d->deferred_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dst.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dst.h + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dst.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        if (!d->deferred())
        {
            d->command_buffer_has_work = true;
            vkCmdDispatch(d->compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_dispatch;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            d->deferred_records.push_back(r);
        }
    }
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

int VkComputePrivate::replay_record(const record& r, VkCommandBuffer command_buffer)
{
    switch (r.type)
    {
    case VkComputePrivate::record::TYPE_pipeline:
        return emit_pipeline(pipelines[r.pipeline_index], command_buffer);
    case VkComputePrivate::record::TYPE_copy_buffer:
    {
        if (!weight_cache || !r.copy_buffer.binding.source_identity)
        {
            vkCmdCopyBuffer(command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight_binding(r.copy_buffer.binding, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferCopy> regions(r.copy_buffer.region_count);
        for (uint32_t i = 0; i < r.copy_buffer.region_count; i++)
        {
            regions[i] = r.copy_buffer.regions[i];
            regions[i].srcOffset = descriptor.offset + regions[i].srcOffset - r.copy_buffer.binding.descriptor.offset;
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
        if (!weight_cache || !r.copy_buffer_to_image.binding.source_identity)
        {
            vkCmdCopyBufferToImage(command_buffer, r.copy_buffer_to_image.src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, r.copy_buffer_to_image.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight_binding(r.copy_buffer_to_image.binding, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferImageCopy> regions(r.copy_buffer_to_image.region_count);
        for (uint32_t i = 0; i < r.copy_buffer_to_image.region_count; i++)
        {
            regions[i] = r.copy_buffer_to_image.regions[i];
            regions[i].bufferOffset = descriptor.offset + regions[i].bufferOffset - r.copy_buffer_to_image.binding.descriptor.offset;
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
        vkCmdBindDescriptorSets(command_buffer, r.bind_descriptorsets.bind_point, r.bind_descriptorsets.pipeline_layout, 0, r.bind_descriptorsets.descriptorset_count,
                                &descriptorsets[r.bind_descriptorsets.descriptorset_offset], 0, 0);
        break;
    }
    case VkComputePrivate::record::TYPE_push_descriptors:
    {
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

    if (d->record_error)
        return d->record_error;
    bool has_deferred_weights = false;
    if (d->weight_deferred)
    {
        for (size_t i = 0; i < d->deferred_records.size() && !has_deferred_weights; i++)
        {
            const VkComputePrivate::record& r = d->deferred_records[i];
            if (r.type == VkComputePrivate::record::TYPE_pipeline)
            {
                const std::vector<buffer_binding>& buffers = d->pipelines[r.pipeline_index].buffers;
                for (size_t j = 0; j < buffers.size(); j++)
                    has_deferred_weights |= buffers[j].source_identity != 0;
            }
            else if (r.type == VkComputePrivate::record::TYPE_copy_buffer || r.type == VkComputePrivate::record::TYPE_copy_buffer_to_image)
            {
                has_deferred_weights = r.type == VkComputePrivate::record::TYPE_copy_buffer ? r.copy_buffer.binding.source_identity != 0 : r.copy_buffer_to_image.binding.source_identity != 0;
            }
        }
    }
    if (has_deferred_weights)
    {
        int ret = d->submit_managed();
        if (ret != 0)
            return ret;
    }
    else
    {
        if (!vkdev->info.support_VK_KHR_push_descriptor() || d->weight_deferred)
        {
            if (!vkdev->info.support_VK_KHR_push_descriptor())
            {
                d->record_error = d->begin_command_buffer();
                if (d->record_error != 0)
                    return d->record_error;

#if NCNN_BENCHMARK
                if (d->query_pool)
                {
                    d->command_buffer_has_work = true;
                    vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
                }
#endif // NCNN_BENCHMARK
            }

            const size_t record_count = d->deferred_records.size();

            // handle deferred records
            for (size_t i = 0; i < record_count; i++)
            {
                const VkComputePrivate::record& r = d->deferred_records[i];

                int ret = d->replay_record(r, d->compute_command_buffer);
                if (ret != 0)
                    return ret;
            }
        }

        // end command buffer
        {
            d->record_error = d->end_command_buffer();
            if (d->record_error != 0)
                return d->record_error;
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
    if (d->weight_transfer && (d->weight_transfer->pending() || d->weight_transfer->status() == VK_ERROR_DEVICE_LOST))
        return -1;
#if NCNN_BENCHMARK
    d->query_results_ready = false;
#endif // NCNN_BENCHMARK
    d->record_error = 0;
    d->command_buffer_has_work = false;

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

    d->clear_records();
    // discard the cache after a failed upload
    if (d->weight_cache && (!d->weight_cache->valid() || (d->weight_transfer && d->weight_transfer->status())))
    {
        delete d->weight_transfer;
        d->weight_transfer = 0;
        d->weight_cache->clear();
    }
    d->weight_deferred = false;

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
        {
            d->command_buffer_has_work = true;
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
        }
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkCompute::clear_weight_cache()
{
    if ((d->weight_transfer && (d->weight_transfer->pending() || d->weight_transfer->status() == VK_ERROR_DEVICE_LOST)) || !d->deferred_records.empty() || d->pending_dispatch_total)
    {
        NCNN_LOGE("cannot clear a weight cache while recording or executing");
        return -1;
    }
    delete d->weight_transfer;
    d->weight_transfer = 0;
    if (d->weight_cache)
        d->weight_cache->clear();
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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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
            d->command_buffer_has_work = true;
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
    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
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
            d->command_buffer_has_work = true;
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

int VkComputePrivate::resolve_weight_binding(const buffer_binding& binding, VkDescriptorBufferInfo& descriptor) const
{
    descriptor = binding.descriptor;
    if (!binding.source_identity)
        return 0;
    return weight_transfer ? weight_transfer->resolve_binding(binding, descriptor) : -1;
}

int VkWeightCacheTransfer::resolve_binding(const buffer_binding& binding, VkDescriptorBufferInfo& descriptor) const
{
    VkMat source;
    int ret = cache.binding_source(binding, source);
    if (ret != 0)
        return ret;
    descriptor = binding.descriptor;
    if (source.empty())
        return 0;
    for (size_t i = 0; i < current_batch->weights.size(); i++)
    {
        if (current_batch->weights[i].data != source.data)
            continue;
        descriptor.buffer = current_batch->targets[i].buffer();
        descriptor.offset = current_batch->targets[i].buffer_offset() + binding.view_offset;
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
    // Resident views use the const-destination overload and retain immutable file bytes.
    if (dst.data->storage_type == 2)
        record_upload(src_flattened, (const VkMat&)dst, opt);
    else
    {
        Mat snapshot = src_flattened.clone();
        if (snapshot.empty())
        {
            d->upload_error = -100;
            return;
        }
        record_upload(snapshot, (const VkMat&)dst, opt);
    }
}

void VkTransfer::record_upload(const Mat& src, const VkMat& dst, const Option& opt)
{
    if (d->upload_error || src.empty())
        return;
    if (dst.empty() || !dst.allocator || dst.allocator->vkdev != vkdev || src.n != dst.n
            || dst.offset > dst.data->capacity || dst.buffer_capacity() > dst.data->capacity - dst.offset)
    {
        d->upload_error = -1;
        return;
    }
    for (int b = 0; b < src.n; b++)
    {
        const size_t offset = dst.batch(b).offset;
        const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
        if (offset > end || src.batch(b).total() * src.elemsize > end - offset)
        {
            d->upload_error = -1;
            return;
        }
    }
    if (dst.data->storage_type == 2)
    {
        if (!(dst.data->access_flags & VK_ACCESS_HOST_WRITE_BIT) || !dst.data->source_ptr
                || dst.offset > dst.data->source_size || dst.buffer_capacity() > dst.data->source_size - dst.offset)
        {
            NCNN_LOGE("disk weight requires a writable source range during upload");
            d->upload_error = -1;
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
    if (dst.buffer_capacity() > (uint64_t)-1 - d->queued_upload_total)
    {
        d->upload_error = -1;
        return;
    }
    VkTransferPrivate::upload_record r;
    r.source = src;
    r.destination = dst;
    r.host_memory_limit = opt.weight_host_memory_limit;
    r.staging_allocator = opt.staging_vkallocator;
    d->uploads.push_back(r);
    d->queued_upload_total += dst.buffer_capacity();
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

int VkTransfer::error() const
{
    return d->upload_error;
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
