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
int fit_weight_capacity(const VulkanDevice* vkdev, size_t memory_limit, bool automatic, bool concurrent, size_t& capacity);
VkBufferMemory* create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t memory_limit, size_t& allocation_size, bool allocate);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

int get_weight_uploads(VkWeightAllocator* allocator, int storage_type, std::vector<VkBufferMemory>& uploads);
int get_weight_source_generation(const VkAllocator* allocator, uint64_t& source_generation);

struct buffer_binding;

static const int weight_transfer_slots = 32;
static const int weight_compute_slots = 4;

static bool is_deferred_buffer(const VkBufferMemory* memory)
{
    return memory && memory->storage_type != 0 && memory->source_size != 0;
}

static bool needs_host_copy(const VkBufferMemory* memory)
{
    return memory->storage_type == 2 || !memory->buffer;
}

struct buffer_binding
{
    const VkBufferMemory* src;
    size_t view_offset;
    VkDescriptorBufferInfo descriptor;
};

class VkWeightCache
{
public:
    struct weight_entry
    {
        const VkBufferMemory* src;
        const VkAllocator* allocator;
        uint64_t source_generation;
        size_t offset;
        size_t size;
        size_t last_batch;
        bool valid;
    };

    VkWeightCache(const VulkanDevice* vkdev, const Option& opt);
    ~VkWeightCache();
    void clear();
    void clear_sources();
    int record_source(const VkMat& mat);
    int validate_sources() const;
    const weight_entry* find_source(const VkBufferMemory* src) const;

    struct buffer_range
    {
        size_t offset;
        size_t size;
    };

    struct weight_batch
    {
        size_t begin;
        size_t end;
        std::vector<const VkBufferMemory*> weights;
        std::vector<size_t> offsets;
        std::vector<bool> missing;
        std::vector<buffer_range> releases;
        std::vector<buffer_range> acquires;
        size_t wait_batch;
    };

    struct upload_slot
    {
        VkCommandBuffer command_buffer;
        VkFence fence;
        VkSemaphore upload_compute_semaphore;
        bool pending;
        bool ready_pending;
        bool host_active;
        uint64_t consumer_serial;
        size_t host_offset;
        size_t host_size;
        size_t offset;
        size_t size;
    };

    struct compute_slot
    {
        VkCommandBuffer command_buffer;
        VkFence fence;
        uint64_t serial;
        bool pending;
    };

    int initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size);
    int binding_source(const buffer_binding& binding, const VkBufferMemory*& src) const;
    size_t source_size(const VkBufferMemory* src) const;
    int acquire_queues();
    int submit_prefix(VkCommandBuffer command_buffer, VkFence fence);
    int submit_command(bool transfer, const VkSubmitInfo& info, VkFence fence);
    int plan_batches(std::vector<weight_batch>& batches, std::vector<buffer_range>& initial_releases);
    int prepare(const weight_batch& batch, uint64_t serial_base);
    int release_ranges(const std::vector<buffer_range>& ranges);
    int begin_compute(VkCommandBuffer* command_buffer);
    int submit_compute(int slot);
    int finish(int result = 0);
    int find_device_entry(const std::vector<VkWeightCache::weight_entry>& entries, const VkBufferMemory* src) const;
    int find_host_entry(const VkBufferMemory* src) const;
    int allocate();
    int create_commands();
    void destroy_commands();
    int upload(const VkBufferMemory* src, size_t offset);
    int upload_chunk(upload_slot& u, const VkBufferMemory* src, size_t src_offset, size_t offset, size_t size, const buffer_range& range, bool last);
    static bool is_pinned(const VkBufferMemory* src, const std::vector<const VkBufferMemory*>& pinned);
    int place(const VkBufferMemory* src, const std::vector<const VkBufferMemory*>& pinned, std::vector<VkWeightCache::weight_entry>& entries, size_t& head,
              std::vector<VkWeightCache::weight_entry>& evicted);
    int wait_fence(VkFence fence);
    int wait_compute(uint64_t serial);
    int wait_upload_slot();
    int retire_upload(upload_slot& u);
    int allocate_host_range(size_t size, size_t& offset);
    int cache_host(const VkBufferMemory* src, size_t& offset);
    int flush_host(size_t offset, size_t size);
    void release_to_compute(VkCommandBuffer command_buffer, const buffer_range& range);
    void acquire_for_compute(VkCommandBuffer command_buffer, const buffer_range& range);
    void release_to_transfer(VkCommandBuffer command_buffer, const buffer_range& range);
    void acquire_for_transfer(VkCommandBuffer command_buffer, const buffer_range& range);

    // upload and compute submissions
    bool automatic_host_size;
    std::vector<buffer_range> pending_acquires;
    std::vector<int> pending_uploads;
    const weight_batch* current_batch;
    VkCommandPool transfer_command_pool;
    VkQueue compute_queue;
    VkQueue transfer_queue;
    VkFence prefix_fence;
    VkFence handoff_fence;
    VkCommandBuffer release_command_buffer;
    VkFence release_fence;
    bool release_pending;
    uint32_t transfer_queue_family_index;
    upload_slot uploads[weight_transfer_slots];
    compute_slot computes[weight_compute_slots];
    int next_upload;
    int next_compute;
    uint64_t compute_serial;
    uint64_t completed_compute_serial;
    bool commands_initialized;
    bool executing;

    VkCommandPool compute_command_pool;

    // cache configuration and buffers
    const VulkanDevice* vkdev;
    Option opt;
    bool automatic_device_size;
    VkBufferMemory* host_buffer;
    VkBufferMemory* device_buffer;
    size_t host_size;
    size_t device_size;
    size_t alignment;
    size_t host_head;
    size_t device_head;
    // recorded sources and retained cache entries
    std::vector<weight_entry> sources;
    std::vector<weight_entry> entries;
    std::vector<weight_entry> host_entries;
    VkSemaphore handoff_semaphore;
    bool handoff_pending;
    bool unified;
    bool initialized;
    int error;
#if NCNN_BENCHMARK
    uint64_t bytes_read;
    uint64_t bytes_uploaded;
    double start_time;
#endif // NCNN_BENCHMARK
};

static int allocate_command_buffer(VkDevice device, VkCommandPool pool, VkCommandBuffer* command_buffer)
{
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

static int reset_command_buffer(VkCommandBuffer command_buffer)
{
    VkResult ret;
    ret = vkResetCommandBuffer(command_buffer, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
        return -1;
    }
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

VkWeightCache::VkWeightCache(const VulkanDevice* _vkdev, const Option& _opt)
    : vkdev(_vkdev), opt(_opt)
{
    automatic_device_size = opt.weight_device_cache_size == 0;
    automatic_host_size = opt.weight_host_cache_size == 0;
    host_buffer = 0;
    device_buffer = 0;
    host_size = opt.weight_host_cache_size;
    device_size = opt.weight_device_cache_size;
    alignment = weight_buffer_alignment(vkdev);
    host_head = 0;
    device_head = 0;
    handoff_semaphore = 0;
    handoff_pending = false;
    unified = false;
    initialized = false;
    error = 0;
    compute_command_pool = 0;
    transfer_command_pool = 0;
    compute_queue = 0;
    transfer_queue = 0;
    prefix_fence = 0;
    handoff_fence = 0;
    release_command_buffer = 0;
    release_fence = 0;
    release_pending = false;
    transfer_queue_family_index = vkdev->info.transfer_queue_family_index();
    next_upload = 0;
    next_compute = 0;
    current_batch = 0;
    compute_serial = 0;
    completed_compute_serial = 0;
    commands_initialized = false;
    executing = false;
    memset(uploads, 0, sizeof(uploads));
    memset(computes, 0, sizeof(computes));
}

VkWeightCache::~VkWeightCache()
{
    clear();
}

void VkWeightCache::clear()
{
    destroy_commands();
    vkDestroySemaphore(vkdev->vkdevice(), handoff_semaphore, 0);
    if (host_buffer)
        destroy_weight_buffer(vkdev, host_buffer);
    if (device_buffer)
        destroy_weight_buffer(vkdev, device_buffer);
    host_buffer = 0;
    device_buffer = 0;
    host_head = 0;
    device_head = 0;
    entries.clear();
    host_entries.clear();
    handoff_semaphore = 0;
    handoff_pending = false;
    initialized = false;
    error = 0;
}

void VkWeightCache::clear_sources()
{
    sources.clear();
}

const VkWeightCache::weight_entry* VkWeightCache::find_source(const VkBufferMemory* src) const
{
    for (size_t i = 0; i < sources.size(); i++)
        if (sources[i].src == src)
            return &sources[i];
    return 0;
}

int VkWeightCache::record_source(const VkMat& mat)
{
    const VkWeightCache::weight_entry* recorded = find_source(mat.data);
    if (recorded)
    {
        if (recorded->allocator != mat.allocator)
            return -1;
        return 0;
    }

    VkWeightCache::weight_entry src;
    memset(&src, 0, sizeof(src));
    src.src = mat.data;
    src.allocator = mat.allocator;
    src.source_generation = 0;
    bool found_allocator = false;
    for (size_t i = 0; i < sources.size(); i++)
    {
        if (sources[i].allocator != mat.allocator)
            continue;
        src.source_generation = sources[i].source_generation;
        found_allocator = true;
        break;
    }
    if (!found_allocator && get_weight_source_generation(mat.allocator, src.source_generation) < 0)
    {
        NCNN_LOGE("weight src must be finalized before recording");
        return -1;
    }

    // invalidate stale cache entries while retaining their queue ownership
    for (size_t i = 0; i < entries.size(); i++)
    {
        VkWeightCache::weight_entry& entry = entries[i];
        if (entry.src == mat.data && (entry.allocator != src.allocator || entry.source_generation != src.source_generation))
            entry.src = 0;
    }
    for (size_t i = 0; i < host_entries.size(); i++)
    {
        VkWeightCache::weight_entry& entry = host_entries[i];
        if (entry.src == mat.data && (entry.allocator != src.allocator || entry.source_generation != src.source_generation))
            entry.src = 0;
    }
    sources.push_back(src);
    return 0;
}

int VkWeightCache::validate_sources() const
{
    // validate allocators before reading recorded sources
    for (size_t i = 0; i < sources.size(); i++)
    {
        const VkWeightCache::weight_entry& src = sources[i];
        if (!src.source_generation)
            continue;
        bool checked = false;
        for (size_t j = 0; j < i; j++)
        {
            if (sources[j].allocator == src.allocator)
            {
                checked = true;
                break;
            }
        }
        if (checked)
            continue;
        uint64_t source_generation;
        if (get_weight_source_generation(src.allocator, source_generation) < 0 || source_generation != src.source_generation)
        {
            NCNN_LOGE("weight src was cleared or changed while recording");
            return -1;
        }
    }
    return 0;
}

int VkWeightCache::initialize(size_t minimum_device_size, size_t max_source_size, size_t max_file_size)
{
    if (error != 0)
        return error;
    if (executing)
    {
        NCNN_LOGE("weight cache is executing");
        return -1;
    }
    if (max_source_size > (size_t)-1 / 2 || max_file_size > (size_t)-1 / 2)
    {
        NCNN_LOGE("weight cache size overflow");
        return -1;
    }

    size_t device_memory_limit = opt.weight_device_memory_limit;
    size_t host_memory_limit = opt.weight_host_memory_limit;
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
    size_t new_host_size = host_size;
    if (unified)
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
    opt.weight_device_memory_limit = device_memory_limit;
    opt.weight_host_memory_limit = host_memory_limit;
    device_size = new_device_size;
    host_size = new_host_size;
    if (!initialized)
    {
        error = allocate();
        initialized = true;
    }
    if (error)
        return error;
    transfer_queue_family_index = vkdev->info.transfer_queue_family_index();
    if (unified)
        transfer_queue_family_index = vkdev->info.compute_queue_family_index();
    if (!commands_initialized)
    {
        error = create_commands();
        commands_initialized = true;
    }
    return error;
}

int VkWeightCache::binding_source(const buffer_binding& binding, const VkBufferMemory*& src) const
{
    src = binding.src;
    if (!src)
        return 0;
    if (!is_deferred_buffer(src) || binding.view_offset > src->source_size
            || binding.descriptor.range > src->source_size - binding.view_offset
            || src->source_size > (size_t)-1 - alignment
            || (src->storage_type != 2 && src->storage_type != 1)
            || (needs_host_copy(src) && !src->source_ptr)
            || (!needs_host_copy(src) && alignSize(src->source_size, 4) > src->capacity))
    {
        NCNN_LOGE("invalid deferred buffer binding");
        return -1;
    }
    return 0;
}

size_t VkWeightCache::source_size(const VkBufferMemory* src) const
{
    return alignSize(src->source_size, alignment);
}

int VkWeightCache::acquire_queues()
{
    int ret;
    if (error || executing)
        return -1;
    const uint32_t compute_family = vkdev->info.compute_queue_family_index();
    compute_queue = vkdev->acquire_queue(compute_family);
    if (!compute_queue)
        return -1;
    executing = true;
#if NCNN_BENCHMARK
    bytes_read = 0;
    bytes_uploaded = 0;
    start_time = get_current_time();
#endif // NCNN_BENCHMARK
    if (transfer_queue_family_index != compute_family)
        transfer_queue = vkdev->acquire_queue(transfer_queue_family_index);
    else if (!unified)
        transfer_queue = vkdev->try_acquire_queue(compute_family);
    if (!transfer_queue && transfer_queue_family_index == compute_family)
        transfer_queue = compute_queue;
    if (!transfer_queue)
        return finish(-1);

    // carry cached-weight visibility across successive leases of compute queues
    if (handoff_pending)
    {
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submitInfo;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = 0;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &handoff_semaphore;
        submitInfo.pWaitDstStageMask = &stage;
        submitInfo.commandBufferCount = 0;
        submitInfo.pCommandBuffers = 0;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = 0;
        ret = vkResetFences(vkdev->vkdevice(), 1, &handoff_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return finish(-1);
        }
        ret = submit_command(false, submitInfo, handoff_fence);
        if (ret != 0)
            return finish(ret);
        ret = wait_fence(handoff_fence);
        if (ret != 0)
            return finish(ret);
        handoff_pending = false;
    }
    return 0;
}

int VkWeightCache::submit_prefix(VkCommandBuffer command_buffer, VkFence fence)
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

int VkWeightCache::submit_command(bool transfer, const VkSubmitInfo& submitInfo, VkFence fence)
{
    VkQueue queue = compute_queue;
    if (transfer)
        queue = transfer_queue;
    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkQueueSubmit failed %d", ret);
        return -1;
    }
    return 0;
}

int VkWeightCache::plan_batches(std::vector<weight_batch>& batches, std::vector<buffer_range>& initial_releases)
{
    std::vector<VkWeightCache::weight_entry> entries = this->entries;
    for (size_t i = 0; i < entries.size(); i++)
        entries[i].last_batch = 0;
    size_t head = device_head;
    for (size_t i = 0; i < batches.size(); i++)
    {
        weight_batch& batch = batches[i];
        std::vector<VkWeightCache::weight_entry> evicted;
        for (size_t j = 0; j < batch.weights.size(); j++)
        {
            if (find_device_entry(entries, batch.weights[j]) >= 0)
                continue;
            if (place(batch.weights[j], batch.weights, entries, head, evicted) >= 0)
                continue;
            // compact fragmented ranges before uploading this batch
            for (size_t k = 0; k < entries.size(); k++)
                if (entries[k].valid)
                    evicted.push_back(entries[k]);
            entries.clear();
            head = 0;
            for (size_t k = 0; k < batch.weights.size(); k++)
                if (place(batch.weights[k], batch.weights, entries, head, evicted) < 0)
                    return -1;
            break;
        }
        batch.wait_batch = 0;
        for (size_t j = 0; j < evicted.size(); j++)
        {
            const VkWeightCache::weight_entry& entry = evicted[j];
            buffer_range range;
            range.offset = entry.offset;
            range.size = entry.size;
            if (entry.last_batch)
                batches[entry.last_batch - 1].releases.push_back(range);
            else
                initial_releases.push_back(range);
            batch.acquires.push_back(range);
            batch.wait_batch = std::max(batch.wait_batch, entry.last_batch);
        }
        for (size_t j = 0; j < batch.weights.size(); j++)
        {
            VkWeightCache::weight_entry& entry = entries[find_device_entry(entries, batch.weights[j])];
            batch.offsets.push_back(entry.offset);
            batch.missing.push_back(!entry.valid);
            entry.valid = true;
            entry.last_batch = i + 1;
        }
    }
    std::swap(this->entries, entries);
    device_head = head;
    return 0;
}

int VkWeightCache::prepare(const weight_batch& batch, uint64_t serial_base)
{
    // wait for the last reader before overwriting its ranges
    if (batch.wait_batch)
    {
        int ret = wait_compute(serial_base + batch.wait_batch);
        if (ret != 0)
            return ret;
    }
    if (!unified && transfer_queue_family_index != vkdev->info.compute_queue_family_index())
        pending_acquires = batch.acquires;
    for (size_t i = 0; i < batch.weights.size(); i++)
    {
        if (!batch.missing[i])
            continue;

        int ret = upload(batch.weights[i], batch.offsets[i]);
        if (ret != 0)
            return ret;
    }
    current_batch = &batch;
    return 0;
}

int VkWeightCache::release_ranges(const std::vector<buffer_range>& ranges)
{
    if (ranges.empty() || unified || transfer_queue_family_index == vkdev->info.compute_queue_family_index())
        return 0;
    int ret = reset_command_buffer(release_command_buffer);
    if (ret != 0)
        return ret;

    ret = vkResetFences(vkdev->vkdevice(), 1, &release_fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetFences failed %d", ret);
        return -1;
    }
    for (size_t i = 0; i < ranges.size(); i++)
        release_to_transfer(release_command_buffer, ranges[i]);
    ret = vkEndCommandBuffer(release_command_buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &release_command_buffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = 0;
    ret = submit_command(false, submitInfo, release_fence);
    if (ret != 0)
        return ret;
    release_pending = true;
    ret = wait_fence(release_fence);
    if (ret != 0)
        return ret;
    release_pending = false;
    return 0;
}

int VkWeightCache::begin_compute(VkCommandBuffer* command_buffer)
{
    int ret;
    VkWeightCache::compute_slot& c = computes[next_compute];
    if (c.pending)
    {
        ret = wait_compute(c.serial);
        if (ret != 0)
            return ret;
        c.pending = false;
    }
    ret = reset_command_buffer(c.command_buffer);
    if (ret != 0)
        return ret;
    ret = vkResetFences(vkdev->vkdevice(), 1, &c.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetFences failed %d", ret);
        return -1;
    }
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        const VkWeightCache::upload_slot& u = uploads[pending_uploads[i]];
        VkWeightCache::buffer_range range;
        range.offset = u.offset;
        range.size = u.size;
        acquire_for_compute(c.command_buffer, range);
    }
    if (unified)
    {
        for (size_t i = 0; i < current_batch->weights.size(); i++)
        {
            const VkBufferMemory* src = current_batch->weights[i];
            if (!needs_host_copy(src))
                continue;
            // barrier device host-write @ host to shader-read @ compute
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device_buffer->buffer;
            barrier.offset = device_buffer->offset + current_batch->offsets[i];
            barrier.size = source_size(src);
            vkCmdPipelineBarrier(c.command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    *command_buffer = c.command_buffer;
    return next_compute;
}

int VkWeightCache::submit_compute(int slot)
{
    int ret;
    VkWeightCache::compute_slot& c = computes[slot];
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(c.command_buffer);
#endif
    ret = vkEndCommandBuffer(c.command_buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }
    VkSemaphore semaphores[weight_transfer_slots];
    VkPipelineStageFlags stages[weight_transfer_slots];
    for (size_t i = 0; i < pending_uploads.size(); i++)
    {
        semaphores[i] = uploads[pending_uploads[i]].upload_compute_semaphore;
        stages[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (transfer_queue_family_index != vkdev->info.compute_queue_family_index())
            stages[i] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = (uint32_t)pending_uploads.size();
    submitInfo.pWaitSemaphores = semaphores;
    submitInfo.pWaitDstStageMask = stages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &c.command_buffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = 0;
    ret = submit_command(false, submitInfo, c.fence);
    if (ret != 0)
        return ret;
    c.serial = ++compute_serial;
    c.pending = true;
    for (size_t i = 0; i < pending_uploads.size(); i++)
        uploads[pending_uploads[i]].consumer_serial = c.serial;
    pending_uploads.clear();
    current_batch = 0;
    next_compute = (next_compute + 1) % weight_compute_slots;
    return 0;
}

int VkWeightCache::finish(int result)
{
    if (!executing)
    {
        if (result != 0)
            return result;
        return error;
    }
    for (int i = 0; i < weight_compute_slots; i++)
    {
        if (computes[i].pending)
        {
            int ret = wait_fence(computes[i].fence);
            if (ret != 0)
                result = ret;
        }
        computes[i].pending = false;
    }
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        VkWeightCache::upload_slot& u = uploads[i];
        if (u.pending)
        {
            int ret = wait_fence(u.fence);
            if (ret != 0)
                result = ret;
        }
        u.pending = false;
        u.ready_pending = false;
        u.host_active = false;
        u.consumer_serial = 0;
    }
    if (prefix_fence)
    {
        int ret = wait_fence(prefix_fence);
        if (ret != 0)
            result = ret;
    }
    prefix_fence = 0;
    if (release_pending)
    {
        int ret = wait_fence(release_fence);
        if (ret != 0)
            result = ret;
    }
    release_pending = false;
    if (result == 0)
    {
        VkSubmitInfo submitInfo;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = 0;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = 0;
        submitInfo.pWaitDstStageMask = 0;
        submitInfo.commandBufferCount = 0;
        submitInfo.pCommandBuffers = 0;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &handoff_semaphore;
        int ret = vkResetFences(vkdev->vkdevice(), 1, &handoff_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            result = -1;
        }
        if (result == 0)
        {
            result = submit_command(false, submitInfo, handoff_fence);
        }
        if (result == 0)
            result = wait_fence(handoff_fence);
        if (result == 0)
        {
            handoff_pending = true;
        }
    }
    if (transfer_queue && transfer_queue != compute_queue)
        vkdev->reclaim_queue(transfer_queue_family_index, transfer_queue);
    if (compute_queue)
        vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
    compute_queue = 0;
    transfer_queue = 0;
    executing = false;
    completed_compute_serial = compute_serial;
    pending_uploads.clear();
    current_batch = 0;
    if (result != 0)
        error = result;
#if NCNN_BENCHMARK
    NCNN_LOGE("weight cache read=%llu upload=%llu time=%.2fms", (unsigned long long)bytes_read, (unsigned long long)bytes_uploaded, get_current_time() - start_time);
#endif // NCNN_BENCHMARK
    return result;
}

int VkWeightCache::find_device_entry(const std::vector<VkWeightCache::weight_entry>& entries, const VkBufferMemory* src) const
{
    for (size_t i = 0; i < entries.size(); i++)
        if (entries[i].src == src)
            return (int)i;
    return -1;
}

int VkWeightCache::find_host_entry(const VkBufferMemory* src) const
{
    for (size_t i = 0; i < host_entries.size(); i++)
        if (host_entries[i].src == src)
            return (int)i;
    return -1;
}

int VkWeightCache::allocate()
{
    if (!device_size)
        return 0;

    size_t allocation_size;
    device_buffer = create_weight_buffer(vkdev, device_size, 0, unified, opt.weight_device_memory_limit, allocation_size, true);
    if (!device_buffer)
        return -100;

    if (host_size)
    {
        host_buffer = create_weight_buffer(vkdev, host_size, 1, false, opt.weight_host_memory_limit, allocation_size, true);
        if (!host_buffer)
            return -100;
    }

    return 0;
}

int VkWeightCache::create_commands()
{
    int ret;
    VkCommandPoolCreateInfo commandPoolCreateInfo;
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = 0;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = transfer_queue_family_index;
    ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateCommandPool failed %d", ret);
        return -1;
    }
    commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();
    ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateCommandPool failed %d", ret);
        return -1;
    }
    VkFenceCreateInfo fenceCreateInfo;
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = 0;
    fenceCreateInfo.flags = 0;
    VkSemaphoreCreateInfo semaphoreCreateInfo;
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = 0;
    semaphoreCreateInfo.flags = 0;
    if (!handoff_semaphore)
    {
        ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &handoff_semaphore);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return -1;
        }
    }
    ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &handoff_fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateFence failed %d", ret);
        return -1;
    }
    if (!unified && transfer_queue_family_index != vkdev->info.compute_queue_family_index())
    {
        ret = allocate_command_buffer(vkdev->vkdevice(), compute_command_pool, &release_command_buffer);
        if (ret != 0)
            return ret;
        ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &release_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        upload_slot& u = uploads[i];
        ret = allocate_command_buffer(vkdev->vkdevice(), transfer_command_pool, &u.command_buffer);
        if (ret != 0)
            return ret;
        ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &u.fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
        ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &u.upload_compute_semaphore);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return -1;
        }
    }
    for (int i = 0; i < weight_compute_slots; i++)
    {
        ret = allocate_command_buffer(vkdev->vkdevice(), compute_command_pool, &computes[i].command_buffer);
        if (ret != 0)
            return ret;
        ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &computes[i].fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }
    return 0;
}

void VkWeightCache::destroy_commands()
{
    if (!commands_initialized)
        return;
    if (executing)
        finish(error);
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        upload_slot& u = uploads[i];
        vkDestroyFence(vkdev->vkdevice(), u.fence, 0);
        vkDestroySemaphore(vkdev->vkdevice(), u.upload_compute_semaphore, 0);
        memset(&u, 0, sizeof(u));
    }
    for (int i = 0; i < weight_compute_slots; i++)
    {
        compute_slot& c = computes[i];
        vkDestroyFence(vkdev->vkdevice(), c.fence, 0);
        if (c.command_buffer)
            vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &c.command_buffer);
        memset(&c, 0, sizeof(c));
    }
    vkDestroyFence(vkdev->vkdevice(), release_fence, 0);
    release_fence = 0;
    if (release_command_buffer)
        vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &release_command_buffer);
    release_command_buffer = 0;
    release_pending = false;
    vkDestroyFence(vkdev->vkdevice(), handoff_fence, 0);
    handoff_fence = 0;
    if (transfer_command_pool)
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
    compute_command_pool = 0;
    transfer_command_pool = 0;
    prefix_fence = 0;
    next_upload = next_compute = 0;
    compute_serial = completed_compute_serial = 0;
    commands_initialized = false;
    pending_acquires.clear();
    pending_uploads.clear();
    current_batch = 0;
}

int VkWeightCache::upload(const VkBufferMemory* src, size_t offset)
{
    const size_t size = alignSize(src->source_size, alignment);
    if (unified && needs_host_copy(src))
    {
        unsigned char* dst = (unsigned char*)device_buffer->mapped_ptr + device_buffer->offset + offset;
        memcpy(dst, src->source_ptr, src->source_size);
#if NCNN_BENCHMARK
        if (src->storage_type == 2)
            bytes_read += src->source_size;
#endif // NCNN_BENCHMARK
        memset(dst + src->source_size, 0, size - src->source_size);
        int ret = flush_weight_buffer(vkdev, device_buffer, offset, size);
        if (ret != 0)
            return ret;
        return 0;
    }
    buffer_range range;
    range.offset = offset;
    range.size = size;
    if (!needs_host_copy(src))
    {
        int slot = wait_upload_slot();
        if (slot < 0)
            return slot;
        const int ret = upload_chunk(uploads[slot], src, src->offset, offset, alignSize(src->source_size, 4), range, true);
        return ret;
    }
    if (size <= host_size)
    {
        size_t host_offset;
        int ret = cache_host(src, host_offset);
        if (ret != 0)
            return ret;
        int slot = wait_upload_slot();
        if (slot < 0)
            return slot;
        return upload_chunk(uploads[slot], host_buffer, host_buffer->offset + host_offset, offset, alignSize(src->source_size, 4), range, true);
    }
    const size_t chunk_capacity = std::max(alignment, host_size / 2 / alignment * alignment);
    for (size_t file_offset = 0; file_offset < src->source_size;)
    {
        const size_t chunk_size = std::min(chunk_capacity, src->source_size - file_offset);
        const size_t host_range_size = alignSize(chunk_size, alignment);
        size_t host_offset;
        int ret = allocate_host_range(host_range_size, host_offset);
        if (ret != 0)
            return ret;
        int slot = wait_upload_slot();
        if (slot < 0)
            return slot;
        unsigned char* dst = (unsigned char*)host_buffer->mapped_ptr + host_buffer->offset + host_offset;
        memcpy(dst, (const unsigned char*)src->source_ptr + file_offset, chunk_size);
#if NCNN_BENCHMARK
        if (src->storage_type == 2)
            bytes_read += chunk_size;
#endif // NCNN_BENCHMARK
        memset(dst + chunk_size, 0, host_range_size - chunk_size);
        ret = flush_host(host_offset, host_range_size);
        if (ret != 0)
            return ret;
        const bool last = chunk_size == src->source_size - file_offset;
        ret = upload_chunk(uploads[slot], host_buffer, host_buffer->offset + host_offset, offset + file_offset, alignSize(chunk_size, 4), range, last);
        if (ret != 0)
            return ret;
        file_offset += chunk_size;
    }
    return 0;
}

int VkWeightCache::upload_chunk(upload_slot& u, const VkBufferMemory* src, size_t src_offset, size_t offset, size_t size, const buffer_range& range, bool last)
{
    int ret;
    if (!(src->usage_flags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return -1;
    ret = reset_command_buffer(u.command_buffer);
    if (ret != 0)
        return ret;
    ret = vkResetFences(vkdev->vkdevice(), 1, &u.fence);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkResetFences failed %d", ret);
        return -1;
    }
    for (size_t i = 0; i < pending_acquires.size(); i++)
        acquire_for_transfer(u.command_buffer, pending_acquires[i]);
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
    vkCmdPipelineBarrier(u.command_buffer, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
    VkBufferCopy copy;
    copy.size = size;
    copy.srcOffset = src_offset;
    copy.dstOffset = device_buffer->offset + offset;
    vkCmdCopyBuffer(u.command_buffer, src->buffer, device_buffer->buffer, 1, &copy);
    if (last)
        release_to_compute(u.command_buffer, range);
    ret = vkEndCommandBuffer(u.command_buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return -1;
    }
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &u.command_buffer;
    submitInfo.signalSemaphoreCount = 0;
    if (last)
        submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &u.upload_compute_semaphore;
    ret = submit_command(true, submitInfo, u.fence);
    if (ret != 0)
        return ret;
#if NCNN_BENCHMARK
    bytes_uploaded += size;
#endif // NCNN_BENCHMARK
    u.pending = true;
    u.ready_pending = last;
    u.host_active = src == host_buffer;
    u.host_offset = 0;
    u.host_size = 0;
    if (u.host_active)
    {
        u.host_offset = src_offset - host_buffer->offset;
        u.host_size = alignSize(size, alignment);
    }
    u.offset = range.offset;
    u.size = range.size;
    pending_acquires.clear();
    if (last)
        pending_uploads.push_back(next_upload);
    next_upload = (next_upload + 1) % weight_transfer_slots;
    return 0;
}

bool VkWeightCache::is_pinned(const VkBufferMemory* src, const std::vector<const VkBufferMemory*>& pinned)
{
    for (size_t i = 0; i < pinned.size(); i++)
        if (pinned[i] == src)
            return true;
    return false;
}

int VkWeightCache::place(const VkBufferMemory* src, const std::vector<const VkBufferMemory*>& pinned, std::vector<VkWeightCache::weight_entry>& entries, size_t& head,
                         std::vector<VkWeightCache::weight_entry>& evicted)
{
    const VkWeightCache::weight_entry* recorded = find_source(src);
    if (!recorded)
        return -1;
    const size_t size = alignSize(src->source_size, alignment);
    size_t offset = head;
    bool found = false;
    for (int pass = 0; pass < 2 && !found; pass++)
    {
        for (size_t attempt = 0; attempt <= entries.size(); attempt++)
        {
            if (offset > device_size || size > device_size - offset)
                break;
            size_t next_offset = offset;
            for (size_t i = 0; i < entries.size(); i++)
            {
                const VkWeightCache::weight_entry& e = entries[i];
                if (!e.src || !is_pinned(e.src, pinned))
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
    for (size_t i = 0; i < entries.size(); i++)
    {
        VkWeightCache::weight_entry& e = entries[i];
        if ((e.src || e.valid) && offset < e.offset + e.size && e.offset < offset + size)
        {
            if (e.valid)
                evicted.push_back(e);
            e.src = 0;
            e.valid = false;
        }
        if (!e.src && !e.valid)
            slot = (int)i;
    }
    VkWeightCache::weight_entry entry;
    entry.src = src;
    entry.allocator = recorded->allocator;
    entry.source_generation = recorded->source_generation;
    entry.offset = offset;
    entry.size = size;
    entry.last_batch = 0;
    entry.valid = false;
    if (slot == -1)
    {
        slot = (int)entries.size();
        entries.push_back(entry);
    }
    else
    {
        entries[slot] = entry;
    }
    head = offset + size;
    return slot;
}

int VkWeightCache::wait_fence(VkFence fence)
{
    VkResult ret = vkGetFenceStatus(vkdev->vkdevice(), fence);
    if (ret == VK_SUCCESS)
        return 0;
    if (ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetFenceStatus failed %d", ret);
        return -1;
    }
    ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkWaitForFences failed %d", ret);
        return -1;
    }
    return 0;
}

int VkWeightCache::wait_compute(uint64_t serial)
{
    if (serial <= completed_compute_serial)
        return 0;
    for (int i = 0; i < weight_compute_slots; i++)
    {
        if (computes[i].pending && computes[i].serial == serial)
        {
            int ret = wait_fence(computes[i].fence);
            if (ret != 0)
                return ret;
            completed_compute_serial = serial;
            return 0;
        }
    }
    return -1;
}

int VkWeightCache::wait_upload_slot()
{
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        upload_slot& u = uploads[next_upload];
        // retain semaphores until the current batch consumes them
        if (!u.ready_pending || u.consumer_serial)
        {
            int ret = retire_upload(u);
            if (ret != 0)
                return ret;
            return next_upload;
        }
        next_upload = (next_upload + 1) % weight_transfer_slots;
    }
    return -1;
}

int VkWeightCache::retire_upload(upload_slot& u)
{
    if (u.pending)
    {
        if (u.ready_pending)
        {
            if (!u.consumer_serial)
                return -1;
            int ret = wait_compute(u.consumer_serial);
            if (ret != 0)
                return ret;
        }
        int ret = wait_fence(u.fence);
        if (ret != 0)
            return ret;
    }
    u.pending = false;
    u.ready_pending = false;
    u.host_active = false;
    u.consumer_serial = 0;
    return 0;
}

int VkWeightCache::allocate_host_range(size_t size, size_t& offset)
{
    if (size > host_size)
        return -1;
    if (size > host_size - host_head)
        host_head = 0;
    offset = host_head;
    for (int i = 0; i < weight_transfer_slots; i++)
    {
        upload_slot& u = uploads[i];
        if (u.host_active && offset < u.host_offset + u.host_size && u.host_offset < offset + size)
        {
            int ret = wait_fence(u.fence);
            if (ret != 0)
                return ret;
            u.host_active = false;
        }
    }
    for (size_t i = 0; i < host_entries.size(); i++)
    {
        VkWeightCache::weight_entry& entry = host_entries[i];
        if (entry.src && offset < entry.offset + entry.size && entry.offset < offset + size)
        {
            entry.src = 0;
        }
    }
    host_head += size;
    return 0;
}

int VkWeightCache::cache_host(const VkBufferMemory* src, size_t& offset)
{
    const int cached = find_host_entry(src);
    if (cached >= 0)
    {
        offset = host_entries[cached].offset;
        return 0;
    }
    const VkWeightCache::weight_entry* recorded = find_source(src);
    if (!recorded)
        return -1;
    const size_t size = alignSize(src->source_size, alignment);
    int ret = allocate_host_range(size, offset);
    if (ret != 0)
        return ret;
    unsigned char* dst = (unsigned char*)host_buffer->mapped_ptr + host_buffer->offset + offset;
    memcpy(dst, src->source_ptr, src->source_size);
#if NCNN_BENCHMARK
    if (src->storage_type == 2)
        bytes_read += src->source_size;
#endif // NCNN_BENCHMARK
    memset(dst + src->source_size, 0, size - src->source_size);
    ret = flush_host(offset, size);
    if (ret != 0)
        return ret;
    int slot = -1;
    for (size_t i = 0; i < host_entries.size(); i++)
        if (!host_entries[i].src)
            slot = (int)i;
    VkWeightCache::weight_entry entry;
    entry.src = src;
    entry.allocator = recorded->allocator;
    entry.source_generation = recorded->source_generation;
    entry.offset = offset;
    entry.size = size;
    entry.last_batch = 0;
    entry.valid = true;
    if (slot < 0)
    {
        slot = (int)host_entries.size();
        host_entries.push_back(entry);
    }
    else
        host_entries[slot] = entry;
    return 0;
}

int VkWeightCache::flush_host(size_t offset, size_t size)
{
    return flush_weight_buffer(vkdev, host_buffer, offset, size);
}

void VkWeightCache::release_to_compute(VkCommandBuffer command_buffer, const buffer_range& range)
{
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
    barrier.buffer = device_buffer->buffer;
    barrier.offset = device_buffer->offset + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCache::acquire_for_compute(VkCommandBuffer command_buffer, const buffer_range& range)
{
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
    barrier.buffer = device_buffer->buffer;
    barrier.offset = device_buffer->offset + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCache::release_to_transfer(VkCommandBuffer command_buffer, const buffer_range& range)
{
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
    barrier.buffer = device_buffer->buffer;
    barrier.offset = device_buffer->offset + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

void VkWeightCache::acquire_for_transfer(VkCommandBuffer command_buffer, const buffer_range& range)
{
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
    barrier.buffer = device_buffer->buffer;
    barrier.offset = device_buffer->offset + range.offset;
    barrier.size = range.size;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
}

class VkComputePrivate
{
public:
    VkComputePrivate(const VulkanDevice* _vkdev, const Option& opt);
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

    VkWeightCache* weight_cache;
    bool weight_deferred;
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
        buffer_binding binding;
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

    int replay_record(const record& r, VkCommandBuffer command_buffer);
    int record_sources(const record& r, std::vector<const VkBufferMemory*>& weights);

    std::vector<record> deferred_records;

    uint64_t pending_dispatch_total;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
    bool query_results_ready;
#endif // NCNN_BENCHMARK
};

VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev, const Option& opt)
    : vkdev(_vkdev)
{
    weight_cache = new VkWeightCache(vkdev, opt);
    weight_deferred = false;
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

    init();
}

VkComputePrivate::~VkComputePrivate()
{
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

int VkComputePrivate::record_sources(const record& r, std::vector<const VkBufferMemory*>& weights)
{
    if (r.type != record::TYPE_pipeline && r.type != record::TYPE_copy_buffer && r.type != record::TYPE_copy_buffer_to_image)
        return 0;

    const size_t count = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers.size() : 1;
    for (size_t i = 0; i < count; i++)
    {
        const buffer_binding& binding = r.type == record::TYPE_pipeline ? pipelines[r.pipeline_index].buffers[i] : r.binding;
        const VkBufferMemory* src;
        int ret = weight_cache->binding_source(binding, src);
        if (ret != 0)
            return ret;
        if (!src)
            continue;
        bool found = false;
        for (size_t j = 0; j < weights.size(); j++)
            if (weights[j] == src)
                found = true;
        if (!found)
            weights.push_back(src);
    }
    return 0;
}

int VkComputePrivate::submit_managed()
{
    int ret = weight_cache->validate_sources();
    if (ret != 0)
        return ret;
    size_t minimum_device_size = 0;
    size_t max_source_size = 0;
    size_t max_file_size = 0;
    std::vector<const VkBufferMemory*> dependencies;
    std::vector<size_t> dependency_offsets;
    std::vector<const VkBufferMemory*> sources;
    dependency_offsets.push_back(0);
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        sources.clear();
        ret = record_sources(deferred_records[i], sources);
        if (ret != 0)
            return ret;
        if (sources.size() > (size_t)weight_transfer_slots)
        {
            NCNN_LOGE("too many simultaneous weight bindings: %llu", (unsigned long long)sources.size());
            return -1;
        }
        size_t total = 0;
        for (size_t j = 0; j < sources.size(); j++)
        {
            const size_t size = weight_cache->source_size(sources[j]);
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
    std::vector<VkWeightCache::weight_batch> batches;
    VkWeightCache::weight_batch batch;
    batch.end = 0;
    batch.wait_batch = 0;
    batch.begin = 0;
    size_t batch_size = 0;
    const size_t capacity = weight_cache->device_size;
    for (size_t i = 0; i < deferred_records.size(); i++)
    {
        sources.clear();
        size_t added_size = 0;
        for (size_t j = dependency_offsets[i]; j < dependency_offsets[i + 1]; j++)
        {
            const VkBufferMemory* src = dependencies[j];
            bool found = false;
            for (size_t k = 0; k < batch.weights.size(); k++)
                found |= batch.weights[k] == src;
            if (found)
                continue;
            const size_t size = weight_cache->source_size(src);
            if (size > (size_t)-1 - added_size)
            {
                NCNN_LOGE("weight batch size overflow");
                return -1;
            }
            added_size += size;
            sources.push_back(src);
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
                batch_size += weight_cache->source_size(dependencies[j]);
            }
        }
        else
        {
            for (size_t j = 0; j < sources.size(); j++)
                batch.weights.push_back(sources[j]);
            batch_size += added_size;
        }
    }
    batch.end = deferred_records.size();
    batches.push_back(batch);

    std::vector<VkWeightCache::buffer_range> initial_releases;
    ret = weight_cache->plan_batches(batches, initial_releases);
    if (ret != 0)
        return ret;
    ret = weight_cache->acquire_queues();
    if (ret != 0)
    {
        weight_cache->error = ret;
        return ret;
    }
    const uint64_t serial_base = weight_cache->compute_serial;
    ret = weight_cache->release_ranges(initial_releases);
    // submit ordinary commands recorded before the first deferred weight
    if (ret == 0 && vkdev->info.support_VK_KHR_push_descriptor())
    {
        ret = end_command_buffer();
        if (ret == 0)
            ret = weight_cache->submit_prefix(compute_command_buffer, compute_command_fence);
    }
    for (size_t i = 0; i < batches.size() && ret == 0; i++)
    {
        ret = weight_cache->prepare(batches[i], serial_base);
        if (ret != 0)
            break;
        VkCommandBuffer command_buffer;
        int slot = weight_cache->begin_compute(&command_buffer);
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
            if (!weight_cache->unified && weight_cache->transfer_queue_family_index != vkdev->info.compute_queue_family_index())
                for (size_t j = 0; j < batches[i].releases.size(); j++)
                    weight_cache->release_to_transfer(command_buffer, batches[i].releases[j]);
            ret = weight_cache->submit_compute(slot);
        }
    }
    ret = weight_cache->finish(ret);
    return ret;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev, Option()))
{
}

VkCompute::VkCompute(const VulkanDevice* _vkdev, const Option& opt)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev, opt))
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
        d->record_error = -100;
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
        d->record_error = -100;
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
        d->record_error = -100;
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
        d->record_error = -100;
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
        d->record_error = -100;
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
        d->record_error = -100;
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
        d->record_error = -100;
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
            r.binding.src = 0;
            if (is_deferred_buffer(src.data))
                r.binding.src = src.data;
            r.binding.view_offset = src.offset;
            r.binding.descriptor.buffer = src.buffer();
            r.binding.descriptor.offset = src.buffer_offset();
            r.binding.descriptor.range = src.buffer_capacity();
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
        d->record_error = -100;
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
    if (d->record_error)
        return;
    if (src.empty())
    {
        d->record_error = -100;
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
            r.binding.src = 0;
            if (is_deferred_buffer(src.data))
                r.binding.src = src.data;
            r.binding.view_offset = src.offset;
            r.binding.descriptor.buffer = src.buffer();
            r.binding.descriptor.offset = src.buffer_offset();
            r.binding.descriptor.range = src.buffer_capacity();
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
        d->record_error = -100;
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
            b.src = 0;
            if (is_deferred_buffer(binding.data))
                b.src = binding.data;
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
    if (!d->deferred())
    {
        if (d->query_pool)
            vkCmdWriteTimestamp(d->compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->query_pool, query);
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

    // record bind pipeline
    {
        if (!d->deferred())
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
        if (!weight_cache || !r.binding.src)
        {
            vkCmdCopyBuffer(command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight_binding(r.binding, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferCopy> regions(r.copy_buffer.region_count);
        for (uint32_t i = 0; i < r.copy_buffer.region_count; i++)
        {
            regions[i] = r.copy_buffer.regions[i];
            regions[i].srcOffset = descriptor.offset + regions[i].srcOffset - r.binding.descriptor.offset;
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
        if (!weight_cache || !r.binding.src)
        {
            vkCmdCopyBufferToImage(command_buffer, r.copy_buffer_to_image.src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, r.copy_buffer_to_image.regions);
            break;
        }
        VkDescriptorBufferInfo descriptor;
        int ret = resolve_weight_binding(r.binding, descriptor);
        if (ret != 0)
            return ret;
        std::vector<VkBufferImageCopy> regions(r.copy_buffer_to_image.region_count);
        for (uint32_t i = 0; i < r.copy_buffer_to_image.region_count; i++)
        {
            regions[i] = r.copy_buffer_to_image.regions[i];
            regions[i].bufferOffset = descriptor.offset + regions[i].bufferOffset - r.binding.descriptor.offset;
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
            vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, r.write_timestamp.query);
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
                    has_deferred_weights |= buffers[j].src != 0;
            }
            else if (r.type == VkComputePrivate::record::TYPE_copy_buffer || r.type == VkComputePrivate::record::TYPE_copy_buffer_to_image)
            {
                has_deferred_weights = r.binding.src != 0;
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
                d->begin_command_buffer();

#if NCNN_BENCHMARK
                if (d->query_pool)
                    vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
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
    if (d->weight_cache->executing)
        return -1;
#if NCNN_BENCHMARK
    d->query_results_ready = false;
#endif // NCNN_BENCHMARK
    d->record_error = 0;

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
    if (d->weight_cache->error)
    {
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
            vkCmdResetQueryPool(d->compute_command_buffer, d->query_pool, 0, d->query_count);
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkCompute::clear_weight_cache()
{
    if (d->weight_cache->executing || !d->deferred_records.empty() || d->pending_dispatch_total)
    {
        NCNN_LOGE("cannot clear a weight cache while recording or executing");
        return -1;
    }
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

int VkComputePrivate::resolve_weight_binding(const buffer_binding& binding, VkDescriptorBufferInfo& descriptor) const
{
    const VkBufferMemory* src = binding.src;
    descriptor = binding.descriptor;
    if (!src)
        return 0;
    for (size_t i = 0; i < weight_cache->current_batch->weights.size(); i++)
    {
        if (weight_cache->current_batch->weights[i] != src)
            continue;
        descriptor.buffer = weight_cache->device_buffer->buffer;
        descriptor.offset = weight_cache->device_buffer->offset + weight_cache->current_batch->offsets[i] + binding.view_offset;
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

    std::vector<VkMat> upload_staging_buffers;
};

VkTransferPrivate::VkTransferPrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    pending_upload_total = 0;
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
    finish_upload(upload_error);

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

    if (dst.data->storage_type == 2)
    {
        const size_t dst_offset = dst.offset;
        if (!(dst.data->access_flags & VK_ACCESS_HOST_WRITE_BIT) || !dst.data->source_ptr || dst_offset > dst.data->source_size
                || dst.buffer_capacity() > dst.data->source_size - dst_offset)
        {
            NCNN_LOGE("disk weight requires a writable source range during upload");
            d->upload_error = -1;
            return;
        }
        unsigned char* ptr = (unsigned char*)dst.data->source_ptr;
        for (int b = 0; b < src_flattened.n; b++)
        {
            const Mat src_b = src_flattened.batch(b);
            const size_t offset = dst.batch(b).offset;
            const size_t size = src_b.total() * src_b.elemsize;
            const size_t end = b + 1 == src_flattened.n ? dst_offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            if (offset > end || end > dst.data->source_size || size > end - offset)
            {
                d->upload_error = -1;
                return;
            }
            memcpy(ptr + offset, src_b.data, size);
            memset(ptr + offset + size, 0, end - offset - size);
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

        d->record_host_upload(*dst.data);

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

    VkBufferMemory src_buffer = *dst_staging.data;
    src_buffer.offset = dst_staging.buffer_offset();
    VkBufferMemory dst_buffer = *dst.data;
    dst_buffer.offset = dst.buffer_offset();
    d->record_copy(src_buffer, dst_buffer, std::min(dst_staging.buffer_capacity(), dst.buffer_capacity()));

    // mark device shader-readwrite @ compute
    dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
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

void VkTransfer::record_upload(VkWeightAllocator* weights, const Option& opt)
{
    if (d->upload_error != 0)
        return;

    for (int storage_type = 0; storage_type < 2; storage_type++)
    {
        std::vector<VkBufferMemory> buffers;
        int ret = get_weight_uploads(weights, storage_type, buffers);
        if (ret != 0)
        {
            d->upload_error = ret;
            return;
        }

        VkBufferMemory* staging = 0;
        for (size_t i = 0; i < buffers.size(); i++)
        {
            const VkBufferMemory& buffer = buffers[i];
            if (buffer.mapped_ptr)
            {
                unsigned char* ptr = (unsigned char*)buffer.mapped_ptr + buffer.offset;
                memcpy(ptr, buffer.source_ptr, buffer.source_size);
                memset(ptr + buffer.source_size, 0, buffer.capacity - buffer.source_size);
                ret = flush_weight_buffer(vkdev, &buffer, 0, buffer.capacity);
                if (ret != 0)
                    break;
                if (storage_type == 0)
                {
                    d->record_host_upload(buffer);
                    d->pending_upload_total += buffer.capacity;
                }
                continue;
            }

            if (!staging)
            {
                size_t capacity = std::min(buffer.capacity, (size_t)8 * 1024 * 1024);
                ret = fit_weight_capacity(vkdev, opt.weight_host_memory_limit, true, true, capacity);
                if (ret != 0)
                    break;
                size_t allocation_size;
                staging = create_weight_buffer(vkdev, capacity, 1, false, opt.weight_host_memory_limit, allocation_size, true);
                if (!staging)
                {
                    ret = -100;
                    break;
                }
            }

            for (size_t offset = 0; offset < buffer.source_size;)
            {
                size_t size = std::min(staging->capacity, buffer.source_size - offset);
                size_t copy_size = alignSize(size, 4);
                memcpy(staging->mapped_ptr, (const unsigned char*)buffer.source_ptr + offset, size);
                memset((unsigned char*)staging->mapped_ptr + size, 0, copy_size - size);
                ret = flush_weight_buffer(vkdev, staging, 0, copy_size);
                if (ret != 0)
                    break;

                VkBufferMemory dst = buffer;
                dst.offset += offset;
                d->record_copy(*staging, dst, copy_size);
                d->pending_upload_total += copy_size;
                ret = submit_and_wait();
                if (ret != 0)
                    break;
                ret = reset();
                if (ret != 0)
                    break;
                offset += size;
            }
            if (ret != 0)
                break;
        }
        destroy_weight_buffer(vkdev, staging);
        if (ret != 0)
        {
            d->upload_error = ret;
            return;
        }
    }
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
    if (d->upload_error)
        return d->upload_error;
    if (d->pending_upload_total == 0)
        return 0;

    //     NCNN_LOGE("submit_and_wait");

    // end command buffer
    {
        int ret = d->end_command_buffer();
        if (ret != 0)
            return d->finish_upload(ret);
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
    if (compute_queue == 0)
    {
        NCNN_LOGE("out of compute queue");
        return d->finish_upload(-1);
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
                return d->finish_upload(-1);
            }
            d->compute_pending = true;
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index());
        if (transfer_queue == 0)
        {
            NCNN_LOGE("out of transfer queue");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
            return d->finish_upload(-1);
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
                return d->finish_upload(-1);
            }
            d->upload_pending = true;
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
                return d->finish_upload(-1);
            }
            d->compute_pending = true;
        }

        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);

    return d->finish_upload();
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

uint64_t VkTransfer::pending_upload_total() const
{
    return d->pending_upload_total;
}

} // namespace ncnn

#endif // NCNN_VULKAN
