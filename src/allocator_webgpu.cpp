// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "allocator.h"

#if NCNN_WEBGPU

#include <algorithm>
#include <utility>
#include <vector>

#include "gpu.h"

namespace ncnn {

static uint64_t g_webgpu_buffer_block_id = 0;


static size_t webgpu_buffer_alignment(const VulkanDevice* vkdev)
{
    return std::max((size_t)vkdev->info.buffer_offset_alignment(), (size_t)4);
}

static WebGpuBufferBlock* create_webgpu_buffer_block(const VulkanDevice* vkdev, size_t size)
{
    if (!vkdev || !vkdev->is_valid() || size == 0)
        return 0;

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    descriptor.size = std::max(alignSize(size, 4), (size_t)4);

    WGPUBuffer buffer = wgpuDeviceCreateBuffer(vkdev->wgpu_device(), &descriptor);
    if (!buffer)
    {
        NCNN_LOGE("WebGPU buffer allocation failed size=%llu", (unsigned long long)descriptor.size);
        return 0;
    }

    WebGpuBufferBlock* block = new WebGpuBufferBlock;
    block->buffer = buffer;
    block->size = descriptor.size;
    block->id = ++g_webgpu_buffer_block_id;
    block->live_allocation_count = 0;
    block->in_flight_refcount = 0;
    return block;
}

static void release_webgpu_buffer_block(WebGpuBufferBlock* block)
{
    if (!block)
        return;

    if (block->live_allocation_count != 0 || block->in_flight_refcount != 0)
    {
        NCNN_LOGE("WebGPU buffer block %llu released while live=%u in-flight=%u",
                  (unsigned long long)block->id, block->live_allocation_count, block->in_flight_refcount);
        return;
    }

    if (block->buffer)
        wgpuBufferRelease(block->buffer);
    delete block;
}

static VkBufferMemory* create_webgpu_allocation(WebGpuBufferBlock* block, size_t offset, size_t capacity, bool host_shadow)
{
    if (!block || !block->buffer || capacity == 0 || offset + capacity > block->size)
        return 0;

    void* mapped_ptr = 0;
    if (host_shadow)
    {
        mapped_ptr = ncnn::fastMalloc(capacity);
        if (!mapped_ptr)
            return 0;
        memset(mapped_ptr, 0, capacity);
    }

    VkBufferMemory* memory = new VkBufferMemory;
    memory->block = block;
    memory->buffer = block->buffer;
    memory->offset = offset;
    memory->capacity = capacity;
    memory->mapped_ptr = mapped_ptr;
    memory->host_shadow = host_shadow;
    memory->host_shadow_dirty = false;
    memory->in_flight_refcount = 0;
    memory->pending_free = false;
    memory->refcount = 0;
    block->live_allocation_count++;
    return memory;
}

static void release_webgpu_allocation(VkBufferMemory* memory)
{
    if (!memory)
        return;

    if (memory->mapped_ptr)
        ncnn::fastFree(memory->mapped_ptr);
    memory->block->live_allocation_count--;
    delete memory;
}

static void merge_webgpu_budget(std::vector<std::pair<size_t, size_t> >& budgets, size_t offset, size_t size)
{
    budgets.push_back(std::make_pair(offset, size));
    std::sort(budgets.begin(), budgets.end());

    std::vector<std::pair<size_t, size_t> > merged;
    for (size_t i = 0; i < budgets.size(); i++)
    {
        if (!merged.empty() && merged.back().first + merged.back().second == budgets[i].first)
            merged.back().second += budgets[i].second;
        else
            merged.push_back(budgets[i]);
    }
    budgets.swap(merged);
}

static int queue_write_webgpu_shadow(const VulkanDevice* vkdev, VkBufferMemory* memory)
{
    if (!vkdev || !memory || !memory->buffer || !memory->mapped_ptr)
        return -1;

    const size_t write_chunk_size = 16 * 1024 * 1024;
    for (size_t offset = 0; offset < memory->capacity; offset += write_chunk_size)
    {
        const size_t size = std::min(write_chunk_size, memory->capacity - offset);
        wgpuQueueWriteBuffer(vkdev->wgpu_queue(), memory->buffer, memory->offset + offset, (const unsigned char*)memory->mapped_ptr + offset, size);
    }
    return 0;
}

VkAllocator::VkAllocator(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    buffer_memory_type_index = 0;
    image_memory_type_index = 0;
    reserved_type_index = 0;
    mappable = false;
    coherent = false;
}

VkAllocator::~VkAllocator()
{
}

void VkAllocator::clear()
{
}

int VkAllocator::flush(VkBufferMemory* memory)
{
    if (!memory || !memory->host_shadow || !memory->host_shadow_dirty)
        return 0;

    if (queue_write_webgpu_shadow(vkdev, memory) != 0)
        return -1;

    memory->host_shadow_dirty = false;
    return 0;
}

int VkAllocator::invalidate(VkBufferMemory*)
{
    return -1;
}

int acquire_webgpu_allocation_in_flight(VkAllocator*, VkBufferMemory* memory)
{
    if (!memory || !memory->block || memory->pending_free)
        return -1;

    memory->in_flight_refcount++;
    memory->block->in_flight_refcount++;
    return 0;
}

void release_webgpu_allocation_in_flight(VkAllocator* allocator, VkBufferMemory* memory)
{
    if (!memory || !memory->block || memory->in_flight_refcount == 0 || memory->block->in_flight_refcount == 0)
    {
        NCNN_LOGE("WebGPU allocation in-flight reference underflow");
        return;
    }

    memory->in_flight_refcount--;
    memory->block->in_flight_refcount--;
    if (memory->in_flight_refcount == 0 && memory->pending_free)
        allocator->fastFree(memory);
}

class VkBlobAllocatorPrivate
{
public:
    size_t block_size;
    size_t alignment;
    std::vector<WebGpuBufferBlock*> blocks;
    std::vector<std::vector<std::pair<size_t, size_t> > > budgets;
    std::vector<VkBufferMemory*> allocations;
};

VkBlobAllocator::VkBlobAllocator(const VulkanDevice* vkdev, size_t preferred_block_size)
    : VkAllocator(vkdev), d(new VkBlobAllocatorPrivate)
{
    d->alignment = webgpu_buffer_alignment(vkdev);
    d->block_size = alignSize(preferred_block_size, d->alignment);
}

VkBlobAllocator::~VkBlobAllocator()
{
    clear();
    delete d;
}

void VkBlobAllocator::clear()
{
    for (size_t i = 0; i < d->allocations.size(); i++)
        release_webgpu_allocation(d->allocations[i]);
    d->allocations.clear();

    for (size_t i = 0; i < d->blocks.size(); i++)
        release_webgpu_buffer_block(d->blocks[i]);
    d->blocks.clear();
    d->budgets.clear();
}

VkBufferMemory* VkBlobAllocator::fastMalloc(size_t size)
{
    if (size == 0)
        return 0;

    const size_t aligned_size = alignSize(std::max(size, (size_t)4), d->alignment);
    for (size_t i = 0; i < d->blocks.size(); i++)
    {
        for (size_t j = 0; j < d->budgets[i].size(); j++)
        {
            if (d->budgets[i][j].second < aligned_size)
                continue;

            const size_t offset = d->budgets[i][j].first;
            VkBufferMemory* memory = create_webgpu_allocation(d->blocks[i], offset, aligned_size, false);
            if (!memory)
            {
                NCNN_LOGE("WebGPU blob suballocation failed size=%zu block=%llu", aligned_size, (unsigned long long)d->blocks[i]->id);
                return 0;
            }

            d->budgets[i][j].first += aligned_size;
            d->budgets[i][j].second -= aligned_size;
            if (d->budgets[i][j].second == 0)
                d->budgets[i].erase(d->budgets[i].begin() + j);
            d->allocations.push_back(memory);
            return memory;
        }
    }

    const size_t block_size = std::max(d->block_size, aligned_size);
    WebGpuBufferBlock* block = create_webgpu_buffer_block(vkdev, block_size);
    if (!block)
    {
        NCNN_LOGE("WebGPU blob block allocation failed size=%zu", block_size);
        return 0;
    }

    VkBufferMemory* memory = create_webgpu_allocation(block, 0, aligned_size, false);
    if (!memory)
    {
        NCNN_LOGE("WebGPU blob allocation failed size=%zu block=%llu", aligned_size, (unsigned long long)block->id);
        release_webgpu_buffer_block(block);
        return 0;
    }

    d->blocks.push_back(block);
    std::vector<std::pair<size_t, size_t> > budget;
    if (block_size > aligned_size)
        budget.push_back(std::make_pair(aligned_size, block_size - aligned_size));
    d->budgets.push_back(budget);
    d->allocations.push_back(memory);
    return memory;
}

void VkBlobAllocator::fastFree(VkBufferMemory* memory)
{
    if (!memory)
        return;
    if (memory->in_flight_refcount != 0)
    {
        memory->pending_free = true;
        return;
    }

    std::vector<VkBufferMemory*>::iterator allocation_it = std::find(d->allocations.begin(), d->allocations.end(), memory);
    std::vector<WebGpuBufferBlock*>::iterator block_it = std::find(d->blocks.begin(), d->blocks.end(), memory->block);
    if (allocation_it == d->allocations.end() || block_it == d->blocks.end())
    {
        NCNN_LOGE("WebGPU blob allocator received a wild allocation");
        return;
    }

    const size_t block_index = block_it - d->blocks.begin();
    merge_webgpu_budget(d->budgets[block_index], memory->offset, memory->capacity);
    d->allocations.erase(allocation_it);
    release_webgpu_allocation(memory);
}

VkImageMemory* VkBlobAllocator::fastMalloc(int, int, int, size_t, int)
{
    return 0;
}

void VkBlobAllocator::fastFree(VkImageMemory*)
{
}

class VkWeightAllocatorPrivate
{
public:
    size_t block_size;
    size_t alignment;
    std::vector<WebGpuBufferBlock*> blocks;
    std::vector<size_t> free_spaces;
    std::vector<VkBufferMemory*> allocations;
};

VkWeightAllocator::VkWeightAllocator(const VulkanDevice* vkdev, bool, size_t preferred_block_size)
    : VkAllocator(vkdev), d(new VkWeightAllocatorPrivate)
{
    d->alignment = webgpu_buffer_alignment(vkdev);
    d->block_size = alignSize(preferred_block_size, d->alignment);
}

VkWeightAllocator::~VkWeightAllocator()
{
    clear();
    delete d;
}

void VkWeightAllocator::clear()
{
    for (size_t i = 0; i < d->allocations.size(); i++)
        release_webgpu_allocation(d->allocations[i]);
    d->allocations.clear();

    for (size_t i = 0; i < d->blocks.size(); i++)
        release_webgpu_buffer_block(d->blocks[i]);
    d->blocks.clear();
    d->free_spaces.clear();
}

VkBufferMemory* VkWeightAllocator::fastMalloc(size_t size)
{
    if (size == 0)
        return 0;

    const size_t aligned_size = alignSize(std::max(size, (size_t)4), d->alignment);
    for (size_t i = 0; i < d->blocks.size(); i++)
    {
        if (d->free_spaces[i] < aligned_size)
            continue;

        const size_t offset = d->blocks[i]->size - d->free_spaces[i];
        VkBufferMemory* memory = create_webgpu_allocation(d->blocks[i], offset, aligned_size, false);
        if (!memory)
            return 0;

        d->free_spaces[i] -= aligned_size;
        d->allocations.push_back(memory);
        return memory;
    }

    const size_t block_size = std::max(d->block_size, aligned_size);
    WebGpuBufferBlock* block = create_webgpu_buffer_block(vkdev, block_size);
    if (!block)
        return 0;

    VkBufferMemory* memory = create_webgpu_allocation(block, 0, aligned_size, false);
    if (!memory)
    {
        release_webgpu_buffer_block(block);
        return 0;
    }

    d->blocks.push_back(block);
    d->free_spaces.push_back(block_size - aligned_size);
    d->allocations.push_back(memory);
    return memory;
}

void VkWeightAllocator::fastFree(VkBufferMemory* memory)
{
    if (!memory)
        return;
    if (memory->in_flight_refcount != 0)
    {
        memory->pending_free = true;
        return;
    }

    std::vector<VkBufferMemory*>::iterator it = std::find(d->allocations.begin(), d->allocations.end(), memory);
    if (it == d->allocations.end())
    {
        NCNN_LOGE("WebGPU weight allocator received a wild allocation");
        return;
    }

    d->allocations.erase(it);
    release_webgpu_allocation(memory);
}

VkImageMemory* VkWeightAllocator::fastMalloc(int, int, int, size_t, int)
{
    return 0;
}

void VkWeightAllocator::fastFree(VkImageMemory*)
{
}

class VkStagingAllocatorPrivate
{
public:
    std::vector<WebGpuBufferBlock*> blocks;
    std::vector<VkBufferMemory*> allocations;
};

static VkBufferMemory* allocate_webgpu_staging(const VulkanDevice* vkdev, std::vector<WebGpuBufferBlock*>& blocks, std::vector<VkBufferMemory*>& allocations, size_t size)
{
    if (size == 0)
        return 0;

    const size_t capacity = alignSize(std::max(size, (size_t)4), 4);
    WebGpuBufferBlock* block = create_webgpu_buffer_block(vkdev, capacity);
    if (!block)
        return 0;

    VkBufferMemory* memory = create_webgpu_allocation(block, 0, capacity, true);
    if (!memory)
    {
        release_webgpu_buffer_block(block);
        return 0;
    }

    blocks.push_back(block);
    allocations.push_back(memory);
    return memory;
}

static void free_webgpu_staging(VkBufferMemory* memory, std::vector<WebGpuBufferBlock*>& blocks, std::vector<VkBufferMemory*>& allocations)
{
    if (!memory)
        return;
    if (memory->in_flight_refcount != 0)
    {
        memory->pending_free = true;
        return;
    }

    std::vector<VkBufferMemory*>::iterator allocation_it = std::find(allocations.begin(), allocations.end(), memory);
    std::vector<WebGpuBufferBlock*>::iterator block_it = std::find(blocks.begin(), blocks.end(), memory->block);
    if (allocation_it == allocations.end() || block_it == blocks.end())
    {
        NCNN_LOGE("WebGPU staging allocator received a wild allocation");
        return;
    }

    WebGpuBufferBlock* block = *block_it;
    allocations.erase(allocation_it);
    blocks.erase(block_it);
    release_webgpu_allocation(memory);
    release_webgpu_buffer_block(block);
}

static void clear_webgpu_staging(std::vector<WebGpuBufferBlock*>& blocks, std::vector<VkBufferMemory*>& allocations)
{
    for (size_t i = 0; i < allocations.size(); i++)
        release_webgpu_allocation(allocations[i]);
    allocations.clear();

    for (size_t i = 0; i < blocks.size(); i++)
        release_webgpu_buffer_block(blocks[i]);
    blocks.clear();
}

VkStagingAllocator::VkStagingAllocator(const VulkanDevice* vkdev)
    : VkAllocator(vkdev), d(new VkStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;
}

VkStagingAllocator::~VkStagingAllocator()
{
    clear();
    delete d;
}

void VkStagingAllocator::set_size_compare_ratio(float)
{
}

void VkStagingAllocator::clear()
{
    clear_webgpu_staging(d->blocks, d->allocations);
}

VkBufferMemory* VkStagingAllocator::fastMalloc(size_t size)
{
    return allocate_webgpu_staging(vkdev, d->blocks, d->allocations, size);
}

void VkStagingAllocator::fastFree(VkBufferMemory* memory)
{
    free_webgpu_staging(memory, d->blocks, d->allocations);
}

VkImageMemory* VkStagingAllocator::fastMalloc(int, int, int, size_t, int)
{
    return 0;
}

void VkStagingAllocator::fastFree(VkImageMemory*)
{
}

class VkWeightStagingAllocatorPrivate
{
public:
    std::vector<WebGpuBufferBlock*> blocks;
    std::vector<VkBufferMemory*> allocations;
};

VkWeightStagingAllocator::VkWeightStagingAllocator(const VulkanDevice* vkdev)
    : VkAllocator(vkdev), d(new VkWeightStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;
}

VkWeightStagingAllocator::~VkWeightStagingAllocator()
{
    clear_webgpu_staging(d->blocks, d->allocations);
    delete d;
}

VkBufferMemory* VkWeightStagingAllocator::fastMalloc(size_t size)
{
    return allocate_webgpu_staging(vkdev, d->blocks, d->allocations, size);
}

void VkWeightStagingAllocator::fastFree(VkBufferMemory* memory)
{
    free_webgpu_staging(memory, d->blocks, d->allocations);
}

VkImageMemory* VkWeightStagingAllocator::fastMalloc(int, int, int, size_t, int)
{
    return 0;
}

void VkWeightStagingAllocator::fastFree(VkImageMemory*)
{
}


} // namespace ncnn

#endif // NCNN_WEBGPU
