// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "kvcache_storage.h"

#include <limits.h>
#include <string.h>

#if NCNN_VULKAN
#include "command.h"
#endif

namespace ncnn {

KVCacheStorage::~KVCacheStorage()
{
}

bool KVCacheStorage::owns(const Mat&) const
{
    return false;
}

static bool is_valid_cache(const Mat& cache)
{
    return cache.data
           && cache.dims == 3
           && cache.w > 0 && cache.h >= 0 && cache.c > 0
           && cache.elemsize > 0 && cache.elempack > 0
           && cache.cstep >= (size_t)cache.w * cache.h;
}

NaiveKVCacheStorage::NaiveKVCacheStorage(Allocator* _allocator)
    : storage_type(0), allocator(_allocator)
#if NCNN_VULKAN
    ,
    vkallocator(0)
#endif
{
}

int NaiveKVCacheStorage::create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack)
{
    if (storage_type != 0 || !cache.empty() || seqlen < 0 || num_kv_head <= 0 || head_dim <= 0 || elemsize == 0 || elempack <= 0)
        return -1;

    const int physical_seqlen = seqlen > 0 ? seqlen : 1;
    Mat new_cache;
    new_cache.create(head_dim, physical_seqlen, num_kv_head, elemsize, elempack, allocator);
    if (new_cache.empty())
        return -100;

    new_cache.h = seqlen;
    cache = new_cache;

    return 0;
}

int NaiveKVCacheStorage::expand(const Mat& cache, Mat& expanded_cache, int new_seqlen)
{
    if (storage_type != 0 || !owns(cache) || !expanded_cache.empty() || new_seqlen < 0)
        return -1;

    const int old_seqlen = cache.h;
    if (new_seqlen <= old_seqlen)
    {
        expanded_cache = cache;
        expanded_cache.h = new_seqlen;
        return 0;
    }

    Mat new_cache;
    new_cache.create(cache.w, new_seqlen, cache.c, cache.elemsize, cache.elempack, allocator);
    if (new_cache.empty())
        return -100;

    const size_t valid_head_size = (size_t)cache.w * old_seqlen * cache.elemsize;
    for (int q = 0; q < cache.c; q++)
    {
        const unsigned char* src = (const unsigned char*)cache.data + cache.cstep * q * cache.elemsize;
        unsigned char* dst = (unsigned char*)new_cache.data + new_cache.cstep * q * new_cache.elemsize;
        memcpy(dst, src, valid_head_size);
    }

    expanded_cache = new_cache;

    return 0;
}

void NaiveKVCacheStorage::destroy(Mat& cache)
{
    if (storage_type == 0 && owns(cache))
        cache.release();
}

bool NaiveKVCacheStorage::owns(const Mat& cache) const
{
    return storage_type == 0 && is_valid_cache(cache);
}

class CPUKVCacheStoragePrivate;
class CPUKVCacheStorageAllocator : public Allocator
{
public:
    CPUKVCacheStorageAllocator(CPUKVCacheStoragePrivate* _d)
        : d(_d)
    {
    }

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    CPUKVCacheStoragePrivate* d;
};

class CPUKVCacheStoragePrivate
{
public:
    int max_seqlen_hint;
    int allocation_count;
    Allocator* allocator;
    CPUKVCacheStorageAllocator* storage_allocator;
};

void* CPUKVCacheStorageAllocator::fastMalloc(size_t size)
{
    void* ptr = d->allocator ? d->allocator->fastMalloc(size) : ncnn::fastMalloc(size);
    if (ptr)
        d->allocation_count++;

    return ptr;
}

void CPUKVCacheStorageAllocator::fastFree(void* ptr)
{
    d->allocation_count--;

    if (d->allocator)
        d->allocator->fastFree(ptr);
    else
        ncnn::fastFree(ptr);
}

CPUKVCacheStorage::CPUKVCacheStorage(int _max_seqlen_hint, Allocator* allocator)
    : d(new CPUKVCacheStoragePrivate)
{
    d->max_seqlen_hint = _max_seqlen_hint;
    d->allocation_count = 0;
    d->allocator = allocator;
    d->storage_allocator = new CPUKVCacheStorageAllocator(d);
}

CPUKVCacheStorage::~CPUKVCacheStorage()
{
    if (d->allocation_count != 0)
        NCNN_LOGE("FATAL ERROR! kvcache storage destroyed too early");

    delete d->storage_allocator;
    delete d;
}

static int kvcache_capacity(int seqlen, int max_seqlen_hint)
{
    int capacity = seqlen > 0 ? seqlen : 1;

    if (max_seqlen_hint >= seqlen && max_seqlen_hint > 0)
        return max_seqlen_hint;

    int reserve = capacity < 16 ? 16 - capacity : capacity;
    if (reserve > 256)
        reserve = 256;

    if (capacity <= INT_MAX - reserve)
        capacity += reserve;

    return capacity;
}

static int grow_kvcache_capacity(int capacity, int new_seqlen)
{
    while (capacity < new_seqlen)
    {
        int growth = capacity / 2;
        if (growth < 16)
            growth = 16;

        if (capacity > INT_MAX - growth)
            return new_seqlen;

        capacity += growth;
    }

    return capacity;
}

int CPUKVCacheStorage::create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack)
{
    if (!cache.empty() || seqlen < 0 || num_kv_head <= 0 || head_dim <= 0 || elemsize == 0 || elempack <= 0)
        return -1;

    const int capacity = kvcache_capacity(seqlen, d->max_seqlen_hint);

    Mat new_cache;
    new_cache.create(head_dim, capacity, num_kv_head, elemsize, elempack, d->storage_allocator);
    if (new_cache.empty())
        return -100;

    // h is the logical sequence length while cstep retains the physical capacity
    new_cache.h = seqlen;
    cache = new_cache;

    return 0;
}

int CPUKVCacheStorage::expand(const Mat& cache, Mat& expanded_cache, int new_seqlen)
{
    if (!expanded_cache.empty() || new_seqlen < 0)
        return -1;

    if (!owns(cache))
    {
        NCNN_LOGE("kvcache storage got foreign cache %p", cache.data);
        return -1;
    }

    const int old_seqlen = cache.h;
    const int capacity = (int)(cache.cstep / cache.w);
    if (new_seqlen <= capacity)
    {
        expanded_cache = cache;
        expanded_cache.h = new_seqlen;
        return 0;
    }

    const int new_capacity = grow_kvcache_capacity(capacity, new_seqlen);

    Mat new_cache;
    new_cache.create(cache.w, new_capacity, cache.c, cache.elemsize, cache.elempack, d->storage_allocator);
    if (new_cache.empty())
        return -100;

    new_cache.h = new_seqlen;

    const size_t valid_head_size = (size_t)cache.w * old_seqlen * cache.elemsize;
    for (int q = 0; q < cache.c; q++)
    {
        const unsigned char* src = (const unsigned char*)cache.data + cache.cstep * q * cache.elemsize;
        unsigned char* dst = (unsigned char*)new_cache.data + new_cache.cstep * q * new_cache.elemsize;
        memcpy(dst, src, valid_head_size);
    }

    expanded_cache = new_cache;

    return 0;
}

void CPUKVCacheStorage::destroy(Mat& cache)
{
    if (cache.empty())
        return;

    if (!owns(cache))
    {
        NCNN_LOGE("kvcache storage got foreign cache %p", cache.data);
        return;
    }

    cache.release();
}

bool CPUKVCacheStorage::owns(const Mat& cache) const
{
    return cache.allocator == d->storage_allocator && is_valid_cache(cache);
}

#if NCNN_VULKAN
bool KVCacheStorage::owns(const VkMat&) const
{
    return false;
}

int KVCacheStorage::create(VkMat&, int, int, int, size_t, int, VkCompute&)
{
    return -1;
}

int KVCacheStorage::expand(const VkMat&, VkMat&, int, VkCompute&)
{
    return -1;
}

void KVCacheStorage::destroy(VkMat&)
{
}

static bool is_valid_vkcache(const VkMat& cache)
{
    return cache.data
           && cache.dims == 3
           && cache.w > 0 && cache.h >= 0 && cache.c > 0
           && cache.elemsize > 0 && cache.elempack > 0
           && cache.cstep >= (size_t)cache.w * cache.h;
}

NaiveKVCacheStorage::NaiveKVCacheStorage(VkAllocator* _allocator)
    : storage_type(1), allocator(0), vkallocator(_allocator)
{
}

int NaiveKVCacheStorage::create(VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute&)
{
    if (storage_type != 1 || !vkallocator || !cache.empty() || seqlen < 0 || num_kv_head <= 0 || head_dim <= 0 || elemsize == 0 || elempack <= 0)
        return -1;

    const int physical_seqlen = seqlen > 0 ? seqlen : 1;
    VkMat new_cache;
    new_cache.create(head_dim, physical_seqlen, num_kv_head, elemsize, elempack, vkallocator);
    if (new_cache.empty())
        return -100;

    new_cache.h = seqlen;
    cache = new_cache;

    return 0;
}

int NaiveKVCacheStorage::expand(const VkMat& cache, VkMat& expanded_cache, int new_seqlen, VkCompute& cmd)
{
    if (storage_type != 1 || !owns(cache) || !expanded_cache.empty() || new_seqlen < 0)
        return -1;

    const int old_seqlen = cache.h;
    if (new_seqlen <= old_seqlen)
    {
        expanded_cache = cache;
        expanded_cache.h = new_seqlen;
        return 0;
    }

    VkMat new_cache;
    new_cache.create(cache.w, new_seqlen, cache.c, cache.elemsize, cache.elempack, vkallocator);
    if (new_cache.empty())
        return -100;

    new_cache.h = old_seqlen;

    if (old_seqlen > 0)
    {
        Option opt;
        opt.blob_vkallocator = new_cache.allocator;
        cmd.record_clone(cache, new_cache, opt);
    }

    new_cache.h = new_seqlen;
    expanded_cache = new_cache;

    return 0;
}

void NaiveKVCacheStorage::destroy(VkMat& cache)
{
    if (storage_type == 1 && owns(cache))
        cache.release();
}

bool NaiveKVCacheStorage::owns(const VkMat& cache) const
{
    return storage_type == 1 && is_valid_vkcache(cache);
}

class VkKVCacheStoragePrivate;
class VkKVCacheStorageAllocator : public VkAllocator
{
public:
    VkKVCacheStorageAllocator(const VulkanDevice* vkdev, VkKVCacheStoragePrivate* _d)
        : VkAllocator(vkdev), d(_d)
    {
    }

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkKVCacheStoragePrivate* d;
};

class VkKVCacheStoragePrivate
{
public:
    int max_seqlen_hint;
    int allocation_count;
    VkBlobAllocator* allocator;
    VkKVCacheStorageAllocator* storage_allocator;
};

VkBufferMemory* VkKVCacheStorageAllocator::fastMalloc(size_t size)
{
    VkBufferMemory* ptr = d->allocator->fastMalloc(size);
    if (ptr)
        d->allocation_count++;

    return ptr;
}

void VkKVCacheStorageAllocator::fastFree(VkBufferMemory* ptr)
{
    d->allocation_count--;
    d->allocator->fastFree(ptr);
}

VkImageMemory* VkKVCacheStorageAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    return d->allocator->fastMalloc(w, h, c, elemsize, elempack);
}

void VkKVCacheStorageAllocator::fastFree(VkImageMemory* ptr)
{
    d->allocator->fastFree(ptr);
}

VkKVCacheStorage::VkKVCacheStorage(const VulkanDevice* vkdev, int _max_seqlen_hint)
    : d(new VkKVCacheStoragePrivate)
{
    d->max_seqlen_hint = _max_seqlen_hint;
    d->allocation_count = 0;
    d->allocator = new VkBlobAllocator(vkdev);
    d->storage_allocator = new VkKVCacheStorageAllocator(vkdev, d);
    d->storage_allocator->buffer_memory_type_index = d->allocator->buffer_memory_type_index;
    d->storage_allocator->image_memory_type_index = d->allocator->image_memory_type_index;
    d->storage_allocator->reserved_type_index = d->allocator->reserved_type_index;
    d->storage_allocator->mappable = d->allocator->mappable;
    d->storage_allocator->coherent = d->allocator->coherent;
}

VkKVCacheStorage::~VkKVCacheStorage()
{
    if (d->allocation_count != 0)
        NCNN_LOGE("FATAL ERROR! vk kvcache storage destroyed too early");

    delete d->storage_allocator;
    delete d->allocator;
    delete d;
}

int VkKVCacheStorage::create(VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute&)
{
    if (!cache.empty() || seqlen < 0 || num_kv_head <= 0 || head_dim <= 0 || elemsize == 0 || elempack <= 0)
        return -1;

    const int capacity = kvcache_capacity(seqlen, d->max_seqlen_hint);

    VkMat new_cache;
    new_cache.create(head_dim, capacity, num_kv_head, elemsize, elempack, d->storage_allocator);
    if (new_cache.empty())
        return -100;

    new_cache.h = seqlen;
    cache = new_cache;

    return 0;
}

int VkKVCacheStorage::expand(const VkMat& cache, VkMat& expanded_cache, int new_seqlen, VkCompute& cmd)
{
    if (!expanded_cache.empty() || new_seqlen < 0)
        return -1;

    if (!owns(cache))
    {
        NCNN_LOGE("vk kvcache storage got foreign cache %p", cache.data);
        return -1;
    }

    const int old_seqlen = cache.h;
    const int capacity = (int)(cache.cstep / cache.w);
    if (new_seqlen <= capacity)
    {
        expanded_cache = cache;
        expanded_cache.h = new_seqlen;
        return 0;
    }

    const int new_capacity = grow_kvcache_capacity(capacity, new_seqlen);

    VkMat new_cache;
    new_cache.create(cache.w, new_capacity, cache.c, cache.elemsize, cache.elempack, d->storage_allocator);
    if (new_cache.empty())
        return -100;
    new_cache.h = old_seqlen;

    if (old_seqlen > 0)
    {
        Option opt;
        opt.blob_vkallocator = new_cache.allocator;
        cmd.record_clone(cache, new_cache, opt);
    }

    new_cache.h = new_seqlen;
    expanded_cache = new_cache;

    return 0;
}

void VkKVCacheStorage::destroy(VkMat& cache)
{
    if (cache.empty())
        return;

    if (!owns(cache))
    {
        NCNN_LOGE("vk kvcache storage got foreign cache %p", cache.data);
        return;
    }

    cache.release();
}

int VkKVCacheStorage::create(Mat&, int, int, int, size_t, int)
{
    return -1;
}

int VkKVCacheStorage::expand(const Mat&, Mat&, int)
{
    return -1;
}

void VkKVCacheStorage::destroy(Mat&)
{
}

bool VkKVCacheStorage::owns(const VkMat& cache) const
{
    return cache.allocator == d->storage_allocator && is_valid_vkcache(cache);
}

#endif // NCNN_VULKAN

} // namespace ncnn
