// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_ALLOCATOR_H
#define NCNN_ALLOCATOR_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdlib.h>
#include <list>
#include <vector>
#include "platform.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#include "gpu.h"
#endif // NCNN_VULKAN

namespace ncnn {

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif __ANDROID__
    return memalign(MALLOC_ALIGN, size);
#elif _POSIX_C_SOURCE >= 200112L
    void* ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif __ANDROID__
        free(ptr);
#elif _POSIX_C_SOURCE >= 200112L
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe branch
static inline int NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

#ifdef _WIN32
class Mutex
{
public:
    Mutex() { InitializeSRWLock(&srwlock); }
    ~Mutex() {}
    void lock() { AcquireSRWLockExclusive(&srwlock); }
    void unlock() { ReleaseSRWLockExclusive(&srwlock); }
private:
    // NOTE SRWLock is available from windows vista
    SRWLOCK srwlock;
};
#else // _WIN32
class Mutex
{
public:
    Mutex() { pthread_mutex_init(&mutex, 0); }
    ~Mutex() { pthread_mutex_destroy(&mutex); }
    void lock() { pthread_mutex_lock(&mutex); }
    void unlock() { pthread_mutex_unlock(&mutex); }
private:
    pthread_mutex_t mutex;
};
#endif // _WIN32

class MutexLockGuard
{
public:
    MutexLockGuard(Mutex& _mutex) : mutex(_mutex) { mutex.lock(); }
    ~MutexLockGuard() { mutex.unlock(); }
private:
    Mutex& mutex;
};

class Allocator
{
public:
    virtual ~Allocator() = 0;
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio;// 0~256
    std::list< std::pair<size_t, void*> > budgets;
    std::list< std::pair<size_t, void*> > payouts;
};

class UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    unsigned int size_compare_ratio;// 0~256
    std::list< std::pair<size_t, void*> > budgets;
    std::list< std::pair<size_t, void*> > payouts;
};

#if NCNN_VULKAN

class VkBufferMemory
{
public:
    VkBuffer buffer;

    // the base offset assigned by allocator
    size_t offset;
    size_t capacity;

    VkDeviceMemory memory;
    void* mapped_ptr;

    // buffer state, modified by command functions internally
    // 0=null
    // 1=created
    // 2=transfer
    // 3=compute
    // 4=readonly
    mutable int state;

    // initialize and modified by mat
    int refcount;
};

class VkAllocator
{
public:
    VkAllocator(const VulkanDevice* _vkdev);
    virtual ~VkAllocator() { clear(); }
    virtual void clear() {}
    virtual VkBufferMemory* fastMalloc(size_t size) = 0;
    virtual void fastFree(VkBufferMemory* ptr) = 0;

public:
    const VulkanDevice* vkdev;
    bool mappable;

protected:
    VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
    VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
    VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer);
};

class VkUnlockedBlobBufferAllocator : public VkAllocator
{
public:
    VkUnlockedBlobBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkUnlockedBlobBufferAllocator();

public:
    // buffer block size, default=16M
    void set_block_size(size_t size);

    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    size_t block_size;
    size_t buffer_offset_alignment;
    std::vector< std::list< std::pair<size_t, size_t> > > budgets;
    std::vector<VkBufferMemory*> buffer_blocks;
};

class VkBlobBufferAllocator : public VkUnlockedBlobBufferAllocator
{
public:
    VkBlobBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkBlobBufferAllocator();

public:
    virtual void clear();
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    Mutex budgets_lock;
};

class VkWeightBufferAllocator : public VkAllocator
{
public:
    VkWeightBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkWeightBufferAllocator();

public:
    // buffer block size, default=8M
    void set_block_size(size_t block_size);

    // release all blocks immediately
    virtual void clear();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    size_t block_size;
    size_t buffer_offset_alignment;
    std::vector<size_t> buffer_block_free_spaces;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<VkBufferMemory*> dedicated_buffer_blocks;
};

class VkUnlockedStagingBufferAllocator : public VkAllocator
{
public:
    VkUnlockedStagingBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkUnlockedStagingBufferAllocator();

public:
    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    uint32_t memory_type_index;
    unsigned int size_compare_ratio;// 0~256
    std::list<VkBufferMemory*> budgets;
};

class VkStagingBufferAllocator : public VkUnlockedStagingBufferAllocator
{
public:
    VkStagingBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkStagingBufferAllocator();

public:
    virtual void clear();
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    Mutex budgets_lock;
};

class VkWeightStagingBufferAllocator : public VkAllocator
{
public:
    VkWeightStagingBufferAllocator(const VulkanDevice* vkdev);
    virtual ~VkWeightStagingBufferAllocator();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

private:
    uint32_t memory_type_index;
};

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
