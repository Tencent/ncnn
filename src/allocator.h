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

#include "platform.h"

#include <list>
#include <stdlib.h>

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26

namespace ncnn {

#if __AVX__
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 256
#else
// the alignment of all the allocated buffers
#define MALLOC_ALIGN 16
#endif

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, MALLOC_ALIGN, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(MALLOC_ALIGN, size);
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
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __riscv && !defined __riscv_atomic
// riscv target without A extension
static inline int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#elif defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#ifdef __ATOMIC_ACQ_REL
#define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#else
#define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#endif
#else
#if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#else
#define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#endif
#endif
#elif defined _MSC_VER && !defined RC_INVOKED
#define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe branch
static inline int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif

class Allocator
{
public:
    virtual ~Allocator();
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
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
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
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

#if NCNN_VULKAN

class VulkanDevice;

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
    mutable VkAccessFlags access_flags;
    mutable VkPipelineStageFlags stage_flags;

    // initialize and modified by mat
    int refcount;
};

class VkImageMemory
{
public:
    VkImage image;
    VkImageView imageview;

    // underlying info assigned by allocator
    VkImageType image_type;
    VkImageViewType imageview_type;
    int width;
    int height;
    int depth;
    VkFormat format;

    VkDeviceMemory memory;
    void* mapped_ptr;

    // the base offset assigned by allocator
    size_t bind_offset;
    size_t bind_capacity;

    // image state, modified by command functions internally
    mutable VkAccessFlags access_flags;
    mutable VkImageLayout image_layout;
    mutable VkPipelineStageFlags stage_flags;

    // in-execution state, modified by command functions internally
    mutable int command_refcount;

    // initialize and modified by mat
    int refcount;
};

class VkAllocator
{
public:
    VkAllocator(const VulkanDevice* _vkdev);
    virtual ~VkAllocator()
    {
        clear();
    }
    virtual void clear()
    {
    }

    virtual VkBufferMemory* fastMalloc(size_t size) = 0;
    virtual void fastFree(VkBufferMemory* ptr) = 0;
    virtual int flush(VkBufferMemory* ptr);
    virtual int invalidate(VkBufferMemory* ptr);

    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack) = 0;
    virtual void fastFree(VkImageMemory* ptr) = 0;

public:
    const VulkanDevice* vkdev;
    uint32_t buffer_memory_type_index;
    uint32_t image_memory_type_index;
    bool mappable;
    bool coherent;

protected:
    VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
    VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
    VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer);

    VkImage create_image(VkImageType type, int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
    VkImageView create_imageview(VkImageViewType type, VkImage image, VkFormat format);
};

class VkBlobAllocator : public VkAllocator
{
public:
    VkBlobAllocator(const VulkanDevice* vkdev);
    virtual ~VkBlobAllocator();

public:
    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

protected:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<std::list<std::pair<size_t, size_t> > > buffer_budgets;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<std::list<std::pair<size_t, size_t> > > image_memory_budgets;
    std::vector<VkDeviceMemory> image_memory_blocks;
};

class VkWeightAllocator : public VkAllocator
{
public:
    VkWeightAllocator(const VulkanDevice* vkdev);
    virtual ~VkWeightAllocator();

public:
    // release all blocks immediately
    virtual void clear();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

protected:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<size_t> buffer_block_free_spaces;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<VkBufferMemory*> dedicated_buffer_blocks;
    std::vector<size_t> image_memory_block_free_spaces;
    std::vector<VkDeviceMemory> image_memory_blocks;
    std::vector<VkDeviceMemory> dedicated_image_memory_blocks;
};

class VkStagingAllocator : public VkAllocator
{
public:
    VkStagingAllocator(const VulkanDevice* vkdev);
    virtual ~VkStagingAllocator();

public:
    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

protected:
    unsigned int size_compare_ratio; // 0~256
    std::list<VkBufferMemory*> buffer_budgets;
};

class VkWeightStagingAllocator : public VkAllocator
{
public:
    VkWeightStagingAllocator(const VulkanDevice* vkdev);
    virtual ~VkWeightStagingAllocator();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int /*dims*/, int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/)
    {
        return 0;
    }
    virtual void fastFree(VkImageMemory* /*ptr*/)
    {
    }

protected:
};

#if __ANDROID_API__ >= 26
class ImportAndroidHardwareBufferPipeline;
class VkAndroidHardwareBufferImageAllocator : public VkAllocator
{
public:
    VkAndroidHardwareBufferImageAllocator(const VulkanDevice* _vkdev, AHardwareBuffer* _hb);
    virtual ~VkAndroidHardwareBufferImageAllocator();

public:
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);
    virtual VkBufferMemory* fastMalloc(size_t /*size*/)
    {
        return 0;
    }
    virtual void fastFree(VkBufferMemory* /*ptr*/)
    {
    }

public:
    int init();

    int width() const;
    int height() const;
    uint64_t external_format() const;

public:
    AHardwareBuffer* hb;
    AHardwareBuffer_Desc bufferDesc;
    VkAndroidHardwareBufferFormatPropertiesANDROID bufferFormatProperties;
    VkAndroidHardwareBufferPropertiesANDROID bufferProperties;
    VkSamplerYcbcrConversionKHR samplerYcbcrConversion;
};
#endif // __ANDROID_API__ >= 26

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
