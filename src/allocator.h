// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_ALLOCATOR_H
#define NCNN_ALLOCATOR_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "platform.h"

#include <stdlib.h>

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

namespace ncnn {

// the alignment of all the allocated buffers
#if NCNN_AVX512
#define NCNN_MALLOC_ALIGN 64
#elif NCNN_AVX
#define NCNN_MALLOC_ALIGN 32
#else
#define NCNN_MALLOC_ALIGN 16
#endif

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instructions
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define NCNN_MALLOC_OVERREAD 64

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static NCNN_FORCEINLINE _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static NCNN_FORCEINLINE size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static NCNN_FORCEINLINE void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static NCNN_FORCEINLINE void fastFree(void* ptr)
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

#if NCNN_THREADS
// exchange-add operation for atomic operations on reference counters
#if defined __riscv && !defined __riscv_atomic
// riscv target without A extension
static NCNN_FORCEINLINE int NCNN_XADD(int* addr, int delta)
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
static NCNN_FORCEINLINE int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif
#else  // NCNN_THREADS
static NCNN_FORCEINLINE int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif // NCNN_THREADS

class NCNN_EXPORT Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocatorPrivate;
class NCNN_EXPORT PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0
    void set_size_compare_ratio(float scr);

    // budget drop threshold
    // default threshold = 10
    void set_size_drop_threshold(size_t);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    PoolAllocator(const PoolAllocator&);
    PoolAllocator& operator=(const PoolAllocator&);

private:
    PoolAllocatorPrivate* const d;
};

class UnlockedPoolAllocatorPrivate;
class NCNN_EXPORT UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0
    void set_size_compare_ratio(float scr);

    // budget drop threshold
    // default threshold = 10
    void set_size_drop_threshold(size_t);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    UnlockedPoolAllocator(const UnlockedPoolAllocator&);
    UnlockedPoolAllocator& operator=(const UnlockedPoolAllocator&);

private:
    UnlockedPoolAllocatorPrivate* const d;
};

#if NCNN_VULKAN

class VulkanDevice;

class NCNN_EXPORT VkBufferMemory
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

class NCNN_EXPORT VkImageMemory
{
public:
    VkImage image;
    VkImageView imageview;

    // underlying info assigned by allocator
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

class NCNN_EXPORT VkAllocator
{
public:
    explicit VkAllocator(const VulkanDevice* _vkdev);
    virtual ~VkAllocator();

    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size) = 0;
    virtual void fastFree(VkBufferMemory* ptr) = 0;
    virtual int flush(VkBufferMemory* ptr);
    virtual int invalidate(VkBufferMemory* ptr);

    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack) = 0;
    virtual void fastFree(VkImageMemory* ptr) = 0;

public:
    const VulkanDevice* vkdev;
    uint32_t buffer_memory_type_index;
    uint32_t image_memory_type_index;
    uint32_t reserved_type_index;
    bool mappable;
    bool coherent;

protected:
    VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
    VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
    VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer);

    VkImage create_image(int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
    VkImageView create_imageview(VkImage image, VkFormat format);
};

class VkBlobAllocatorPrivate;
class NCNN_EXPORT VkBlobAllocator : public VkAllocator
{
public:
    explicit VkBlobAllocator(const VulkanDevice* vkdev, size_t preferred_block_size = 16 * 1024 * 1024); // 16M
    virtual ~VkBlobAllocator();

public:
    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkBlobAllocator(const VkBlobAllocator&);
    VkBlobAllocator& operator=(const VkBlobAllocator&);

private:
    VkBlobAllocatorPrivate* const d;
};

class VkWeightAllocatorPrivate;
class NCNN_EXPORT VkWeightAllocator : public VkAllocator
{
public:
    explicit VkWeightAllocator(const VulkanDevice* vkdev, size_t preferred_block_size = 8 * 1024 * 1024); // 8M
    virtual ~VkWeightAllocator();

public:
    // release all blocks immediately
    virtual void clear();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkWeightAllocator(const VkWeightAllocator&);
    VkWeightAllocator& operator=(const VkWeightAllocator&);

private:
    VkWeightAllocatorPrivate* const d;
};

class VkStagingAllocatorPrivate;
class NCNN_EXPORT VkStagingAllocator : public VkAllocator
{
public:
    explicit VkStagingAllocator(const VulkanDevice* vkdev);
    virtual ~VkStagingAllocator();

public:
    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkStagingAllocator(const VkStagingAllocator&);
    VkStagingAllocator& operator=(const VkStagingAllocator&);

private:
    VkStagingAllocatorPrivate* const d;
};

class VkWeightStagingAllocatorPrivate;
class NCNN_EXPORT VkWeightStagingAllocator : public VkAllocator
{
public:
    explicit VkWeightStagingAllocator(const VulkanDevice* vkdev);
    virtual ~VkWeightStagingAllocator();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkWeightStagingAllocator(const VkWeightStagingAllocator&);
    VkWeightStagingAllocator& operator=(const VkWeightStagingAllocator&);

private:
    VkWeightStagingAllocatorPrivate* const d;
};

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
class NCNN_EXPORT VkAndroidHardwareBufferImageAllocator : public VkAllocator
{
public:
    VkAndroidHardwareBufferImageAllocator(const VulkanDevice* _vkdev, AHardwareBuffer* _hb);
    virtual ~VkAndroidHardwareBufferImageAllocator();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack);
    virtual void fastFree(VkImageMemory* ptr);

private:
    VkAndroidHardwareBufferImageAllocator(const VkAndroidHardwareBufferImageAllocator&);
    VkAndroidHardwareBufferImageAllocator& operator=(const VkAndroidHardwareBufferImageAllocator&);

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
#endif // NCNN_PLATFORM_API

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
