// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_KVCACHE_STORAGE_H
#define NCNN_KVCACHE_STORAGE_H

#include "allocator.h"
#include "mat.h"
#include "platform.h"

namespace ncnn {

#if NCNN_VULKAN
class VkCompute;
class VulkanDevice;
#endif

class NCNN_EXPORT KVCacheStorage
{
public:
    virtual ~KVCacheStorage();

    virtual int create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack) = 0;
    virtual int expand(const Mat& cache, Mat& expanded_cache, int new_seqlen) = 0;
    virtual void destroy(Mat& cache) = 0;
    virtual bool owns(const Mat& cache) const;

#if NCNN_VULKAN
    virtual int create(VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute& cmd);
    virtual int expand(const VkMat& cache, VkMat& expanded_cache, int new_seqlen, VkCompute& cmd);
    virtual void destroy(VkMat& cache);
    virtual bool owns(const VkMat& cache) const;
#endif
};

// naive ordinary Mat/VkMat storage for legacy kvcache usage
class NaiveKVCacheStorage : public KVCacheStorage
{
public:
    explicit NaiveKVCacheStorage(Allocator* allocator);
#if NCNN_VULKAN
    explicit NaiveKVCacheStorage(VkAllocator* allocator);
#endif

    virtual int create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack);
    virtual int expand(const Mat& cache, Mat& expanded_cache, int new_seqlen);
    virtual void destroy(Mat& cache);
    virtual bool owns(const Mat& cache) const;

#if NCNN_VULKAN
    virtual int create(VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute& cmd);
    virtual int expand(const VkMat& cache, VkMat& expanded_cache, int new_seqlen, VkCompute& cmd);
    virtual void destroy(VkMat& cache);
    virtual bool owns(const VkMat& cache) const;
#endif

private:
    int storage_type;
    Allocator* allocator;
#if NCNN_VULKAN
    VkAllocator* vkallocator;
#endif
};

#if NCNN_VULKAN
class VkKVCacheStoragePrivate;
class NCNN_EXPORT VkKVCacheStorage : public KVCacheStorage
{
public:
    explicit VkKVCacheStorage(const VulkanDevice* vkdev, int max_seqlen_hint = 0);
    virtual ~VkKVCacheStorage();

    virtual int create(VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute& cmd);
    virtual int expand(const VkMat& cache, VkMat& expanded_cache, int new_seqlen, VkCompute& cmd);
    virtual void destroy(VkMat& cache);
    virtual bool owns(const VkMat& cache) const;

private:
    virtual int create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack);
    virtual int expand(const Mat& cache, Mat& expanded_cache, int new_seqlen);
    virtual void destroy(Mat& cache);
    VkKVCacheStorage(const VkKVCacheStorage&);
    VkKVCacheStorage& operator=(const VkKVCacheStorage&);

private:
    VkKVCacheStoragePrivate* const d;
};
#endif // NCNN_VULKAN

class CPUKVCacheStoragePrivate;
class NCNN_EXPORT CPUKVCacheStorage : public KVCacheStorage
{
public:
    explicit CPUKVCacheStorage(int max_seqlen_hint = 0, Allocator* allocator = 0);
    virtual ~CPUKVCacheStorage();

    virtual int create(Mat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack);
    virtual int expand(const Mat& cache, Mat& expanded_cache, int new_seqlen);
    virtual void destroy(Mat& cache);
    virtual bool owns(const Mat& cache) const;

private:
    CPUKVCacheStorage(const CPUKVCacheStorage&);
    CPUKVCacheStorage& operator=(const CPUKVCacheStorage&);

private:
    CPUKVCacheStoragePrivate* const d;
};

} // namespace ncnn

#endif // NCNN_KVCACHE_STORAGE_H
