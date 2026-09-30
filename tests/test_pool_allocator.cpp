// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "allocator.h"

#include <stdio.h>
#include <string.h>

static int check_cleared(void* ptr, size_t size)
{
    const unsigned char* p = (const unsigned char*)ptr;
    for (size_t i = 0; i < size + NCNN_MALLOC_OVERREAD; i++)
    {
        if (p[i] != 0)
        {
            fprintf(stderr, "byte %zu not cleared (got 0x%02x)\n", i, p[i]);
            return -1;
        }
    }
    return 0;
}

template<typename AllocatorType>
static int test_pool_allocator_clears_overread(size_t size, size_t reused_size)
{
    AllocatorType allocator;
    allocator.set_size_compare_ratio(0.f);

    void* ptr = allocator.fastMalloc(size);
    if (!ptr)
        return -1;

    const int fresh_ret = check_cleared(ptr, size);
    if (fresh_ret != 0)
    {
        allocator.fastFree(ptr);
        return fresh_ret;
    }

    // Poison payload + SIMD overread tail, then return to the pool.
    memset(ptr, 0xff, size + NCNN_MALLOC_OVERREAD);
    allocator.fastFree(ptr);

    void* ptr2 = allocator.fastMalloc(reused_size);
    if (!ptr2)
        return -1;

    if (ptr2 != ptr)
    {
        fprintf(stderr, "expected recycled pointer\n");
        allocator.fastFree(ptr2);
        return -1;
    }

    // The pool may reuse a larger block. Check its entire capacity and tail,
    // including bytes beyond the smaller request's overread range.
    const int ret = check_cleared(ptr2, size);
    allocator.fastFree(ptr2);
    return ret;
}

int main()
{
    const size_t sizes[] = {1, 3, 15, 16, 17, 127, 128, 129, 255, 256, 257, 4096};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
    {
        const size_t size = sizes[i];
        if (test_pool_allocator_clears_overread<ncnn::PoolAllocator>(size, size)
                || test_pool_allocator_clears_overread<ncnn::UnlockedPoolAllocator>(size, size)
                || test_pool_allocator_clears_overread<ncnn::PoolAllocator>(size, (size + 1) / 2)
                || test_pool_allocator_clears_overread<ncnn::UnlockedPoolAllocator>(size, (size + 1) / 2))
        {
            fprintf(stderr, "test_pool_allocator_clears_overread failed size=%zu\n", size);
            return -1;
        }
    }
    return 0;
}
