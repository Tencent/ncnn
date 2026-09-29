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

static int test_pool_allocator_clears_overread()
{
    const size_t size = 256;

    ncnn::PoolAllocator allocator;
    allocator.set_size_compare_ratio(0.f);

    void* ptr = allocator.fastMalloc(size);
    if (!ptr)
        return -1;

    // Poison payload + SIMD overread tail, then return to the pool.
    memset(ptr, 0xff, size + NCNN_MALLOC_OVERREAD);
    allocator.fastFree(ptr);

    void* ptr2 = allocator.fastMalloc(size);
    if (!ptr2)
        return -1;

    if (ptr2 != ptr)
    {
        fprintf(stderr, "expected recycled pointer\n");
        allocator.fastFree(ptr2);
        return -1;
    }

    const int ret = check_cleared(ptr2, size);
    allocator.fastFree(ptr2);
    return ret;
}

static int test_unlocked_pool_allocator_clears_overread()
{
    const size_t size = 128;

    ncnn::UnlockedPoolAllocator allocator;
    allocator.set_size_compare_ratio(0.f);

    void* ptr = allocator.fastMalloc(size);
    if (!ptr)
        return -1;

    memset(ptr, 0x7f, size + NCNN_MALLOC_OVERREAD);
    allocator.fastFree(ptr);

    void* ptr2 = allocator.fastMalloc(size);
    if (!ptr2)
        return -1;

    if (ptr2 != ptr)
    {
        fprintf(stderr, "expected recycled unlocked pointer\n");
        allocator.fastFree(ptr2);
        return -1;
    }

    const int ret = check_cleared(ptr2, size);
    allocator.fastFree(ptr2);
    return ret;
}

int main()
{
    return test_pool_allocator_clears_overread()
           || test_unlocked_pool_allocator_clears_overread();
}
