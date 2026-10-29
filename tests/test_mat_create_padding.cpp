// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"

#include <stdio.h>
#include <string.h>

class PoisonAllocator : public ncnn::Allocator
{
public:
    virtual void* fastMalloc(size_t size)
    {
        void* ptr = ncnn::fastMalloc(size);
        if (ptr)
            memset(ptr, 0xcd, size);
        return ptr;
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::fastFree(ptr);
    }
};

static int test_create_padding(int dims, size_t scalar_size, int elempack, int batch, bool packed, int width = 9)
{
    PoisonAllocator allocator;
    ncnn::Mat m;
    const size_t elemsize = scalar_size * elempack;
    if (packed && batch > 1)
    {
        if (dims == 1) m.create(width, elemsize, elempack, batch, &allocator);
        if (dims == 2) m.create(width, 3, elemsize, elempack, batch, &allocator);
        if (dims == 3) m.create(width, 3, 3, elemsize, elempack, batch, &allocator);
        if (dims == 4) m.create(width, 3, 5, 3, elemsize, elempack, batch, &allocator);
    }
    else if (packed)
    {
        if (dims == 1) m.create(width, elemsize, elempack, &allocator);
        if (dims == 2) m.create(width, 3, elemsize, elempack, &allocator);
        if (dims == 3) m.create(width, 3, 3, elemsize, elempack, &allocator);
        if (dims == 4) m.create(width, 3, 5, 3, elemsize, elempack, &allocator);
    }
    else
    {
        if (dims == 1) m.create(width, scalar_size, &allocator);
        if (dims == 2) m.create(width, 3, scalar_size, &allocator);
        if (dims == 3) m.create(width, 3, 3, scalar_size, &allocator);
        if (dims == 4) m.create(width, 3, 5, 3, scalar_size, &allocator);
    }
    if (m.empty() || !m.refcount || *m.refcount != 1)
        return -1;
#if NCNN_BATCH
    if (m.n != batch)
        return -1;
#endif

    const unsigned char* data = (const unsigned char*)m.data;
    const size_t payload = (size_t)m.w * m.h * m.d * m.elemsize;
    const size_t totalsize = (const unsigned char*)m.refcount - data;
    size_t cursor = 0;
    for (int b = 0; b < m.n; b++)
    {
        ncnn::Mat batch_view = m.batch(b);
        for (int q = 0; q < m.c; q++)
        {
            const unsigned char* channel = (const unsigned char*)batch_view.channel(q).data;
            const size_t begin = channel - data;
            while (cursor < begin)
            {
                if (data[cursor++] != 0)
                    return -1;
            }
            for (size_t i = 0; i < payload; i++)
            {
                if (data[cursor++] != 0xcd)
                {
                    fprintf(stderr, "payload cleared dims=%d size=%zu pack=%d batch=%d b=%d q=%d\n", dims, scalar_size, elempack, batch, b, q);
                    return -1;
                }
            }
        }
    }
    while (cursor < totalsize)
    {
        if (data[cursor++] != 0)
            return -1;
    }
    return 0;
}

int main()
{
    int failed = 0;
    int cases = 0;
    for (int dims = 1; dims <= 4; dims++)
    {
        for (size_t scalar_size = 1; scalar_size <= 4; scalar_size *= 2)
        {
            failed += test_create_padding(dims, scalar_size, 1, 1, false) != 0;
            cases++;
            for (int elempack = 1; elempack <= 4; elempack *= 4)
            {
                failed += test_create_padding(dims, scalar_size, elempack, 1, true) != 0;
                failed += test_create_padding(dims, scalar_size, elempack, 3, true) != 0;
                cases += 2;
            }
        }
    }
    failed += test_create_padding(1, 3, 1, 1, false, 1) != 0;
    failed += test_create_padding(1, 4, 1, 3, true, 1024) != 0;
    cases += 2;
    fprintf(stderr, "Mat create padding: %d/%d passed\n", cases - failed, cases);
    return failed ? 1 : 0;
}
